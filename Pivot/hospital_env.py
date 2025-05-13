import gym
import numpy as np
from gym import spaces
from model_classes import CustomResource, TreatmentCentreModel, SimulationSummary, Scenario
from hospital_env_helpers import EventLogger, FatigueManager
import simpy


class DictToBoxAction(gym.ActionWrapper):
    """
    Wraps a Dict action space (MultiDiscrete + 2 floats) into a flat Box vector.
    Compatible with PPO. Each staffing unit gets a value in [0.0, 1.0], scaled to 0–max_delta.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.n_units = len(env.resource_units)
        self.max_delta = env._max_delta

        # Flattened action space for SB3 PPO: roster (n_units), push, deferral
        self.action_space = spaces.Box(
            low=np.zeros(self.n_units + 2, dtype=np.float32),
            high=np.ones(self.n_units + 2, dtype=np.float32),
            dtype=np.float32
        )

    def action(self, action_vec):
        action_vec = np.clip(action_vec, 0.0, 1.0).astype(np.float32)
        roster_levels = np.rint(action_vec[:self.n_units] * self.max_delta).astype(np.int32)
        push          = float(action_vec[-2])
        defer         = float(action_vec[-1])
        return {
            "roster": roster_levels,
            "push": push,
            "deferral": defer
        }




class HospitalSimEnv(gym.Env):
    def __init__(self, sim_config, shift_duration=480, rc_period=1440*3, inject_resources=False):
        super().__init__()
        self.scenario = Scenario(**sim_config) if isinstance(sim_config, dict) else sim_config
        self.scenario.rc_period = getattr(self.scenario, "rc_period", rc_period)

        self.shift_duration = shift_duration
        self.rc_period = rc_period
        self.inject_resources = inject_resources 
        self._max_delta = 3

        # Dynamic staff-controlled units
        self.resource_units = [
            ("triage", "n_triage"),
            ("reg", "n_reg"),
            ("exam", "n_exam"),
            ("trauma", "n_trauma"),
            ("cubicles_1", "n_cubicles_1"),
            ("cubicles_2", "n_cubicles_2"),
        ]

        # Static capacity-controlled units
        self.static_units = [("ward", "n_ward_beds"), ("icu", "n_icu_beds")]

        # Maximum caps (can be used for rescaling or observation)
        self._max_capacity = {
            "n_triage": 6,
            "n_reg": 6,
            "n_exam": 10,
            "n_trauma": 10,
            "n_cubicles_1": 10,
            "n_cubicles_2": 10,
            "n_ward_beds": 51,
            "n_icu_beds": 13,
        }

        # --- ACTION SPACE: MultiBinary + 2 flow levers ---
        n_units = len(self.resource_units)
        self.action_space = spaces.Dict({
            "roster": spaces.MultiDiscrete([self._max_delta + 1] * n_units),
            "push":   spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "deferral": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })

        # --- OBSERVATION SPACE (unchanged) ---
        total_units = len(self.resource_units) + len(self.static_units)  # 6 + 2
        obs_dim = 5 + total_units * 3 + len(self.resource_units) * 2 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Staff cost setup ---
        self.staff_unit_cost = {
            "triage": 50, "reg": 45, "exam": 60,
            "trauma": 70, "cubicles_1": 55, "cubicles_2": 65
        }
        self.base_counts = {
            "triage": self.scenario.n_triage,
            "reg": self.scenario.n_reg,
            "exam": self.scenario.n_exam,
            "trauma": self.scenario.n_trauma,
            "cubicles_1": self.scenario.n_cubicles_1,
            "cubicles_2": self.scenario.n_cubicles_2,
        }

        self.logger = EventLogger()
        self.fatigue_mgr = FatigueManager(step_size=shift_duration)
        self.episode_idx = 0
        self.persistent_staff = {}
        self.last_action = np.zeros(n_units, dtype=np.int32)

        self.reset()


    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def _staff_cost(self, counts_dynamic):
        cost = 0.0
        for (unit, _), n in zip(self.resource_units, counts_dynamic):
            base = self.base_counts.get(unit, 0)
            reg = min(n, base)
            ot = max(0, n - base)

            cost += self.staff_unit_cost.get(unit, 0) * reg
            cost += self.staff_unit_cost.get(unit, 0) * ot * 1.5  # Overtime penalty
        return cost / 100.0  # Normalized


    def reset(self):
        # Reset logging & fatigue
        self.logger.reset()
        self.fatigue_mgr.reset()
        for staff_list in self.persistent_staff.values():
            for res in staff_list:
                res.fatigue = 0.0
        self.episode_idx += 1

        # Randomize base staffing by ±1 to introduce variability
        for unit, param in self.resource_units:
            base = getattr(self.scenario, param)
            delta = np.random.choice([-1, 0, 1])
            new_base = int(np.clip(base + delta, 1, self._max_capacity[param]))
            setattr(self.scenario, param, new_base)
            self.base_counts[unit] = new_base

        self.current_time = 0
        self.prev_discharged = 0
        self._last_queue_total = 0
        self.last_action = np.zeros(len(self.resource_units), dtype=int)

        self.scenario.init_nspp()
        self.model = TreatmentCentreModel(self.scenario)
        self.model.env.process(self.model.arrivals_generator())
        self.model.env.process(self.model.fatigue_sampler())
        self.persistent_staff = {}

        if self.inject_resources:
            for unit, param in self.resource_units:
                staff = [CustomResource(self.model.env, capacity=1, id_attribute=i)
                         for i in range(self._max_capacity[param])]
                self.persistent_staff[unit] = staff
                store = simpy.Store(self.model.env)
                for res in staff:
                    store.put(res)
                setattr(self.scenario, unit, store)
        else:
            for unit, param in self.resource_units:
                self.persistent_staff[unit] = [CustomResource(self.model.env, capacity=1, id_attribute=i)
                        for i in range(self._max_capacity[param])]

        return self._get_observation()

    



    def step(self, action):
        # ─── Extract action dictionary ────────────────────────────────
        roster_levels   = action["roster"]                      # MultiDiscrete: 0–_max_delta per unit
        discharge_push  = float(np.clip(action["push"], 0.0, 1.0))
        icu_deferral    = float(np.clip(action["deferral"], 0.0, 1.0))

        self.model.env.icu_deferral_prob = icu_deferral
        self.last_discharge_push = discharge_push
        self.last_icu_deferral   = icu_deferral

        # ─── Compute counts based on roster levels ──────────────────────
        counts_dynamic = []
        for i, (unit, prop) in enumerate(self.resource_units):
            base = self.base_counts[unit]
            n    = base + int(roster_levels[i])
            counts_dynamic.append(min(n, self._max_capacity[prop]))

        counts_static = [getattr(self.scenario, p) for _, p in self.static_units]

        full_resource_counts = dict(zip(
            [unit for unit, _ in self.resource_units + self.static_units],
            list(counts_dynamic) + list(counts_static)
        ))

        # ─── Apply on_shift logic ─────────────────────────────────────
        for idx, (unit, _) in enumerate(self.resource_units):
            target = int(counts_dynamic[idx])
            for i, res in enumerate(self.persistent_staff[unit]):
                res.on_shift = i < target
                if not res.on_shift:
                    res.busy = False

        # ─── Early-discharge: force ward/ICU releases ─────────────────
        if discharge_push > 0.1:
            to_free = int(round(discharge_push * 2))  # up to 2 beds/shift
            freed = 0
            while freed < to_free and self.model.args.ward_beds.store.items:
                self.model.args.ward_beds.store.items.pop()
                freed += 1
            while freed < to_free and self.model.args.icu_beds.store.items:
                self.model.args.icu_beds.store.items.pop()
                freed += 1

        # ─── Advance SimPy simulation ─────────────────────────────────
        next_time = self.current_time + self.shift_duration
        self.model.env.run(until=next_time)
        self.current_time = next_time

        # ─── Fatigue update (post-shift) ──────────────────────────────
        resources = self.persistent_staff
        active_ids = {
            unit: {res.id_attribute for res in resources[unit] if getattr(res, 'busy', False)}
            for unit in resources
        }
        resting = {
            unit: len(resources[unit]) - len(active_ids[unit])
            for unit in resources
        }
        self.fatigue_mgr.update(resources, active_ids, resting)

        # ─── Store last action for smoothness penalty ─────────────────
        if not hasattr(self, 'last_action'):
            self.last_action = np.zeros_like(roster_levels)
        self.smoothness_penalty = np.linalg.norm(self.last_action - roster_levels)
        self.last_action = roster_levels.copy()

        # ─── Extract summary metrics ─────────────────────────────────
        summary = SimulationSummary(self.model)
        summary.process_run_results_live()

        fatigues = []
        for unit, res_list in self.persistent_staff.items():
            for res in res_list:
                if getattr(res, 'on_shift', False):
                    fatigues.append(res.fatigue)
        summary.avg_fatigue = np.mean(fatigues) if fatigues else 0.0

        # ─── Build final observation, reward, done, info ──────────────
        return self._build_step(summary, counts_dynamic)


        



    def _build_step(self, summary, counts_dynamic):
        shift_discharges = summary.total_discharged - getattr(self, 'prev_discharged', 0)
        self.prev_discharged = summary.total_discharged

        queue_end = sum(summary.queue_lengths.get(u, 0) for u, _ in self.resource_units)
        queue_change = queue_end - getattr(self, '_last_queue_total', 0)
        self._last_queue_total = queue_end

        staff_cost = self._staff_cost(counts_dynamic)
        
        reward, wait_pen, ed_q_pen, fat_pen, smoothness_penalty, sla_bonus = \
            self._compute_reward(summary, shift_discharges, staff_cost, self.smoothness_penalty)

        if not np.isfinite(reward):
            print('[WARNING] Invalid reward encountered. Replacing with 0.')
            reward = 0.0

        obs = self._get_observation(summary, queue_end, queue_change)
        done = self.current_time >= self.rc_period

        if self.episode_idx % 10 == 0:
            print(f"[DEBUG] Step {self.episode_idx} | t={self.current_time:.0f} | Reward={reward:.2f}")
            print(f"   Roster: {self.last_action.tolist()} | Obs[:10]: {obs[:10]}")
            print(f"   AvgWait={summary.avg_total_wait_time:.1f}, AvgFatigue={summary.avg_fatigue:.6f}, Q={queue_end}")

        info = {
            'reward': reward,
            'discharges': shift_discharges,
            'wait_penalty': wait_pen,
            'queue_penalty': ed_q_pen,
            'fatigue_penalty': fat_pen,
            'cost_penalty': staff_cost,
            'smooth_penalty': smoothness_penalty,
            'sla_bonus': sla_bonus,
            'icu_q': summary.queue_lengths.get('icu', 0),
            'ward_q': summary.queue_lengths.get('ward', 0)
        }
        return obs, reward, done, info

    

    def _compute_reward(self, summary, shift_disch, staff_cost, smoothness_penalty):
        wait_pen   = summary.avg_total_wait_time / 30.0
        ed_q       = sum(summary.queue_lengths.get(u, 0) for u, _ in self.resource_units)
        ed_q_pen   = ed_q / 30.0
        bed_q_pen  = (summary.queue_lengths.get('ward', 0) + summary.queue_lengths.get('icu', 0)) / 20.0
        # fatigue penalty: base + nonlinear extra when high
        fat_base   = summary.avg_fatigue / 100.0
        extra_fat  = max(0.0, summary.avg_fatigue - 6.0)
        fat_pen    = fat_base + (extra_fat ** 1.2) / 10.0
        # staff cost weight halved
        cost_pen   = 0.15 * staff_cost

        push_pen   = getattr(self, 'last_discharge_push', 0.0)

        sla_bonus        = 0.5 if summary.avg_total_wait_time < 30 else 0.0
        discharge_reward = 0.2 * shift_disch

        queue_reduction_bonus = 0.0
        if hasattr(self, 'prev_queue') and ed_q < self.prev_queue:
            queue_reduction_bonus = 0.1 * np.sum(self.last_action)
        self.prev_queue = ed_q

        raw = (
            + discharge_reward
            - 0.6 * wait_pen
            - 0.4 * ed_q_pen
            - 0.3 * bed_q_pen
            - fat_pen
            - cost_pen
            - 0.2 * push_pen
            - 0.1 * smoothness_penalty
            + sla_bonus
            + queue_reduction_bonus
        )

        reward = float(raw) - 2.0 
        return reward, wait_pen, ed_q_pen + bed_q_pen, fat_pen, smoothness_penalty, sla_bonus











    # hospital_env.py, inside class HospitalSimEnv



    def _get_observation(self, summary=None, queue_end=None, queue_change=None):
        if summary is None:
            summary = SimulationSummary(self.model)
            summary.process_run_results_live()
            queue_end = sum(summary.queue_lengths.get(u, 0) for u, _ in self.resource_units)
            queue_change = queue_end - self._last_queue_total

        obs = [
            summary.avg_total_wait_time,
            summary.avg_fatigue,
            self.prev_discharged,
            queue_end,
            queue_change
        ]

        for (unit, param) in self.resource_units + self.static_units:
            util = summary.utilization.get(unit, 0.0)
            qlen = summary.queue_lengths.get(unit, 0)
            cap = getattr(self.scenario, param, 1)
            qprop = qlen / cap if cap else 0.0
            util_dev = util - 0.85
            obs.extend([util, qprop, util_dev])

        norm_baseline = np.array([getattr(self.scenario, p) for _, p in self.resource_units])
        norm_baseline = norm_baseline / np.maximum(norm_baseline.max(), 1)
        norm_deltas = self.last_action / self._max_delta

        obs.extend(norm_baseline.tolist())
        obs.extend(norm_deltas.tolist())

        icu_q_norm  = min(summary.queue_lengths.get('icu', 0), 10) / 10.0
        ward_q_norm = min(summary.queue_lengths.get('ward', 0), 10) / 10.0
        obs.extend([icu_q_norm, ward_q_norm])


        obs = np.array(obs, dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            print('[WARNING] Invalid observation detected. Replacing NaNs/Infs with 0.')
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs
