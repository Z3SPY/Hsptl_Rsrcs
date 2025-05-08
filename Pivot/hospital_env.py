import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from model_classes import Scenario, TreatmentCentreModel, CustomResource, SimulationSummary
import hospital_env_helpers as helpers

class HospitalSimEnv(gym.Env):
    """
    Hospital ED simulation environment for RL with shift-level decisions.
    Uses MSO baseline plus small delta actions and a potential-based, multi-objective reward.
    """
    metadata = {'render.modes': []}

    def __init__(self, sim_config, shift_duration=480, rc_period=10080):
        super().__init__()
        if not isinstance(sim_config, Scenario):
            raise TypeError("sim_config must be a Scenario object")
        self.scenario       = sim_config   # keep the real object
        self.shift_duration = shift_duration
        self.rc_period      = rc_period
        self.current_time   = 0.0
        self.prev_discharged= 0
        self.episode_idx    = 0

        self.logger       = helpers.EventLogger(enable_console=False)
        self.fatigue_mgr  = helpers.FatigueManager(step_size=self.shift_duration)
        self.shift_logger = helpers.CSVShiftLogger(path='shift_metrics.csv')

        self.resource_units = [
            ('triage',       'n_triage'),
            ('registration', 'n_reg'),
            ('exam',         'n_exam'),
            ('trauma',       'n_trauma'),
            ('cubicle_1',    'n_cubicles_1'),
            ('cubicle_2',    'n_cubicles_2'),
            ('ward_beds',    'n_ward_beds'),
            ('icu_beds',     'n_icu_beds'),
        ]

        max_delta = 2
        action_dims = [(2 * max_delta + 1) for _ in self.resource_units]
        self.action_space = spaces.MultiDiscrete(action_dims)
        self._max_delta   = max_delta

        obs_dim = 3 + 2*len(self.resource_units) + len(self.resource_units)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _init_model(self):
        self.model             = TreatmentCentreModel(self.scenario, logger=self.logger)
        self.model.rc_period   = self.rc_period
        self.model.sim_summary = SimulationSummary(self.model)
        self.model.env.process(self.model.arrivals_generator())

    def reset(self):
        self.episode_idx += 1
        if self.episode_idx < 300:
            #-self.sim_config['override_arrival_rate'] = True           
            # self.sim_config['manual_arrival_rate']   = 60.0
            self.scenario.override_arrival_rate = True
            self.scenario.manual_arrival_rate   = 60.0
        else:
            self.scenario.override_arrival_rate = False 

        self.current_time    = 0.0
        self.prev_discharged = 0

        self._init_model()
        self.logger.log(None, 'Env', 'reset', 'env_reset', self.current_time)

        raw_plan = self.mso_shift_planner()
        self.latest_mso_plan = np.array(raw_plan, dtype=int)
        summary = SimulationSummary(self.model)
        summary.process_run_results_live()
        self._last_queue_total = sum(
            summary.queue_lengths.get(u,0)
            for u,_ in self.resource_units
        )

        return self._get_observation()

    def step(self, action):
        baseline = self.latest_mso_plan
        max_caps = np.array([getattr(self.scenario, p) for _, p in self.resource_units], dtype=int)
        deltas = action.astype(int) - self._max_delta
        counts = np.clip(baseline + deltas, 0, max_caps)

        for idx, (unit, _) in enumerate(self.resource_units):
            store = getattr(self.model.args, unit)
            if hasattr(store, "items"):
                store.items.clear()
                for i in range(int(counts[idx])):
                    store.put(CustomResource(
                        self.model.env, capacity=1, id_attribute=i
                    ))

        next_time = self.current_time + self.shift_duration
        self.model.env.run(until=next_time)
        self.current_time = next_time

        resources = {
            unit: getattr(self.model.args, unit).items
            if hasattr(getattr(self.model.args, unit), 'items') else []
            for unit,_ in self.resource_units
        }
        active_ids = {
            unit: {r.id_attribute for r in items if getattr(r,'busy',False)}
            for unit,items in resources.items()
        }
        resting = {
            unit: getattr(self.scenario, p) - len(items)
            for (unit,p),items in zip(self.resource_units, resources.values())
        }
        self.fatigue_mgr.update(resources, active_ids, resting)

        summary = SimulationSummary(self.model)
        summary.process_run_results_live()
        shift_discharges = summary.total_discharged - self.prev_discharged
        self.prev_discharged = summary.total_discharged

        queue_start = self._last_queue_total
        queue_end   = sum(
            summary.queue_lengths.get(u,0) for u,_ in self.resource_units
        )
        max_queue   = max(sum(getattr(self.scenario, p) for _, p in self.resource_units), 1)
        phi_change  = (queue_start - queue_end) / max_queue
        self._last_queue_total = queue_end

        throughput_norm = shift_discharges / max(sum(baseline),1)
        util_mismatch   = np.mean([
            abs(summary.utilization.get(u,0.0) - 0.85)
            for u,_ in self.resource_units
        ])
        unit_fats       = self._average_fatigue_by_unit()
        avg_fatigue     = float(np.mean(list(unit_fats.values()))) if unit_fats else 0.0
        fatigue_norm    = avg_fatigue / 100.0

        avg_wait = summary.avg_total_wait_time
        wait_penalty = min(avg_wait / 60.0, 2.0)

        raw = (
            + 1.0 * throughput_norm
            + 1.0 * phi_change
            - 0.3 * util_mismatch
            - 0.2 * fatigue_norm
            - 0.3 * wait_penalty          
        )
        clip_val = max(10.0, 100.0 * (0.98 ** self.episode_idx))
        reward   = np.clip(raw, -clip_val, clip_val)

        self.logger.log(None, 'Env', 'periodic', 'shift_summary', self.current_time,
                        throughput=shift_discharges,
                        phi_change=phi_change,
                        util_mismatch=util_mismatch,
                        fatigue_norm=fatigue_norm,
                        reward=reward)
        self.shift_logger.log(self.current_time, {
            'throughput':      shift_discharges,
            'avg_wait':        summary.avg_total_wait_time,
            'wait_penalty':     wait_penalty,
            'avg_fatigue':     avg_fatigue,
            'queue_shaping':   phi_change,
            'util_penalty':    util_mismatch,
            'fatigue_penalty': fatigue_norm,
            'reward':          reward
        })

        obs  = self._get_observation()
        done = (self.current_time >= self.rc_period)
        info = {'shift_discharges': shift_discharges}

        print(f"DEBUG ACTION at time {self.current_time}: {action}, OBS: {obs}, REWARD: {reward}")

        self.model.full_event_log = self.logger.full_event_log  # ✅ Copy logs from logger


        return obs, reward, done, info

    def _get_observation(self):
        summary = SimulationSummary(self.model)
        summary.process_run_results_live()

        obs = [
            summary.avg_total_wait_time,
            summary.avg_fatigue,
            self.prev_discharged
        ]
        for (unit, param) in self.resource_units:
            util = summary.utilization.get(unit, 0.0)
            qlen = summary.queue_lengths.get(unit, 0)
            cap  = getattr(self.scenario, param, 1)
            q_pres = qlen / cap if cap > 0 else 0.0
            obs += [util, q_pres]

        norm_baseline = self.latest_mso_plan / np.maximum(
            [getattr(self.scenario, p) for _, p in self.resource_units], 1
        )
        obs += list(norm_baseline)
        return np.array(obs, dtype=np.float32)

    def _average_fatigue_by_unit(self):
        fatigues = {}
        for unit, param in self.resource_units:
            store = getattr(self.model.args, unit, None)
            if hasattr(store, 'items') and store.items:
                values = [getattr(res, "fatigue", 0.0) for res in store.items]
                fatigues[unit] = np.mean(values)
            else:
                fatigues[unit] = 0.0
        return fatigues

    def mso_shift_planner(self):
        return [getattr(self.scenario, p) for _, p in self.resource_units]

    def render(self, mode='human'):
        pass

    def close(self):
        pass
