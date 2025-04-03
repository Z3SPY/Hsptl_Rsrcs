# File: sim_planner_env.py

from env_hospital import HospitalEnv

class SimPlannerEnv(HospitalEnv):
    """
    A stripped-down environment for MCTS planning. 
    Key differences from HospitalEnv:
     - 'planning_mode' always True => doesn't call MSO in step()
     - Skips heavy background processes or logging if desired
     - Minimizes side-effects by overriding run/step as needed
    """
    def __init__(self, scenario):
        # Initialize with base environment but no MSO override
        super().__init__(
            scenario=scenario,
            use_mso=False,
            mso_planner=None, 
            debug_logs=False
        )

        # Hardcode planning_mode to True 
        # so that 'env.step()' won't re-trigger planner
        self.planning_mode = True  

        # Optionally skip background processes
        self.skip_background_processes = True

    def reset(self):
        """
        Overridden to skip or reduce heavy setup. 
        Or we can just call super().reset() but keep planning_mode = True.
        """
        obs = super().reset()

        # If skipping background processes:
        if self.skip_background_processes:
            # we can remove them from WardFlowModel
            self.model.background_processes = []
        self.planning_mode = True
        return obs

    def step(self, action):
        """
        Overridden step so it never calls best_action from the planner 
        or runs heavy logs if not needed.
        """
        # Force planning_mode
        self.planning_mode = True

        # We do exactly what HospitalEnv step does except skip the MSO override
        return super().step(action)

    def run(self, rc_period):
        """
        If you want to skip heavy nurse scheduling, arrivals, etc. 
        you can override. For short MCTS rollouts, we often 
        just do a minimal run or do run( until= now + 60.0 ) inside step.
        """
        if self.skip_background_processes:
            # do nothing or run minimal
            self.model.env.run(until=rc_period)
        else:
            super().run(rc_period)

    def snapshot_state(self):
        """
        Potentially a simpler snapshot 
        if you don't want heavy logs or event data.
        """
        snap = {
            'time': self.model.env.now,
            'icu': self.current_icu_beds,
            'medsurg': self.current_medsurg_beds,
            'nurse_cost': self.nurse_change_cost,
        }
        return snap

    def restore_state(self, snap):
        self.model.env._now = snap['time']
        self.model.env.now = snap['time']
        self.current_icu_beds = snap['icu']
        self.current_medsurg_beds = snap['medsurg']
        self.nurse_change_cost = snap['nurse_cost']
