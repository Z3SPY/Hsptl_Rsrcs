import math
import pulp

class HospitalMSOPlanner:
    """
    Enhanced Multistage Stochastic Optimization (MSO) planner for hospital resource allocation.
    This version incorporates:
      - Estimation based on expected arrivals and stay durations (as before).
      - An additional penalty term that pulls the recommended ICU bed allocation toward a provided current ICU occupancy.
      - Separate weights for ICU and MedSurg mismatches.
    """
    def __init__(self, total_beds, p_icu, p_medsurg, mean_icu_stay, mean_medsurg_stay, horizon_hours=24):
        """
        :param total_beds: Total number of beds available (ICU + MedSurg).
        :param p_icu: Estimated probability a patient will require ICU after ED.
        :param p_medsurg: Estimated probability a patient will require MedSurg after ED.
        :param mean_icu_stay: Average ICU stay duration (in minutes).
        :param mean_medsurg_stay: Average MedSurg stay duration.
        :param horizon_hours: Planning horizon for MSO (default 24h).
        """
        self.total_beds = total_beds
        self.p_icu = p_icu
        self.p_medsurg = p_medsurg
        self.mean_icu_stay = mean_icu_stay
        self.mean_medsurg_stay = mean_medsurg_stay
        self.horizon = horizon_hours

        # Default weights for penalties in the objective function:
        self.w1 = 1.0  # weight for ICU mismatch
        self.w2 = 1.0  # weight for MedSurg mismatch
        self.w3 = 2.0  # weight for deviation from current ICU occupancy (if provided)

    def plan_allocation(self, current_state=None):
        """
        Compute an optimal bed allocation (ICU vs MedSurg) for the next planning horizon.
        If current_state is provided (expected to be a dict with key "icu_occupancy"), an extra penalty term is added.
        Returns a dictionary with recommended ICU and MedSurg bed counts.
        """
        # Estimate demand based on expected arrivals and average stay durations.
        expected_arrivals = self._estimate_arrivals(self.horizon)
        exp_icu_demand = expected_arrivals * self.p_icu * (self.mean_icu_stay / self.horizon)
        exp_ms_demand = expected_arrivals * self.p_medsurg * (self.mean_medsurg_stay / self.horizon)

        # Decision variable: number of ICU beds to allocate (MedSurg beds = total_beds - icu_beds)
        icu_beds = pulp.LpVariable('icu_beds', lowBound=0, upBound=self.total_beds, cat='Integer')

        # Auxiliary variables to linearize absolute differences
        # z1: absolute difference between expected ICU demand and allocated ICU beds
        z1 = pulp.LpVariable('z1', lowBound=0, cat='Continuous')
        # z2: absolute difference between expected MedSurg demand and allocated MedSurg beds
        z2 = pulp.LpVariable('z2', lowBound=0, cat='Continuous')

        # If current_state is provided, incorporate the absolute difference from current ICU occupancy.
        if current_state and "icu_occupancy" in current_state:
            current_icu = current_state["icu_occupancy"]
            z_current = pulp.LpVariable('z_current', lowBound=0, cat='Continuous')
        else:
            z_current = None

        # Define the linear programming problem
        prob = pulp.LpProblem("BedAllocationPlan", pulp.LpMinimize)

        # Objective: minimize weighted sum of mismatches.
        # Base cost: mismatch for ICU and MedSurg.
        objective = self.w1 * z1 + self.w2 * z2
        if z_current is not None:
            # Add extra cost for deviation from current ICU occupancy.
            objective += self.w3 * z_current

        prob += objective

        # Constraints for absolute difference of ICU beds:
        prob += exp_icu_demand - icu_beds <= z1
        prob += icu_beds - exp_icu_demand <= z1

        # For MedSurg beds:
        med_beds = self.total_beds - icu_beds
        prob += exp_ms_demand - med_beds <= z2
        prob += med_beds - exp_ms_demand <= z2

        # If current_state is provided, add constraints for deviation from current ICU occupancy:
        if z_current is not None:
            prob += icu_beds - current_icu <= z_current
            prob += current_icu - icu_beds <= z_current

        # Solve the problem using the default solver (CBC)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        optimal_icu = int(max(0, min(self.total_beds, pulp.value(icu_beds))))

        plan = {
            "icu_beds": optimal_icu,
            "medsurg_beds": self.total_beds - optimal_icu
        }
        return plan

    def _estimate_arrivals(self, hours):
        # Placeholder: assume an average of 12 arrivals per hour.
        return 12 * hours
