"""
Summary
-------
Simulate an open jackson network 
"""
import numpy as np

from ..base import Model, Problem

class ExampleModel(Model):
    """
    A model of an open jackson network .

    Attributes
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "OPENJACKSON"
        self.n_rngs = 2
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "number_queues": {
                "description": "The number of queues in the network",
                "datatype": int,
                "default": 5
            },
            "arrival_alphas": {
                "description": "The arrival rates to each queue from outside the network",
                "datatype": list,
                "default": [3,3,3,3,3]
            },
            # "service_mus": {
            #     "description": "The mu values for the exponential service times ",
            #     "datatype": list,
            #     "default": [6,6,6,6,6]
            # },
            "routing_matrix": {
                "description": "The routing matrix that describes the probabilities of moving to the next queue after leaving the current one",
                "datatype": list,
                "default": [[0.1, 0.1, 0.2, 0.2, 0],
                            [0.1, 0.1, 0.2, 0.2, 0],
                            [0.1, 0.1, 0, 0.1, 0.3],
                            [0.1, 0.1, 0.1, 0, 0.3],
                            [0.1, 0.1, 0.1, 0.1, 0.2]]
            },
            # "departure_probabilities": {
            #     "description": "The probabilities that the job leaves the network after the service",
            #     "datatype": list,
            #     "default": [0.4,0.4,0.4,0.4,0.4]
            # },
            "t_end": {
                "description": "A number of replications to run",
                "datatype": list,
                "default": 500
            },
            "warm_up": {
                "description": "A number of replications to use as a warm up period",
                "datatype": int,
                "default": 50
            },
            "service_rates_capacity": {
                "description": "An upper bound on the total service rates",
                "datatype": list,
                "default": 100
            },
            
        }
        self.check_factor_list = {
            "number_queues": self.check_number_queues,
            "arrival_alphas": self.check_arrival_alphas,
            "routing_matrix": self.check_routing_matrix,
            # "departure_probabilities": self.check_departure_probabilities,
            "t_end": self.check_t_end,
            "warm_up": self.check_warm_up,
            "service_rates_capacity": self.check_service_rates_capacity
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_number_queues(self):
        return self.factors["number_queues"]>=0
    def check_arrival_alphas(self):
        return self.factors["arrival_alphas"]>=0
    def check_routing_matrix(self):
        transition_sums = list(map(sum, self.factors["routing_matrix"]))
        if all([len(row) == len(self.factors["routing_matrix"]) for row in self.factors["routing_matrix"]]) & \
                all(transition_sums[i] <= 1 for i in range(self.factors["number_queues"])):
            return True
        else:
            return False
    # def check_departure_probabilities(self):
        # if len(self.factors["departure_probabilities"]) != self.factors["number_queues"]:
        #     return False
        # else:
        #     return all([0 <= prob <= 1 for prob in self.factors["departure_probabilities"]])
    def check_t_end(self):
        return self.factors["t_end"] >= 0
    def check_warm_up(self):
        # Assume f(x) can be evaluated at any x in R^d.
        return self.factors["warm_up"] >= 0
    def check_service_rates_capacity(self):
        # Assume f(x) can be evaluated at any x in R^d.
        return self.factors["service_rates_capacity"] >= 0
    # def check_x(self):
    #     # Assume f(x) can be evaluated at any x in R^d.
    #     return True


    def check_simulatable_factors(self):
        return True

    def replicate(self, rng_list):
        """
        Evaluate a deterministic function f(x) with stochastic noise.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "est_f(x)" = f(x) evaluated with stochastic noise
        """
        # Designate random number generator for stochastic noise.
        noise_rng = rng_list[0]
        x = np.array(self.factors["x"])
        fn_eval_at_x = np.linalg.norm(x) ** 2 + noise_rng.normalvariate()

        # Compose responses and gradients.
        responses = {"est_f(x)": fn_eval_at_x}
        gradients = {"est_f(x)": {"x": tuple(2 * x)}}
        return responses, gradients