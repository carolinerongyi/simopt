"""
Summary
-------
Simulate an open jackson network 
"""
import numpy as np
import math as math

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
                "default": [1,1,1,1,1]
            },
            "service_mus": {
                "description": "The mu values for the exponential service times ",
                "datatype": list,
                "default": [2,2,2,2,2]
            },
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
            "service_rates_capacity": self.check_service_rates_capacity,
            "feasibility" : self.check_feasibility
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_feasibility(self):
        return (np.invert(np.identity(self.factor["number_queues"]) - self.factors["routing_matrix"].T) 
                * self.factors["arrival_alphas"] < self.factors["service_mus"])

    def check_number_queues(self):
        return self.factors["number_queues"]>=0
    def check_arrival_alphas(self):
        return self.factors["arrival_alphas"]>=0
    def check_service_mus(self):
        return self.factors["service_mus"]>=0
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
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "average_queue_length": The time-average of queue length at each station
        """
        # Designate random number generators.
        arrival_rng = rng_list[0]
        transition_rng = rng_list[1]
        time_rng = rng_list[2]

        # Initiate clock variables for statistics tracking and event handling.
        clock = 0
        previous_clock = 0
        next_arrival = arrival_rng.expovariate(sum(self.factors["arrival_alphas"]))

        # initialize list of queues
        stations = range(self.factors["number_attractions"])

        # create list of each station's next completion time and initialize to infinity.
        completion_times = [math.inf for _ in range(self.factors["number_attractions"])]

        # initialize list of each station's average queue length
        time_sum_queue_length = [0 for _ in range(self.factors["number_queues"])]
        
        # initialize the queue at each station
        queue = [0 for _ in range(self.factors["number_queues"])]

        # create external arrival probabilities for each attraction.
        arrival_probabalities = [self.factors["arrival_alphas"][i] / sum(self.factors["arrival_alphas"]) for i in
                                 self.factors["arrival_alphas"]]

        # # initialize time average queue length
        # in_system = 0
        # time_average = 0
        # cumulative_util = [0 for _ in range(self.factors["number_attractions"])]

        # Run simulation over time horizon.
        while min(next_arrival, min(completion_times)) < self.factors["t_end"]:
            # Count number of customers on attractions and in queues
            clock = min(next_arrival, min(completion_times))
            for i in range(self.factors["number_queues"]):
                time_sum_queue_length[i] += queue[i] * (clock - previous_clock)
            
    def replicate_steady_state(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication
            uses geometric rng

        Returns
        -------
        responses : dict
            performance measures of interest
            "average_queue_length": The time-average of queue length at each station
        """    
        #calculate lambdas
        lambdas = []
        for j in range(self.factors["number_queues"]):
            lambdas[j] = (self.factors["arrival_alphas"] + 
                          sum(self.factors["routing_matrix"][i][j] * self.factors["service_mus"][i] for i in self.factors["number_queues"]))
        #calculate rho variables for geometric
        rho = lambdas/self.factors["service"]
        #calculate expected value of queue length as rho/(1-rho)
        expected_queue_length = rho/(1-rho)

        return expected_queue_length