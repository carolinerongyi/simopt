"""
Summary
-------
Simulate a M/M/1 server queue with non-homogeneous service rate 
and non-homogeneous arrival rates.
A detailed description of the model/problem can be found
TODO.
"""
# import numpy as np

# from ..base import Model, Problem

import autograd.numpy as np
from ..base import Auto_Model, Problem
from ..auto_diff_util_h import factor_dict, resp_dict_to_array, replicate_wrapper# WHERE


class FickleServer2(Auto_Model):
    """
    A model that simulates an M/M/1 queue with a (nonhomogeneous) 
    Exponential(lambda) interarrival time distribution and a 
    (nonhomogeneous) Exponential(x) service time distribution. 
    Returns
        - Total amount of service rate (in discrete sense, the number of total servers used)
        - Fractions of late calls of each period
    for customers after a warmup period.

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
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None,random =False):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "MM1"
        self.n_rngs = 3
        self.random = random
        self.n_random = 2  # Number of rng used for the random instance -- todo?
        # random instance factors: number_queues, arrival_alphas, service_mus, routing_matrix
        self.n_responses = 2
        self.response_names = ['EL1', 'EL2', 'EL3', 'EL4', 'EL5', 'EL6','arrival_counts', "late_calls"]
        
        self.specifications = {
            "T": {
                "description": "simulation length",
                "datatype": int,
                "default": 18 # in hours
            },
            "N": {
                "description": "number of time periods",
                "datatype": int,
                "default": 6
            },
            "lambdas": {
                "description": "rate parameter of interarrival time distribution",
                "datatype": list,
                "default": [20, 30, 60, 60, 30, 20] # [1/20, 1/30, 1/60, 1/60, 1/30, 1/20] # interarrival time in hours
            },
            "mus":{
                "description": "rate parameter of servicee time distribution",
                "datatype": list,
                "default": [1,2,3,0.5,1.5,1.5]
            },
            # "mu1": {
            #     "description": "rate parameter of service time distribution",
            #     "datatype": float,
            #     "default": 1.5
            # },
            # "mu2": {
            #     "description": "rate parameter of service time distribution",
            #     "datatype": float,
            #     "default": 1.5
            # },
            # "mu3": {
            #     "description": "rate parameter of service time distribution",
            #     "datatype": float,
            #     "default": 60
            # },
            # "mu4": {
            #     "description": "rate parameter of service time distribution",
            #     "datatype": float,
            #     "default": 60
            # },
            # "mu5": {
            #     "description": "rate parameter of service time distribution",
            #     "datatype": float,
            #     "default": 1.5
            # },
            # "mu6": {
            #     "description": "rate parameter of service time distribution",
            #     "datatype": float,
            #     "default": 1.5
            # },
            "late_threshold": {
                "description": "threshold for late calls",
                "datatype": float,
                "default": 1/180 # 20 seconds
            }
        }
        self.check_factor_list = {
            "T": self.check_T,
            "N": self.check_N,
            "lambdas": self.check_lambdas,
            "mus": self.check_mus,
            # "mu1" : self.check_mu1,
            # "mu2" : self.check_mu2,
            # "mu3" : self.check_mu3,
            # "mu4" : self.check_mu4,
            # "mu5" : self.check_mu5,
            # "mu6" : self.check_mu6,
            "late_threshold": self.check_late_threshold
        }
        # self.response_names = ['arrival_counts', "late_calls"]
        # for i in range(self.specifications('N')):
        #     self.response_names.append("EL" + str(i+1))
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
        
    def check_T(self):
        return self.factors["T"] > 0
    
    def check_N(self):
        return self.factors["N"] > 0

    def check_lambdas(self):
        return all(lambd > 0 for lambd in self.factors["lambdas"])

    def check_mus(self):
        return all(mu > 0 for mu in self.factors["mus"])

    # def check_mu1(self):
    #     return self.factors["mu1"] > 0
    
    # def check_mu2(self):
    #     return self.factors["mu2"] > 0
    
    # def check_mu3(self):
    #     return self.factors["mu3"] > 0
    
    # def check_mu4(self):
    #     return self.factors["mu4"] > 0
    
    # def check_mu5(self):
    #     return self.factors["mu5"] > 0
    
    # def check_mu6(self):
    #     return self.factors["mu6"] > 0

    def check_late_threshold(self):
        return self.factors["late_threshold"] >= 0


    def _replicate(self, diff_factors, rng_list, response_names):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "" = 
            "" = 
            "" = 
        gradients : dict of dicts
            gradient estimates for each response
        """
        factors = factor_dict(self, diff_factors)
        
        # Unpack factors
        T, N = factors["T"], factors["N"]
        
        # Designate separate RNGs for interarrival and serivce times.
        arrival_rng = rng_list[0]
        service_rng = rng_list[1]
        
        # Generate all interarrival and service times up front.
        arrival_times = []
        interarrivals = []
        t = 0
        lambda_star = max(factors["lambdas"])
        while t < T:
            if t > 0 and u <= factors["lambdas"][int(t//(T/N))] / lambda_star:
                arrival_times.append(t)
            interarrival = arrival_rng.expovariate(lambda_star)
            t += interarrival
            interarrivals.append(interarrival)
            u = arrival_rng.random() # uniform(0, 1)
        
        # Generate service times for each customer.
        service_times = []
        arr_count = [0] * N
        for arr_time in arrival_times:
            # Case 1: service would finish within a period
            time_period = int(arr_time//(T/N))
            # service_time = service_rng.expovariate(factors["mu"+str(time_period + 1)])
            service_time = service_rng.expovariate(factors["mus"][(time_period)])
            # Case 2: service would finish after T
            if arr_time + service_time >= T:
                service_time = T - arr_time
            # Case 3: service time would cross a period boundary
            elif arr_time + service_time > (int(arr_time/(T/N)) + 1) * int(T/N):
                # amount_finished = factors["mu"+str(time_period + 1)] * ((int(arr_time/(T/N)) + 1) * (T/N) - (arr_time + service_time))
                amount_finished = factors["mus"][(time_period)] * ((int(arr_time/(T/N)) + 1) * (T/N) - (arr_time + service_time))
                amount_left = service_rng.expovariate(1) - amount_finished 
                # service_time_left = amount_left / factors["mu"+str(time_period + 2)] ---------------------------
                service_time_left = amount_left / factors["mus"][(time_period + 1)]
                service_time = service_rng.expovariate(factors["mus"][(time_period )])
                service_time = (int(arr_time/(T/N)) + 1) * (T/N) - arr_time + service_time_left
            service_times.append(service_time)
            
            # Count the number of arrivals in each period
            arr_count[int(arr_time//(T/N))] += 1 
            
        # Count the number of late calls in each period
        late_calls = [0] * N
        EL = [0] * N
        wait_time = [0] * len(arrival_times)
        wait_time[0] = 0
        
        for i in range(len(arrival_times) - 1):
            
            # print(wait_time)
            # print("arrival:", i, "wait time:", wait_time[i])
            
            finish_time = arrival_times[i] + wait_time[i] + service_times[i]
            wait_time[i+1] = finish_time - arrival_times[i+1] if finish_time > arrival_times[i+1] else 0 
            
            if wait_time[i+1] > factors["late_threshold"]:
                late_calls[int(arrival_times[i]//(T/N))] += 1 
                
            y = factors["late_threshold"] + interarrivals[i] - wait_time[i]
            if y < 0:
                ELi = 1 
            else:
                # ELi = np.exp(-1/(y * factors["mu" + str(int(arrival_times[i]//(T/N))+1)]))
                ELi = np.exp(-1/(y * factors["mus"][(int(arrival_times[i]//(T/N)))]))
            EL[int(arrival_times[i]//(T/N))] += ELi

        # EL = [x for x in EL]
        
        # responses = {'late_calls': late_calls, "EL": EL, 'arrival_counts': arr_count}
        # gradient = {}
        responses = {}
        for i in range(N):
            responses['EL' + str(i+1)] = EL[i]
        responses['arrival_counts'] = arr_count
        responses['late_calls'] = late_calls
        # return responses, gradient
        return resp_dict_to_array(self, responses, response_names)

    def replicate(self, rng_list, **kwargs):
       return replicate_wrapper(self, rng_list, **kwargs)
"""
Summary
-------
Minimize the total service rate of the M/M/1 queue with nonhomogeneous arrival
rates and service rates.
"""


class FickleServerMinServiceRate2(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="MM1-1", fixed_factors=None, model_fixed_factors=None, random = False):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name

        self.random = random
        self.n_random = 2  # Number of rng used for the random instance -- todo? --hagen
        # random instance factors: number_queues, arrival_alphas, service_mus, routing_matrix

        self.dim = 6 # For now, we will only consider 6 periods ---- hagen changed
        # self.dim = 18 # For now, we will only consider 6 periods
        self.n_objectives = 1
        self.n_stochastic_constraints = 6 # 1 for each period  ##Hagen changedproblem dim to 6
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "continuous"
        self.lower_bounds = tuple([0] * self.dim)
        self.upper_bounds = tuple([np.inf] * self.dim)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"mus"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution from which solvers start",
                "datatype": tuple,
                "default": tuple([60,10,20,20,30,30])
                # "default": tuple([10,10,20,20,30,30,60,60,80,80,60,60,30,30,20,20,10,10])
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000
            },
            "upper_thres": {
                "description": "upper limit of amount of contamination",
                "datatype": list,
                # "default": tuple([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                "default": tuple([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = FickleServer2(self.model_fixed_factors)

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "mus": vector
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = (factor_dict["mus"],)
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        sum = 0
        for i in range(self.model.factors["N"]):
            sum = sum + response_dict['EL' + str(i+1)]

        print(response_dict.keys())
        objectives = (sum / np.sum(response_dict["arrival_counts"]), )
        # objectives = (np.sum(response_dict["late_calls"]) / np.sum(response_dict["arrival_counts"]), )
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        # under_control = response_dict["late_percentage"] <= self.factors["upper_thres"]\
        lhs = []
        for i in range(len(response_dict["late_calls"])):
            lhs.append(response_dict["late_calls"][i] - self.factors["upper_thres"][i] * response_dict["arrival_counts"][i])
        stoch_constraints = tuple(lhs)
        print(stoch_constraints, type(stoch_constraints))
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim, )
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic
            constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = tuple([0] * self.dim)
        det_stoch_constraints_gradients = (0,)
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        # Check box constraints.
        return np.all(x >= 0) & np.all(x <= 1)

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple([rand_sol_rng.random() * 50 for _ in range(self.dim)])
        return x