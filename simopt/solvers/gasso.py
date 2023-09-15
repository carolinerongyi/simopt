"""
Summary
-------
Randomly sample solutions from the feasible region.
Can handle stochastic constraints.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/randomsearch.html>`_.
"""
from ..base import Solver
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class GASSO(Solver):
    """
    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """
    def __init__(self, name="GASSO", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box" # ??
        self.variable_type = "continuous"
        self.gradient_needed = True
        self.specifications = {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "sample_size": {
                "description": "sample size per solution",
                "datatype": int,
                "default": 10
            },
            "max_iter": {
                "description": "maximum number of iterations",
                "datatype": int,
                "default": 10000
            },
            "rho": {
                "description": "quantile parameter",
                "datatype": float,
                "default": 0.15
            },
            "M": {
                "description": "times of simulations for each candidate solution",
                "datatype": int,
                "default": 15
            },
            "alpha_0": {
                "description": "step size numerator",
                "datatype": int,
                "default": 15
            },
            "alpha_c": {
                "description": "step size denominator constant",
                "datatype": int,
                "default": 150
            },
            "alpha_p": {
                "description": "step size denominator exponent",
                "datatype": float,
                "default": 0.6
            },
            "MaxNumSoln": {
                "description": "maximum number of solutions that can be reported within max budget",
                "datatype": int,
                "default": 10002
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "sample_size": self.check_sample_size,
            "max_iter": self.check_max_iter,
            "rho": self.check_rho,
            "M": self.check_M,
            "alpha_0": self.check_alpha_0,
            "alpha_c": self.check_alpha_c,
            "alpha_p": self.check_alpha_p,
            "MaxNumSoln": self.check_MaxNumSoln
        }
        super().__init__(fixed_factors)

    def check_sample_size(self):
        return self.factors["sample_size"] > 0
    
    def check_max_iter(self):
        return self.factors["max_iter"] > 0
    
    def check_rho(self):
        return 0 < self.factors["rho"] < 1
    
    def check_M(self):
        return self.factors["M"] > 0
    
    def check_alpha_0(self):
        return self.factors["alpha_0"] > 0
    
    def check_alpha_c(self):
        return self.factors["alpha_c"] > 0
    
    def check_alpha_p(self):
        return 0 < self.factors["alpha_p"] < 1
    
    def check_MaxNumSoln(self):
        return self.factors["MaxNumSoln"] > 0
    
    def solve(self, problem): # Initialize
        Ancalls = np.zeros(MaxNumSoln, 1)
        A = np.zeros(MaxNumSoln, dim)
        AFnMean = np.zeros(MaxNumSoln, 1)
        AFnVar = np.zeros(MaxNumSoln, 1)
        
        Ancalls[0] = 0
        A[0, :] = problem.factors["initial_solution"]
        AFnMean[0] = None
        AFnVar[0] = None

        mu_k = np.mean(problem.factors["initial_solution"])
        var_k = np.var(problem.factors["initial_solution"])
        theta1_k = mu_k/var_k
        theta2_k = -0.5 * np.ones(problem.dim) / var_k
        theta_k = [theta1_k, theta2_k]
        dim = problem.dim
        N = int(50 * np.sqrt(dim))
        K = np.floor(problem.factors['budget']/(N * self.factors['M']))
        MaxNumSoln = K + 2
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        
        # Get random solutions from normal distribution (truncated)
        x = np.zeros(N, dim)
        kk = 0
        while kk < N:
            X_k = find_next_soln_rng.random(N, dim) * np.diag(np.sqrt(var_k)) + np.ones(N, 1) * mu_k
            for i in range(N):
                if all(X_k[i, :] >= problem.lb) and all(X_k[i, :] <= problem.ub):
                    x[kk, :] = X_k[i, :]
                    kk += 1
        X_k = x
        
        
        
        
        

    # def solve(self, problem):
    #     """
    #     Run a single macroreplication of a solver on a problem.

    #     Arguments
    #     ---------
    #     problem : Problem object
    #         simulation-optimization problem to solve
    #     crn_across_solns : bool
    #         indicates if CRN are used when simulating different solutions

    #     Returns
    #     -------
    #     recommended_solns : list of Solution objects
    #         list of solutions recommended throughout the budget
    #     intermediate_budgets : list of ints
    #         list of intermediate budgets when recommended solutions changes
    #     """
    #     recommended_solns = []
    #     intermediate_budgets = []
    #     expended_budget = 0
    #     # Designate random number generator for random sampling.
    #     find_next_soln_rng = self.rng_list[1]
    #     # Sequentially generate random solutions and simulate them.
    #     while expended_budget < problem.factors["budget"]:
    #         if expended_budget == 0:
    #             # Start at initial solution and record as best.
    #             new_x = problem.factors["initial_solution"]
    #             new_solution = self.create_new_solution(new_x, problem)
    #             best_solution = new_solution
    #             recommended_solns.append(new_solution)
    #             intermediate_budgets.append(expended_budget)
    #         else:
    #             # Identify new solution to simulate.
    #             new_x = problem.get_random_solution(find_next_soln_rng)
    #             new_solution = self.create_new_solution(new_x, problem)
    #         # Simulate new solution and update budget.
    #         problem.simulate(new_solution, self.factors["sample_size"])
    #         expended_budget += self.factors["sample_size"]
    #         # Check for improvement relative to incumbent best solution.
    #         # Also check for feasibility w.r.t. stochastic constraints.
    #         if (problem.minmax * new_solution.objectives_mean
    #                 > problem.minmax * best_solution.objectives_mean and
    #                 all(new_solution.stoch_constraints_mean[idx] <= 0 for idx in range(problem.n_stochastic_constraints))):
    #             # If better, record incumbent solution as best.
    #             best_solution = new_solution
    #             recommended_solns.append(new_solution)
    #             intermediate_budgets.append(expended_budget)
    #     return recommended_solns, intermediate_budgets