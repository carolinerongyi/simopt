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
        self.constraint_type = "box" 
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
    
    def get_random_sols(self, problem, n, rand_sol_rng):
        lbs, ubs = problem.lower_bounds, problem.upper_bounds
        xs = []
        for j in range(n):
            x = []
            for i in range(problem.dim):
                x.append(rand_sol_rng.uniform(lbs[i], ubs[i]))
            xs.append(tuple(x))
        
        return xs
   
    def solve(self, problem): # Initialize
        dim = problem.dim
        # Ancalls = np.zeros(MaxNumSoln)
        # A = np.zeros(MaxNumSoln, dim)
        # AFnMean = np.zeros(MaxNumSoln)
        # AFnVar = np.zeros(MaxNumSoln)
        
        # A = problem.factors["initial_solution"]
        # AFnMean[0] = None
        # AFnVar[0] = None

        rand_sol_rng = self.rng_list[0]
        x_ini = [problem.get_random_solution(rand_sol_rng) for i in range(19)]
        # x_ini = self.get_random_sols(problem, 20, rand_sol_rng)
        x_ini.append(problem.factors["initial_solution"])

        mu_k = np.mean(x_ini, axis = 0)  # SHould be the mean for each dim!!!
        var_k = np.var(x_ini, axis = 0)  # Currently this is 0!!! --> should be a vector!!! var for each dim
        theta1_k = mu_k/var_k
        theta2_k = -0.5 * np.ones(problem.dim) / var_k
        # Append theta1_k to the ahead of theta2_k
        theta_k = np.append(theta1_k, theta2_k)
        # theta_k = np.vstack((theta1_k, theta2_k)) ######
        N = int(50 * np.sqrt(dim))
        K = int(np.floor(problem.factors['budget']/(N * self.factors['M'])))
        MaxNumSoln = K + 2
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        
        # Get random solutions from normal distribution (truncated)
        x = np.zeros((N, dim))
        kk = 0
        while kk < N:
            normal_vec = np.array([find_next_soln_rng.normalvariate() for i in range(N * dim)]).reshape((N,dim))
            X_k = np.multiply(normal_vec, np.sqrt(var_k)) + np.ones((N, dim)) * mu_k  ## If bug, double check
            for i in range(N):
                if all(X_k[i, :] >= problem.lower_bounds) and all(X_k[i, :] <= problem.upper_bounds) and kk < N:
                    x[kk, :] = X_k[i, :]
                    kk += 1
        X_k = x

        # new_solution = self.create_new_solution(X_k, problem)
        # recommended_solns.append(new_solution)
        # intermediate_budgets.append(expended_budget)
        # problem.simulate(new_solution, self.factors['M'])
        # expended_budget += self.factors['M']
        # best_solution = new_solution

        Hbar = np.zeros(K)
        xbar = np.zeros((K, dim))
        hvar = np.zeros(K)
        Ntotal = 0
        k = 0

        # Sequentially generate random solutions and simulate them.
        while expended_budget < problem.factors['budget'] and k < K:
            # Iterations
            # while k <= K:
            # X_k = new_solution.x
            alpha_k = self.factors['alpha_0'] / (k + self.factors['alpha_c']) ** self.factors['alpha_p']
            H = np.zeros(N)
            H_var = np.zeros(N)
            for i in range(N):
                new_solution = self.create_new_solution(X_k[i, :], problem)
                X_k[i, :] = new_solution.x
                # intermediate_budgets.append(expended_budget)
                problem.simulate(new_solution, self.factors['M'])
                expended_budget += self.factors['M']
                # best_solution = new_solution
                H[i] = problem.minmax * new_solution.objectives_mean  ####
                # H[i] = np.dot(-1, problem.minmax) * new_solution.objectives_mean
                H_var[i] = np.var(new_solution.objectives)
            # new_solution = np.mean(X_k, axis = 0)
            # recommended_solns.append(new_solution)
            Hbar[k], idx = np.max(H), np.argmax(H)
            # Hbar[k], idx = np.min(H), np.argmin(H)  #######
            hvar[k] = H_var[idx]
            xbar[k, :] = X_k[idx, :]
            new_solution = X_k[idx, :]
            # new_solution = self.create_new_solution(tuple(new_solution), problem) ###1
            new_solution = self.create_new_solution(tuple(xbar[k, :]), problem) ###1
            recommended_solns.append(new_solution) ###1
            intermediate_budgets.append(expended_budget) ###1

            if k >= 1:
                if Hbar[k] < Hbar[k-1] and Hbar[k] != None:  ###<
                    Hbar[k] = Hbar[k-1]
                    xbar[k, :] = xbar[k-1, :]
                    hvar[k] = hvar[k-1]
                    new_solution = xbar[k, :]
                    # best_solution = self.create_new_solution(tuple(xbar[k-1, :]), problem) ###1
                    # recommended_solns[-1] = best_solution # xbar[k-1, :]  ###1
            
            # Shape function
            G_sort = np.sort(H)[::-1]
            gm = G_sort[int(np.ceil(N * self.factors['rho']))]
            S_theta = H > gm

            # Estimate gradient and hessian
            w_k = S_theta/sum(S_theta)
            CX_k = np.vstack((np.sum(X_k, axis = 0), np.sum(X_k * X_k, axis = 0))).T  #element wise product
            # print(np.vstack((mu_k, var_k + mu_k * mu_k)).shape)
            # grad_k = np.matmul(w_k.T, CX_k.T) - np.vstack((mu_k, var_k + mu_k * mu_k)).T # [mu_k, var_k + mu_k * mu_k]
            grad_k = np.vstack((w_k @ X_k, w_k @ (X_k * X_k))).T - np.vstack((mu_k, var_k + mu_k * mu_k)).T
            # Hes_k = -np.cov(CX_k, rowvar=False)
            Hes_k = -np.cov(CX_k)
            Hes_k_inv = np.linalg.inv(Hes_k + 1e-5 * np.eye(dim)) @ np.diag(np.ones(dim))
            # Update the parameter using an SA iteration
            theta_k -= (alpha_k * (Hes_k_inv @ grad_k)).reshape(1, -1)[0] #######
            theta1_k = theta_k[:dim]
            theta2_k = theta_k[dim: 2 * dim]
            var_k = -0.5/theta2_k
            mu_k = theta1_k * var_k

            # Project mu_k and var_k to feasible parameter space
            for i in range(dim):
                if mu_k[i] < problem.lower_bounds[i]:
                    mu_k[i] = problem.lower_bounds[i]
                if mu_k[i] > problem.upper_bounds[i]:
                    mu_k[i] = problem.upper_bounds[i]
            var_k = abs(var_k)

            # Set the stream for generating random candidate solutions

            # Generate candidate solutions from the normal distribution
            x = np.zeros((N, dim))
            kk = 0
            while kk < N:
                normal_vec = np.array([find_next_soln_rng.normalvariate() for i in range(N * dim)]).reshape((N,dim))
                X_k = np.multiply(normal_vec, np.sqrt(var_k)) + np.ones((N, dim)) * mu_k
                for i in range(N):
                    if all(X_k[i, :] >= problem.lower_bounds) and all(X_k[i, :] <= problem.upper_bounds) and kk < N:
                        x[kk, :] = X_k[i, :]
                        kk += 1
            k += 1
            X_k = x

        # Ancalls[1: K + 1] = intermediate_budgets
        # A[1: K + 1, :] = xbar
        # AFnMean[1: K + 1] = problem.minmax * Hbar
        # AFnVar[1: K + 1] = hvar

        # Ancalls[K + 1] = problem.factors['budget']
        # A[K + 1, :] = xbar[K - 1, :]
        # AFnMean[K + 1] = problem.minmax * Hbar[K - 1]
        # AFnVar[K + 1] = hvar[K - 1]

        return recommended_solns, intermediate_budgets