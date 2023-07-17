"""
This script is intended to help with debugging a problem.
It imports a problem, initializes a problem object with given factors,
sets up pseudorandom number generators, and runs multiple replications
at a given solution.
"""

import sys
import os.path as o
sys.path.insert(0, "/Users/CarolineHerr/Documents/GitHub/simopt")
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import random number generator.
# from mrg32k3a.mrg32k3a import MRG32k3a
from mrg32k3a.mrg32k3a import MRG32k3a

# Import the Solution class.
# from simopt.base1 import Solution
from simopt.base import Solution

# Import problem.

# from models.<filename> import <problem_class_name>
# Replace <filename> with name of .py file containing problem class.
# Replace <problem_class_name> with name of problem class.

# Fix factors of problem. Specify a dictionary of factors.

# fixed_factors = {}  # Resort to all default values.
# Look at Problem class definition to get names of factors.

# Initialize an instance of the specified problem class.

# myproblem = <problem_class_name>(fixed_factors=fixed_factors)
# Replace <problem_class_name> with name of problem class.

# Initialize a solution x corresponding to the problem.

# x = (,)
# Look at the Problem class definition to identify the decision variables.
# x will be a tuple consisting of the decision variables.

# The following line does not need to be changed.
# mysolution = Solution(x, myproblem)

# Working example for CntNVMaxProfit problem.
# -----------------------------------------------

# from simopt.models.vac import EOQ_Mincost
# from simopt.models.vac import CovidMinInfectVac
# from simopt.models.vac_alias import CovidMinInfectVac
# from simopt.models.covid import CovidMinInfectVac
# from simopt.models.ccbaby_old import CCBabyShift
# from simopt.models.fickleserver import FickleServerMinServiceRate
from simopt.models.openjackson import OpenJacksonMinQueue
from simopt.models.openjackson import OpenJackson

def rebase(random_rng, n):
    new_rngs = []
    for rng in random_rng:
        stream_index = rng.s_ss_sss_index[0]
        substream_index = rng.s_ss_sss_index[1]
        subsubstream_index = rng.s_ss_sss_index[2]
        new_rngs.append(MRG32k3a(s_ss_sss_index=[stream_index, substream_index, subsubstream_index + n]))
    random_rng = new_rngs
    return random_rng

n_inst = 5

# myproblem = SANLongestPath(random=True)
myproblem = OpenJacksonMinQueue(random=True)

random_rng = [MRG32k3a(s_ss_sss_index=[2, ss + 4, 0]) for ss in range(myproblem.model.n_random, myproblem.model.n_random + myproblem.n_rngs)]
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myproblem.model.n_rngs)]
rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4 + ss, 0]) for ss in range(myproblem.model.n_random)]
rng_list = [* rng_list, * rng_list2]

# Generate 5 random problem instances
for i in range(n_inst):
    # x = (8,) * 13
    x = (8.85,9.45,8.85,11.63,10.8)
    random_rng = rebase(random_rng, i)
    rng_list2 = rebase(rng_list2, i)
    myproblem = OpenJacksonMinQueue(random=True, random_rng=rng_list2)
    myproblem.attach_rngs(random_rng)
    mysolution = Solution(x, myproblem)
    mysolution.attach_rngs(rng_list, copy=False)
    
    # mysolution.attach_rngs(rng_list, copy=False)

    # Simulate a fixed number of replications (n_reps) at the solution x.
    n_reps = 1 #10

    myproblem.simulate(mysolution, m=n_reps)

    # Print results to console.
    print(mysolution.objectives_mean[0])
    print(type(mysolution))

    # print("Objective coefficients: ", myproblem.factors["c"])  # For SAN

    print(f"Ran {n_reps} replications of the {myproblem.name} problem at solution x = {x}.\n")
    # print(f"The mean objective estimate was {round(mysolution.objectives_mean[0], 4)} with standard error {round(mysolution.objectives_stderr[0], 4)}.")

    # print("Objectives coefficients:  ")
    # for i in range(n_reps):
    #     print(myproblem.factors["c"])
        
    print("The individual observations of the objective were:")
    for idx in range(n_reps):
        print(f"\t {round(mysolution.objectives[idx][0], 4)}")
    if myproblem.gradient_available:
        print("\nThe individual observations of the gradients of the objective were:")
        for idx in range(n_reps):
            print(f"\t {[round(g, 4) for g in mysolution.objectives_gradients[idx][0]]}")
    else:
        print("\nThis problem has no known gradients.")
    if myproblem.n_stochastic_constraints > 0:
        print(f"\nThis problem has {myproblem.n_stochastic_constraints} stochastic constraints of the form E[LHS] <= 0.")
        for stc_idx in range(myproblem.n_stochastic_constraints):
            print(f"\tFor stochastic constraint #{stc_idx + 1}, the mean of the LHS was {round(mysolution.stoch_constraints_mean[stc_idx], 4)} with standard error {round(mysolution.stoch_constraints_stderr[stc_idx], 4)}.")
            print("\tThe observations of the LHSs were:")
            for idx in range(n_reps):
                print(f"\t\t {round(mysolution.stoch_constraints[idx][stc_idx], 4)}")
    else:
        print("\nThis problem has no stochastic constraints.")

    
    
    
## For CCBABY:
# myproblem = SANLongestPath(random=True)
# x = (8,)*13

# random_rng = [MRG32k3a(s_ss_sss_index=[2, ss + 4, 0]) for ss in range(myproblem.model.n_random, myproblem.model.n_random + myproblem.n_rngs)]
# myproblem.attach_rngs(random_rng)
# myproblem.rebase(5)
# mysolution = Solution(x, myproblem)
# # -----------------------------------------------

# # Another working example for FacilitySizingTotalCost problem. (Commented out)
# # This example has stochastic constraints.
# # -----------------------------------------------
# # from models.facilitysizing import FacilitySizingTotalCost
# # fixed_factors = {"epsilon": 0.1}
# # myproblem = FacilitySizingTotalCost(fixed_factors=fixed_factors)
# # x = (200, 200, 200)
# # mysolution = Solution(x, myproblem)
# # -----------------------------------------------


# # The rest of this script requires no changes.

# # Create and attach rngs to solution
# rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myproblem.model.n_rngs)]
# rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4 + ss, 0]) for ss in range(myproblem.model.n_random)]
# rng_list = [* rng_list, * rng_list2]
# mysolution.attach_rngs(rng_list, copy=False)

# # Simulate a fixed number of replications (n_reps) at the solution x.
# n_reps = 1 #10

# myproblem.simulate(mysolution, m=n_reps)

# # Print results to console.
# print(mysolution.objectives_mean[0])
# print(type(mysolution))

# print("Objective coefficients: ", myproblem.factors["c"])

# print(f"Ran {n_reps} replications of the {myproblem.name} problem at solution x = {x}.\n")
# # print(f"The mean objective estimate was {round(mysolution.objectives_mean[0], 4)} with standard error {round(mysolution.objectives_stderr[0], 4)}.")

# print("Objectives coefficients:  ")
# for i in range(n_reps):
#     print(myproblem.factors["c"])
    
# print("The individual observations of the objective were:")
# for idx in range(n_reps):
#     print(f"\t {round(mysolution.objectives[idx][0], 4)}")
# if myproblem.gradient_available:
#     print("\nThe individual observations of the gradients of the objective were:")
#     for idx in range(n_reps):
#         print(f"\t {[round(g, 4) for g in mysolution.objectives_gradients[idx][0]]}")
# else:
#     print("\nThis problem has no known gradients.")
# if myproblem.n_stochastic_constraints > 0:
#     print(f"\nThis problem has {myproblem.n_stochastic_constraints} stochastic constraints of the form E[LHS] <= 0.")
#     for stc_idx in range(myproblem.n_stochastic_constraints):
#         print(f"\tFor stochastic constraint #{stc_idx + 1}, the mean of the LHS was {round(mysolution.stoch_constraints_mean[stc_idx], 4)} with standard error {round(mysolution.stoch_constraints_stderr[stc_idx], 4)}.")
#         print("\tThe observations of the LHSs were:")
#         for idx in range(n_reps):
#             print(f"\t\t {round(mysolution.stoch_constraints[idx][stc_idx], 4)}")
# else:
#     print("\nThis problem has no stochastic constraints.")

