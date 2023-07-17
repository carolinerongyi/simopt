"""
This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

"""
Instead of modifying the problem and model class, we modify the demo_model and demo_problems.
"""

import sys
import os.path as o
# sys.path.insert(0, "/Users/CarolineHerr/Documents/GitHub/simopt")
sys.path.insert(0, "C:\\Users\\hagen\\colab_simopt\\simopt")
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

import numpy as np
# Import random number generator.
# from mrg32k3a.mrg32k3a import MRG32k3a
from mrg32k3a.mrg32k3a import MRG32k3a

# Import model.    
# from simopt.models.vac import COVID_vac      #vaccination (simulate individuals)
# from simopt.models.EOQ import EOQ
# from simopt.models.covid import COVID_vac    #combined case

# from simopt.models.ccbaby1 import BabyCC
from simopt.models.openjackson import OpenJacksonMinQueue
from simopt.models.openjackson import OpenJackson
# from simopt.models.chessmm import ChessMatchmaking
# from simopt.models.chessmm import ChessAvgDifference

x = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 100)

# fixed_factors = {"num_x": x, "mu_h": 10/60}

# fixed_factors = {"num_arcs": 13}
fixed_factors = {}

# mymodel = SAN(fixed_factors = fixed_factors, random=True)
mymodel = ChessMatchmaking(fixed_factors = fixed_factors, random=True)
# from models.<filename> import <model_class_name>
# Replace <filename> with name of .py file containing model class.
# Replace <model_class_name> with name of model class.

# Fix factors of model. Specify a dictionary of factors.

# fixed_factors = {}  # Resort to all default values.
# Look at Model class definition to get names of factors.

# Initialize an instance of the specified model class.

# mymodel = <model_class_name>(fixed_factors)
# Replace <model_class_name> with name of model class.

# Working example for MM1 model.
# -----------------------------------------------
# from simopt.models.mm1queue import MM1Queue
# fixed_factors = {"lambda": 3.0, "mu": 8.0}
# mymodel = MM1Queue(fixed_factors)
# -----------------------------------------------

# The rest of this script requires no changes.

# Check that all factors describe a simulatable model.
# Check fixed factors individually.

for key, value in mymodel.factors.items():
    print(f"The factor {key} is set as {value}. Is this simulatable? {bool(mymodel.check_simulatable_factor(key))}.")
# Check all factors collectively.
print(f"Is the specified model simulatable? {bool(mymodel.check_simulatable_factors())}.")

# Create a list of RNG objects for the simulation model to use when
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4 + ss, 0]) for ss in range(mymodel.n_random)]
# rng_list = [* rng_list, * rng_list2]

# Run a single replication of the model.
# before change:

# rng_random = [MRG32k3a(s_ss_sss_index=[2, ss, 0]) for ss in range(mymodel.n_rngs+4)]
for i in range(5):
    # print('\nRandom graph: ')
    # print()
    mymodel.attach_rng(rng_list2)
    responses, gradients = mymodel.replicate(rng_list)
    print("\nFor a single replication:")
    print("\nResponses:")
    for key, value in responses.items():
        print(f"\t {key} is {value}.")

# print("\n Gradients:")
# for outerkey in gradients:
#     print(f"\tFor the response {outerkey}:")
#     for innerkey, value in gradients[outerkey].items():
#         print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")


# # Run multiple replication of model For CCBABy
# L_perf= []
# L_cost = []
# L_lost = []
# # L_hl_perf = []  #!!for combined case only


# for i in range(10):
#     responses, gradients = mymodel.replicate(rng_list)
#     ##if want to print out the lines in original code, comment these prints.
#     print("\nFor a single replication:")
#     print("\nResponses:")
#     L_perf.append(responses.get('Mean_performance'))  #change to only get the last day's suspectible
#     # L_hl_perf.append(responses.get('Half_length_performance'))  
#     L_cost.append(responses.get('Mean_cost')) #!!for combined case only
#     L_lost.append(responses.get('Lost_call'))
        
#     for key, value in responses.items():
#         print(f"\t {key} is {value}.")
   
# L_cost = np.array(L_cost)  
# L_perf = np.array(L_perf)
# L_lost = np.array(L_lost)
# # L_hl_perf = np.array(L_hl_perf)  #!!for combined case only
# meanCost = np.mean(L_cost) 
# meanPerf = np.mean(L_perf, axis=0)
# meanLost = np.mean(L_lost)
# varPerf = np.var(L_perf, axis=0)
# halfPerf = 1.96*np.sqrt(varPerf/len(L_perf)) 
# halfLost = 1.96*np.sqrt(np.var(meanLost)/len(L_lost))
# # meanHalflen = np.mean(L_hl_perf)   #!!for combined case only
# print("num_x: ", x)
# print("Averaged costs of 50 replication: ", meanCost)   
# print("Averaged performance of 20 replication: ", meanPerf) 
# print("variance: ", varPerf)
# print("Half length of performance CI: ", halfPerf)
# print("Averaged lost of 20 replication: ", meanLost)
# print("Half length of lost calls CI: ", halfLost)

# # print("Averaged half length of CI for performance of 20 replication: ", meanHalflen)   

