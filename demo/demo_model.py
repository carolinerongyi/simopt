"""
This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np

# Import random number generator.
from mrg32k3a.mrg32k3a import MRG32k3a

# Import model.

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
import sys
  
# # Insert the path of modules folder 
# sys.path.insert(0, "C:\\Users\\hagen\\colab_simopt\\simopt")
sys.path.insert(0, "/Users/CarolineHerr/Documents/GitHub/simopt")

  
from simopt.models.openjackson import OpenJackson
# fixed_factors = {"lambda": 3.0, "mu": 8.0}
fixed_factors = {'steady_state_initialization': False}
mymodel = OpenJackson(fixed_factors)
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


def IPA_MM1(W, X, mu):
    IPA = [0 for i in range(len(X))]
    for i in range(1, len(X)):
        # IPA[i] = max(W[i], 0) * (IPA[i-1] - X[i-1]/mu**2)
        if W[i] <= 0:
            IPA[i] = 0
        else:
            IPA[i] = IPA[i-1] - X[i-1]/mu
    
    return IPA


IPA = [[] for _ in range(mymodel.factors['number_queues'])]
mu_IPA = []
IPA_CI = []
orig_grad = [[] for _ in range(mymodel.factors['number_queues'])]
service_mus = (10,10,10,10,10)

for i in range(1000):
    responses, gradients = mymodel.replicate(rng_list)
    waiting = responses['waiting_times']
    service = responses['service_times']
    for j in range(mymodel.factors['number_queues']):
        IPA[j].append(np.mean(IPA_MM1(waiting[j], service[j], service_mus[j])))
    print(i)
lambdas = mymodel.calc_lambdas()
orig_grad = gradients['total_jobs']['service_mus']
for j in range(mymodel.factors['number_queues']):
    mu_IPA.append(lambdas[j]*np.mean(IPA[j]))
    var_IPA = (lambdas[j]**2) * np.var(IPA[j])
    IPA_CI.append([mu_IPA[j] - 2.576 * np.sqrt(var_IPA/len(IPA[j])), mu_IPA[j] + 2.576 * np.sqrt(var_IPA/len(IPA[j]))])

print(IPA_CI)
print(orig_grad)
    


# Run a single replication of the model.
# responses, gradients = mymodel.replicate(rng_list)
# print("\nFor a single replication:")
# print("\nResponses:")
# for key, value in responses.items():
#     print(f"\t {key} is {value}.")
# print("\n Gradients:")
# for outerkey in gradients:
#     print(f"\tFor the response {outerkey}:")
#     for innerkey, value in gradients[outerkey].items():
#         print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")
