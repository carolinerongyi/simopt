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
sys.path.insert(0, "C:\\Users\\hagen\\colab_simopt\\simopt")
# sys.path.insert(0, "/Users/CarolineHerr/Documents/GitHub/simopt")

  
from simopt.models.openjackson6 import OpenJackson
# fixed_factors = {"lambda": 3.0, "mu": 8.0}
def calc_lambdas(matrix, arrival, numq):
    routing_matrix = np.asarray(matrix)
    lambdas = np.linalg.inv(np.identity(numq) - routing_matrix.T) @ arrival
    return lambdas
numq = 5
routing_matrix = [[0.2, 0.2, 0.2, 0.2, 0],
                  [0, 0.2, 0.2, 0.2, 0.2],
                  [0.3, 0.1, 0, 0.2, 0.2],
                  [0.1, 0.1, 0.4, 0, 0.2],
                  [0.1, 0.1, 0.1, 0.3, 0.2]]
arrival = (2,2,2,2,2)
lambdas = calc_lambdas(routing_matrix, arrival, numq)
service = 2*np.array(lambdas)
fixed_factors = {'steady_state_initialization': False,
                 'routing_matrix': routing_matrix,
                 'arrival_alphas': arrival, 'warm_up':0, 't_end':2500,
                 'service_mus': service}
mymodel = OpenJackson(fixed_factors)
# -----------------------------------------------

# # The rest of this script requires no changes.

# # Check that all factors describe a simulatable model.
# # Check fixed factors individually.
# for key, value in mymodel.factors.items():
#     print(f"The factor {key} is set as {value}. Is this simulatable? {bool(mymodel.check_simulatable_factor(key))}.")
# # Check all factors collectively.
# print(f"Is the specified model simulatable? {bool(mymodel.check_simulatable_factors())}.")

# # Create a list of RNG objects for the simulation model to use when
# # running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

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

# waiting = responses['waiting_times']
# enter = responses['arrival_record']
# D = responses["D"]
# print(sum([len(waiting[i]) for i in range(len(waiting))]))
# print(sum([len(enter[i]) for i in range(len(enter))]))
# print(sum([len(D[i]) for i in range(len(D))]))

# print(len(D))
# print(len(responses["D"].values()))
# II = np.array([D[i][-1] - D[i-1][-1] for i in range(1, len(D))])
# print(np.all(II>=0))
# print(II)


# # # July 1st version (Biased, but close)
# # def IPA_MM1(W, X, mu):
# #     IPA = [0 for i in range(len(X))]
# #     for i in range(1, len(X)):
# #         # IPA[i] = max(W[i], 0) * (IPA[i-1] - X[i-1]/mu**2)
# #         if W[i] <= 0:
# #             IPA[i] = 0
# #         else:
# #             IPA[i] = IPA[i-1] - X[i-1]/mu
    
# #     return IPA

def I(x, k):
    if x==k:
        return 1
    else:
        return 0
    
    
# # Pre-process
# def get_D(A, St, q):  # A: arrival, St: previous station, q: number of queues
#     D = {}
#     L = [0 for i in range(q)]
#     now = 0
#     tot = sum([len(v) for v in A])
#     while len(list(D.keys())) != tot:
#         v_c = [A[i][L[i]] for i in range(q)]
#         v, idx = np.min(v_c), np.argmin(v_c)  # idx is the index queue
#         pos = L[idx]  # position in the queue
#         st = St[idx][pos]
#         if L[idx] != len(A[idx]) - 1:
#             L[idx] += 1
#         else:
#             A[idx][L[idx]] = 10**8
#         D[now] = [idx, pos, st]
#         now += 1
#     return D

# def get_station(leave):
#     station = [[[-1] for j in range(len(leave[i]))] for i in range(len(leave))]
#     for i in range(len(leave)):
#         for j in range(len(leave[i])):
#             if len(leave[i][j]) == 2:
#                 # print(i, j)
#                 queue_i, posi = leave[i][j][0], leave[i][j][1]
#                 station[queue_i][posi] = [i, j]
    
#     return station

#W: waiting time, A: arrival time, V: service time, Dk: ith customer, Dl: all infor
def get_IPA(Dl, V, W, q, k, mu):  # D is the dictionary, St L[i][1]: ith arrive cust's 
    IA, IW = [[] for i in range(q)], [[-V[i][0]/mu * I(i, k)] for i in range(q)]
    for i in range(len(Dl)):
        queue = int(Dl[i][0])
        idx = Dl[i][1]
        # print(idx, len(V[queue]))
        v = V[queue][idx]
        if idx == 0:
            if Dl[i][2][0] == -1:
                IA[queue].append(0)
            else:
                pre_queue = Dl[i][2][0] 
                pre_idx = Dl[i][2][1]
                # print('i: ', i, ', prequeue: ', pre_queue, ', pre_idx: ', pre_idx)
                if len(IA[pre_queue]) == 0:   # Warm up bug..
                    # print('warmup')
                    a = 0
                else:
                    # print(pre_queue, pre_idx)
                    # print(IW[pre_queue], IA[pre_queue])
                    a = IW[pre_queue][pre_idx] + IA[pre_queue][pre_idx]
                IA[queue].append(a)
        else:
            # Calculate IA
            if Dl[i][2][0] == -1:
                IA[queue].append(0)
            else:
                pre_queue = Dl[i][2][0] 
                pre_idx = Dl[i][2][1]
                # print(pre_queue, pre_idx, IW[pre_queue], IA[pre_queue])
                # if len(IA[pre_queue]) == 0:   # Warm up bug..
                #     # print('warmup')
                #     a = 0
                # else: 
                # print('i: ', i, ', prequeue: ', pre_queue, ', pre_idx: ', pre_idx)
                # print(len(IW[pre_queue]), len(IA[pre_queue]))
                a = IW[pre_queue][pre_idx] + IA[pre_queue][pre_idx]
                IA[queue].append(a)
            if W[queue][idx] <= 0:
                v = -V[queue][idx]/mu * I(queue, k)
                IW[queue].append(v)
            else:
                v = -V[queue][idx]/mu * I(queue, k) + IW[queue][idx-1]
                # print('pre: ', IA[queue][idx-1])
                # print('it: ', IA[queue][idx])
                u = IA[queue][idx-1] - IA[queue][idx]
                IW[queue].append(u + v)
    
    return IW


# IPA = [[] for _ in range(mymodel.factors['number_queues'])]
# mu_IPA = []
# IPA_CI = []
# orig_grad = [[] for _ in range(mymodel.factors['number_queues'])]
# service_mus = (10,10,10,10,10)

# for i in range(1):
#     q = mymodel.factors['number_queues']
#     responses, gradients = mymodel.replicate(rng_list)
#     waiting = responses['waiting_times']
#     service = responses['service_times']
#     enter = responses['arrival_record']  ####
#     stations = responses['transfer_record']
#     complete = responses['complete_times']
#     # stations = get_station(pre_st)
#     # print('lengyth')
#     # print(len(stations[4]))
#     # print(len(enter[4]))
#     # print(len(service[4]))
#     for i in range(q):
#         service_len = len(service[i])
#         enter[i] = enter[i][:service_len]
#         stations[i] = stations[i][:service_len]
#     D = get_D(enter, stations, q)
#     print('yes')
#     # print(D)
#     for j in range(q):
#         IPA[j].append(np.mean(get_IPA(D, service, waiting, q, j, service_mus[j])))
#     print(i)
# lambdas = mymodel.calc_lambdas()
# orig_grad = gradients['total_jobs']['service_mus']
# for j in range(mymodel.factors['number_queues']):
#     mu_IPA.append(lambdas[j]*np.mean(IPA[j]))
#     var_IPA = (lambdas[j]**2) * np.var(IPA[j])
#     IPA_CI.append([mu_IPA[j] - 1.96 * np.sqrt(var_IPA/len(IPA[j])), mu_IPA[j] + 1.96 * np.sqrt(var_IPA/len(IPA[j]))])

# print(IPA_CI)
# print(orig_grad)
    
IPA = [[] for _ in range(mymodel.factors['number_queues'])]
mu_IPA = []
IPA_CI = []
orig_grad = [[] for _ in range(mymodel.factors['number_queues'])]
service_mus = [10, 10, 10, 10, 10]

obj_confs = []
for m in range(10):
    q = mymodel.factors['number_queues']
    responses, gradients = mymodel.replicate(rng_list)
    obj = responses['total_jobs']
    obj_confs.append(obj)   

    waiting = responses['waiting_times']
    service = responses['service_times']
    enter = responses['arrival_record']  ####
    stations = responses['transfer_record']
    # complete = responses['complete_times']
    t = mymodel.factors["t_end"]
    D = responses['IPA_record']
    # print(len(D), sum([len(service[i]) for i in range(len(service))]), sum([len(waiting[i]) for i in range(len(waiting))]))
    # print(D)

    
    # service = service[:len(service)-2]
    # waiting = waiting[:len(service-2)]
    
    # print(len(stations[3]))
    # print(len(enter[3]))
    # print(len(service[3]))
    
    # for i in range(q):
    #     service_len = len(service[i])
    #     enter_len = len(enter[i])
    #     last_wait = waiting[i][-1] + service[i][-1]
    #     for j in range(enter_len - service_len):
    #         service[i].append(0)
    #         waiting[i].append(last_wait)
        
    # stations = get_station(pre_st)
    # print('lengyth')
    # print(len(stations[3]))
    # print(len(enter[3]))
    # print(len(service[3]))
    # # # print(D)
    for j in range(q):
        # print(get_IPA(D, service, waiting, q, j, service_mus[j])[0])
        # print(np.mean(get_IPA(D, service, waiting, q, j, service_mus[j])[0]))
        # for k in range(q):
        ipa = get_IPA(D, service, waiting, q, j, service_mus[j])
        for k in range(q):
            IPA[k].append(np.sum(ipa[k])/t)
            
    # print(m)
obj_mean = np.mean(obj_confs)
obj_var = np.var(obj_confs)
obj_CI = [obj_mean - 1.96 * np.sqrt(obj_var/len(obj_confs)), obj_mean + 1.96 * np.sqrt(obj_var/len(obj_confs))]
lambdas = mymodel.calc_lambdas()
orig_grad = gradients['total_jobs']['service_mus']
for j in range(mymodel.factors['number_queues']):
    mu_IPA.append(lambdas[j]*np.mean(IPA[j]))
    var_IPA = (lambdas[j]**2) * np.var(IPA[j])
    IPA_CI.append([mu_IPA[j] - 1.96 * np.sqrt(var_IPA/len(IPA[j])), mu_IPA[j] + 1.96 * np.sqrt(var_IPA/len(IPA[j]))])

print(IPA_CI)
print(orig_grad)
print(obj_CI)