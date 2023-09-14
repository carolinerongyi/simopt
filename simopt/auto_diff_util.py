from __future__ import absolute_import
from autograd.wrap_util import unary_to_nary
from autograd.core import make_vjp as _make_vjp, make_jvp as _make_jvp
from autograd.extend import vspace

import autograd.numpy as np


make_vjp = unary_to_nary(_make_vjp)
make_jvp = unary_to_nary(_make_jvp)

@unary_to_nary
def value_and_jacobian(fun, x):
    vjp, ans = _make_vjp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return ans, np.reshape(np.stack(grads), jacobian_shape)


def bi_dict(names):
    '''
    Takes in a list of strings. Outputs a dict with keys consisting of both elements in the input list and the indices of those elements
    and the values of the dict are the corresponding indices and list elements respectively.
    '''
    my_dict = {}
    for i in range(len(names)):
        name = names[i]
        my_dict[i] = name
        my_dict[name] = i
    return my_dict

def get_response_indx_list(response_names, bi_dict):
    '''
    Takes in list of response names and the associated bi_dict to output a list response indices to be returned
    '''
    response_inds = []
    for name in response_names:
        response_inds.append(bi_dict[name])
    response_inds.sort()
    return response_inds

def get_diff_factor_arr(diff_factor_name_list, factor_dict):
    '''
    Takes in a list of names of the differentiable factors and a dictionary of factor values and returns a dictionary of the differentiable
    factor names and values. This is an extended list if the tuple is of length 3 or less then it appends x,y,z if longer than appends a number
    '''
    # print('diff_factor_name_list', diff_factor_name_list)
    diff_factor_list = []
    extended_diff_factor_name_list = []
    for name in diff_factor_name_list:
        # print("factor dict", factor_dict)
        # print("factor dict each type",type(factor_dict[name]))
        if type(factor_dict[name]) == float:
                diff_factor_list.append(factor_dict[name])
                extended_diff_factor_name_list.append(name)
        ## Hagen changes
        if type(factor_dict[name]) == tuple:
            # print(len(factor_dict[name]))
            tup_length = len(factor_dict[name])
            if tup_length <= 3:
                for i in range(tup_length):
                    extended_diff_factor_name_list.append(name + "_" + str(chr(120+i)))
                    diff_factor_list.append((factor_dict[name][i]))
            else:
                 for i in range(tup_length):
                    extended_diff_factor_name_list.append(name + "_" + str(i))
                    diff_factor_list.append((factor_dict[name][i]))
        # print("name list",extended_diff_factor_name_list)
        # print("factor list",diff_factor_list)
    try:
        # print("diff factor list values",np.array(diff_factor_list)._value)
        return np.array(diff_factor_list)._value, np.array(extended_diff_factor_name_list)._value
    except:
        # print("woah boy", np.array(diff_factor_list))
        return np.array(diff_factor_list), np.array(extended_diff_factor_name_list)

def response_arr_to_dict(responses_arr, response_names):
    '''
    Takes in an array of responses and a list of response names and converts the array to a dictionary of response names and values
    '''
    responses_dict = {}
    for i in range(len(response_names)):
        thing = responses_arr[i]
        response = response_names[i]
        # sometimes thing will be a float, other times it will be an Autograd array, thus we need to following code
        try:
            responses_dict[response] = thing._value
        except:
            responses_dict[response] = thing
    return responses_dict

def gradient_arr_to_dict(gradients_arr, response_names, diff_factors_names):
    '''
    Takes in an array of gradients, a list of response names, and a list of differentiable factor names and converts 
    the array of gradients to a dictionary with format consistent to what we use elsewhere
    '''
    gradients_dict = {}
    for i in range(len(response_names)):
        response = response_names[i]
        gradients_dict[response] = {}
        # print('diff_factor_names =',diff_factors_names)
        for j in range(len(diff_factors_names)):
            # print('gradients_arr', gradients_arr)
            # print('i=',i)
            # print('j=',j)
            thing = gradients_arr[i,j]
            # sometimes thing will be a float, other times it will be an Autograd array, thus we need to following code
            try:
                gradients_dict[response][diff_factors_names[j]] = thing._value
            except:
                gradients_dict[response][diff_factors_names[j]] = thing
    return gradients_dict
    
def replicate_wrapper(model, rng_list, **kwargs):
    '''
    wrapper to put around model.inner_replicate() to allow Autograd to get gradients if gradient_needed=True
    '''
    
    
    #if response_names not specified, return all responses
    if 'response_names' in kwargs:
        response_names = kwargs['response_names']
    else:
        response_names = model.response_names
        
    if 'gradient_needed' in kwargs:
        gradient_needed = kwargs['gradient_needed']
    else:
        gradient_needed = True
        
    response_inds = get_response_indx_list(response_names, model.bi_dict)
    diff_factor_vals, extended_diff_factor_names = get_diff_factor_arr(model.differentiable_factor_names, model.factors)
    
    # calculate gradients only if gradient needed, otherwise return only np.nan's as gradient values to save computation
    if gradient_needed:
        # hagen changed so it is the extended list
        # responses_arr, gradients_arr = value_and_jacobian(model.inner_replicate)(diff_factor_vals, rng_list,response_names)
        # return response_arr_to_dict(responses_arr, response_names), gradient_arr_to_dict(gradients_arr, response_names,
        #                                                                                  model.differentiable_factor_names)
        # print("DIFF FACTOR VLS",diff_factor_vals)
        responses_arr, gradients_arr = value_and_jacobian(model.inner_replicate)(diff_factor_vals, rng_list,response_names)
        return response_arr_to_dict(responses_arr, response_names), gradient_arr_to_dict(gradients_arr, response_names,
                                                                                         extended_diff_factor_names)
    else:
        responses_arr = model.inner_replicate(diff_factor_vals, rng_list, response_names)
        # ##### Hagen Change
        # gradients = {response_key: {factor_key: np.nan for factor_key in model.differentiable_factor_names} for response_key in 
        #              response_names}
        gradients = {response_key: {factor_key: np.nan for factor_key in extended_diff_factor_names} for response_key in 
                     response_names}
        return  response_arr_to_dict(responses_arr, response_names), gradients
    
def factor_dict(model, diff_factors):
    '''
    creates a expanded list of factors that expands tuples so they can be differentiated
    '''
    factor_dict = model.factors
    counter = 0 # track the number of added indices
    for i in range(len(model.differentiable_factor_names)):
        name = model.differentiable_factor_names[i]
        # print(type(factor_dict[name]))
        if type(factor_dict[name]) == float:
            factor_dict[name] = diff_factors[i]
        if type(factor_dict[name]) == tuple: ### hagen changed so that factor dict returns an altered list of factors
            # factor_dict[name] = np.array(diff_factors[i])
            tup_length = len(factor_dict[name])
            if tup_length <= 3:
                for j in range(tup_length):
                    factor_dict[name + "_" + str(chr(120+j))] = (diff_factors[i+counter+j])
            else:
                 for j in range(tup_length):
                    factor_dict[name + "_" + str(j)]= (diff_factors[name][i+counter+j])
            counter += tup_length-1
    return factor_dict
        
def resp_dict_to_array(model, resp_dict, wanted_names):
    resp_list = []
    for name in wanted_names:
        # print("resp_dict", resp_dict[name])
        resp_list.append(resp_dict[name])
    return np.array(resp_list)
    
    

