# Functions for generating operators and artificial data from models based 
# on such graphs

import numpy as np
import torch
import torch.nn as nn

def cycle_operator(num_vertices, not_movingprobability = 0.3, jump_size = 1):
    #Computes the random walk operator on the vertices 0,..,N-1 of an N-cycle
    # with a given probability of not moving and equidistributing what is left.
    # the jumps go from x to x+jump_size mod N
    N = num_vertices
    operator = torch.zeros(N,N)
    p = not_movingprobability
    assert p<=1.0 and p>=0.0,"ERROR: not_movingprobability MUST be a probability"
    assert jump_size>=1 and jump_size<=N, "ERROR: jump size must be an integer between 1 and N"
    q = (1-p)/2
    for i in range(N):
        for j in range(N):
            if j==i: 
                operator[i,j] = p
            elif (j-i)%N == jump_size or (i-j)%N==jump_size:
                operator[i,j] = q
    return operator

def cycle_operator_tuple(num_vertices, not_moving_probabilities_vector, jump_sizes_vector):
    assert len(not_moving_probabilities_vector) == len(jump_sizes_vector)
    k = len(not_moving_probabilities_vector)
    t0 = cycle_operator(num_vertices, not_movingprobability= not_moving_probabilities_vector[0], jump_size=jump_sizes_vector[0])
    t1 = cycle_operator(num_vertices, not_movingprobability= not_moving_probabilities_vector[1], jump_size=jump_sizes_vector[1])
    return (t0,t1)


def dataLab_cycles(num_vertices, not_moving_probabilities_vector, jump_sizes_vector, noise_stdev, n_samples):
    """WE ADOPT THE CONVENTION THAT IN A DATA TENSOR THE FIRST INDEX IS THE DATAPOINT INDEX"""
    assert len(not_moving_probabilities_vector) == len(jump_sizes_vector)
    k = len(not_moving_probabilities_vector)
    assert k==2, "Current experiment uses only 2-tuples."
    #We will generate a noisy version of the operation op_1@op_2 as traning samples
    ts = cycle_operator_tuple(num_vertices=num_vertices,
                        not_moving_probabilities_vector=not_moving_probabilities_vector,
                        jump_sizes_vector=jump_sizes_vector)

    t0 = ts[0]
    t1 = ts[1]
    comb_operator = 0.76*t1@t0+0.33*t0@t1+0.3*t0@t0@t0
    #pdb.set_trace()
    input_tensor = torch.Tensor(n_samples,num_vertices) #We use the convention that the training data sample_index isthe FIRST index    
    x = input_tensor.reshape(n_samples,1,num_vertices)
    nn.init.uniform_(input_tensor)
    #We compute the output tensor
    res_tensor = torch.tensordot(comb_operator, input_tensor,dims = ([1],[1]))
    res_tensor = torch.transpose(res_tensor,1,0)
    noise_tensor = torch.Tensor(n_samples,num_vertices)
    nn.init.normal_(noise_tensor,std = noise_stdev)
    output_tensor = res_tensor + noise_tensor
    y = output_tensor.reshape(n_samples,1,num_vertices)
    return x,y


if __name__ == "__main__":
    num_vertices = 293
    not_moving_probabilities_vector = [0.05,0.05]
    jump_sizes_vector = [1,30]
    ts = cycle_operator_tuple(num_vertices=num_vertices,
                        not_moving_probabilities_vector=not_moving_probabilities_vector,
                        jump_sizes_vector=jump_sizes_vector)

    operator_tuple = ts
    num_vertices = 5
    jump_sizes_vector = [1,2]

    x, y = dataLab_cycles(num_vertices=num_vertices,
                          not_moving_probabilities_vector=not_moving_probabilities_vector,
                          jump_sizes_vector=jump_sizes_vector,
                          noise_stdev = 0.0,
                          n_samples = 2)

    ts = cycle_operator_tuple(num_vertices=num_vertices,
                        not_moving_probabilities_vector=not_moving_probabilities_vector,
                        jump_sizes_vector=jump_sizes_vector)
    operator_tuple = ts


    import architectures as archit
    M = archit.MonomialWordSupport(num_variables=2, allowed_degree = 3)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    filter_layer = archit.OperatorFilterLayer(num_features_in = 1, num_features_out = 1, monomial_word_support = M)        
    z1 = filter_layer.forward(x)#First index should be the training sample index
    #ztot = filter_layer.forward(x)#ACHTUNG: automatic summations...
   