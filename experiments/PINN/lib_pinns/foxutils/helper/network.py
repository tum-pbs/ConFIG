#usr/bin/python3

#version:0.0.2
#last modified:20231023

import numpy as np

def conv_outputsize(input_size,kernel,stride,pad):
    print(int((input_size+2*pad-kernel)/stride)+1)

def show_paras(model,print_result=True):
    nn_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in nn_parameters])
    # crucial parameter to keep in view: how many parameters do we have?
    if print_result:
        print("model has {} trainable params".format(params))
    return params
