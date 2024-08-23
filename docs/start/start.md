# Quick Start

## Installation

* Install through `pip`: `pip install conflictfree`
* Install from repository online: `pip install git+https://github.com/tum-pbs/ConFIG`
* Install from repository offline: Download the repository and run `pip install .` or `install.sh` in terminal.
* Install from released wheel: Download the wheel and run `pip install conflictfree-x.x.x-py3-none-any.whl` in terminal.

## Use ConFIG method

Suppose you have a muti-loss training mission where each loss can be calculated with a loss function `loss_fn`. All the loss functions are then stored in a `loss_fns` list. Your code would probably looks like this

```python
optimizer=torch.Adam(network.parameters(),lr=1e-3)
for input_i in dataset:
    losses=[]
    optimizer.zero_grad()
    for loss_fn in loss_fns:
        losses.append(loss_fn(network,input_i))
    torch.cat(losses).sum().backward()
    optimizer.step()
```

To use our ConFIG method, you can simply modify the code as

```python
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector
optimizer=torch.Adam(network.parameters(),lr=1e-3)
for input_i in dataset:
    grads=[] # we record gradients rather than losses
    for loss_fn in loss_fns:
    	optimizer.zero_grad()
    	loss_i=loss_fn(input_i)
        loss_i.backward()
        grads.append(get_gradient_vector(network)) #get loss-specfic gradient
    g_config=ConFIG_update(grads) # calculate the conflict-free direction
    apply_gradient_vector(network) # set the condlict-free direction to the network
    optimizer.step()
```

Or, you can use our `ConFIGOperator` class:

```python
from conflictfree.grad_operator import ConFIGOperator
from conflictfree.utils import get_gradient_vector,apply_gradient_vector
optimizer=torch.Adam(network.parameters(),lr=1e-3)
operator=ConFIGOperator() # initialize operator
for input_i in dataset:
    grads=[]
    for loss_fn in loss_fns:
    	optimizer.zero_grad()
    	loss_i=loss_fn(input_i)
        loss_i.backward()
        grads.append(get_gradient_vector(network))
    g_config=operator.calculate_gradient(grads) # calculate the conflict-free direction
    apply_gradient_vector(network) # or simply use `operator.update_gradient(network,grads)` to calculate and set the condlict-free direction to the network
    optimizer.step()
```

The `ConFIGOperator` class and `ConFIG_update` is basically the same, you can choose any one as you like. Besides our ConFIG method, we also provide `PCGradOperator` and `IMTLGOperator` from [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) and [Towards Impartial Multi-task Learning ](https://openreview.net/forum?id=IMPnRXEWpvr), respectively. The usage of these two operators are the same with `ConFIGOperator`.

## Use M-ConFIG method

The basic usage of `M-ConFIG` method in our code is similar to `ConFIGOperator` :

```python
from conflictfree.momentum_operator import PseudoMomentumOperator
from conflictfree.utils import get_gradient_vector,apply_gradient_vector
optimizer=torch.Adam(network.parameters(),lr=1e-3)
operator=PseudoMomentumOperator(num_vector=len(loss_fns)) # initialize operator, the only difference here is we need to specify the number of gradient vectors.
for input_i in dataset:
    grads=[]
    for loss_fn in loss_fns:
    	optimizer.zero_grad()
    	loss_i=loss_fn(input_i)
        loss_i.backward()
        grads.append(get_gradient_vector(network))
    g_config=operator.calculate_gradient(grads) # calculate the conflict-free direction
    apply_gradient_vector(network) # or simply use `operator.update_gradient(network,grads)` to calculate and set the condlict-free direction to the network
    optimizer.step()
```

You can also specify an instance of `PCGradOperator` or `IMTLGOperator` to the `gradient_operator` parameter of `PseudoMomentumOperator` to build momentum-based version of these two methods.