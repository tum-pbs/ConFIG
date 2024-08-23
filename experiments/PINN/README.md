## Install requirements

Before running the experiments, you need to install all the required packages into your environments:

```bash
pip install -r requirements.txt
```

## Training 

The following code shows how to start a training:

```python
from lib_pinns.burgers.trainer import *
from conflictfree.grad_operator import *

SAVE_PATH="./PINN_trained/burgers/"

# Train with usual Adam optimizer:
# Replace "burgers" with "schrodinger", "kovasznay", or "beltrami"
run_burgers( 
    name="baseline",
    trainer=StandardTrainer(),
    save_path=SAVE_PATH
)

# Run gradient-based method:
run_burgers(
    name="ConFIG_2",
    # Replace "Burgers" with "Schrodinger", "Kovasznay", or "Beltrami"; 
    # Replace "ConFIGOperator()" with "PCGradOperator()" or "IMTLGOperator()"
    trainer=get_gradvec_trainer(BurgersTrainerBasis,ConFIGOperator()), 
    save_path=SAVE_PATH,
    n_losses=2, # Number of loss terms, 2 or 3
)

# Run weight-based method
run_burgers(
    name="MinMax_2",
    # Replace "MinMaxWeight" with "ReLoWeight" or "LRAWeight"
    trainer=get_MinMaxWeight_trainer(BurgersTrainerBasis), 
    save_path=SAVE_PATH,
    n_losses=2,
)

# Run momentum-based method
run_burgers(
    name="M_ConFIG_3",
    trainer=get_momentum_trainer(BurgersTrainerBasis,ConFIGOperator()),
    save_path=SAVE_PATH,
    n_losses=3,
)

```

## Test

For each PDE, we offer a function for test:

```python
from lib_pinns.burgers.run_test import run_test # You can check functions for other PDES
from lib_pinns.foxutils.trainerX import TrainedProject
import numpy as np

network=TrainedProject("./PINN_trained/burgers/baseline/2024-05-20-12_43_38/").get_saved_network() #The folder in specificed in training procedure
simulation_data=np.load("./data/burgers/simulation_data.npy")
mse_value,mse,prediction=run_test(network=network,simulation_data=simulation_data,device="cuda:0")
```
To get the best performance during training, you can use the following code:
```python
from lib_pinns.foxutils.trainerX import TrainedProject
import numpy as np
from matplotlib import pyplot as plt

epoch,loss=TrainedProject("./PINN_trained/burgers/baseline/2024-05-20-12_43_38/").get_full_records(key="Loss/validation")
plt.plot(epoch,loss)
plt.yscale("log")
plt.show()
print(np.min(loss))
```

## Additional information

* We use [foxutils](https://github.com/qiauil/Foxutils) to help set up the training procedure. Please check `lib_pinns.foxutils` or the original repository for more information.

* The ground truth for Schrödinger equation is from [RobustPINNs](https://github.com/CVC-Lab/RobustPINNs). Please visit the original repository for more information.