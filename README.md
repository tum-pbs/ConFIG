<center>
<img src="./pics/config.png"/>
</center>

## About

Official Implementation of **Con**flict-**F**ree **I**nverse **G**radients (ConFIG) method.

ConFIG method is a generic method for gradient-based multi-object optimizations. It calculates an optimal update direction for all optimization objects, leading them to converge simultaneously. Combined with the unique momentum component, it can achieve efficient training for Multi-Task Learning (MTL), Continuous Learning (CL), Physics-Informed Neural networks (PINNs), and other potential multi-object deep learning missions.

For more details for the ConFIG method, please refers to our research paper:

```bibtex
@inproceedings{liu2024config,
      title={ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks}, 
      author={Qiang Liu, Mengyu Chu, Nils Thuerey},
      year={2024},
      booktitle={arXiv XXXX},
      url={https://github.com/tum-pbs/ConFIG}
}
```

## Installation

Please direct run `install.sh` or the following code to install ConFIG package:

```bash
python3 setup.py sdist bdist_wheel
cd dist
pip install ConFIG-*.whl
```

Note that the current package is PyTorch-based, implementation for other framework is under active consideration.

## Usage

We provide a detailed [tutorial notebook](https://github.com/tum-pbs/ConFIG/blob/main/quick-how-to.ipynb) to illustrate how to use the ConFIG method. 

We also offer the source code of the PINNs and MTL experiments discussed in our research paper in the `experiments` folder. Each experiment contains separate introductions for the configuration detail.

## Additional Information

This project is part of the physics-based deep learning topic in **Physics-based Simulation group** at TUM. Please visit our [homepage](https://ge.in.tum.de/) to see more related publications.