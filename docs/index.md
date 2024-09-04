---
hide:
  - toc
---

<p align="center">
  <img src="./assets/config.png" width="400"/>
</p>
<h4 align="center">Towards Conflict-free Training for Everything and Everyone!</h4>

<p align="center">
  [ <a href="https://arxiv.org/abs/2408.11104">📄 Research Paper</a> ]•[ <a href="https://github.com/tum-pbs/ConFIG"><img src="./assets/github.svg" width="16"> GitHub Repository</a> ]
</p>

---

## About

* **What is the ConFIG method?**

​	The conFIG method is a generic method for optimization problems involving **multiple loss terms** (e.g., Multi-task Learning, Continuous Learning, and Physics Informed Neural Networks). It prevents the optimization from getting stuck into a local minimum of a specific loss term due to the conflict between losses. On the contrary, it leads the optimization to the **shared minimum of all losses** by providing a **conflict-free update direction.**

<p align="center">
<img src="./assets/config_illustration.png" style="zoom: 33%;" />
</p>

* **How does the ConFIG work?**

​	The ConFIG method obtains the conflict-free direction by calculating the inverse of the loss-specific gradients matrix:

$$
\mathbf{g}_{ConFIG}=\left(\sum_{i=1}^{m} \mathbf{g}_{i}^\top\mathbf{g}_{u}\right)\mathbf{g}_u,
$$

$$
\mathbf{g}_u = \mathcal{U}\left[
[\mathcal{U}(\mathbf{g}_1),\mathcal{U}(\mathbf{g}_2),\cdots, \mathcal{U}(\mathbf{g}_m)]^{-\top} \mathbf{1}_m\right].
$$

Then the dot product between $\mathbf{g}_{ConFIG}$ and each loss-specific gradient is always positive and equal, i.e., $\mathbf{g}_{i}^{\top}\mathbf{g}_{ConFIG}=\mathbf{g}_{j}^{\top}\mathbf{g}_{ConFIG} > 0 \quad \forall i,j \in [1,m]$​.

* **Is the ConFIG computationally expensive?**

​	Like many other gradient-based methods, ConFIG needs to calculate each loss's gradient in every optimization iteration, which could be computationally expensive when the number of losses increases. However, we also introduce a **momentum-based method** where we can reduce the computational cost **close to or even lower than a standard optimization procedure** with a slight degeneration in accuracy. This momentum-based method is also applicable to other gradient-based methods.

---

## Paper Info

<h4 align="center">ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks</h4>
<h6 align="center"><img src="./assets/TUM.svg" width="16"> <a href="https://qiauil.github.io/">Qiang Liu</a>, <img src="./assets/PKU.svg" width="14"> <a href="https://rachelcmy.github.io/">Mengyu Chu</a>, and <img src="./assets/TUM.svg" width="16"> <a href="https://ge.in.tum.de/about/n-thuerey/">Nils Thuerey</a></h6>

<h6 align="center">
    <img src="./assets/TUM.svg" width="16"> Technical University of Munich
    <img src="./assets/PKU.svg" width="14"> Peking University
</h6>

***Abstract:*** The loss functions of many learning problems contain multiple additive terms that can disagree and yield conflicting update directions. For Physics-Informed Neural Networks (PINNs), loss terms on initial/boundary conditions and physics equations are particularly interesting as they are well-established as highly difficult tasks. To improve learning the challenging multi-objective task posed by PINNs, we propose the ConFIG method, which provides conflict-free updates by ensuring a positive dot product between the final update and each loss-specific gradient. It also maintains consistent optimization rates for all loss terms and dynamically adjusts gradient magnitudes based on conflict levels. We additionally leverage momentum to accelerate optimizations by alternating the back-propagation of different loss terms. The proposed method is evaluated across a range of challenging PINN scenarios, consistently showing superior performance and runtime compared to baseline methods. We also test the proposed method in a classic multi-task benchmark, where the ConFIG method likewise exhibits a highly promising performance. 

***Read from:*** [[Arxiv](https://arxiv.org/abs/2408.11104)]

***Cite as:*** 

```latex
@article{Liu2024ConFIG,
author = {Qiang Liu and Mengyu Chu and Nils Thuerey},
title = {ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks},
year={2024},
url={https://arxiv.org/abs/2408.11104},
}
```

---

## Additional Info
This project is part of the physics-based deep learning topic in [**Physics-based Simulation group**](https://ge.in.tum.de/) at TUM.