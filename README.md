DeGaborNet, a decoupled Gabor convolutional network with bio-inspired design, high efficiency, light weight and robustness advantages, IEEE TCSVT, 2026
==
[Long Yu](https://faculty.scut.edu.cn/zdhkxygc/yl31_en/main.htm), [Haoze Huo](https://orcid.org/0009-0000-8431-4127), [Jia Chen](https://ieeexplore.ieee.org/author/37087092939), [Zhaozhao Zeng](https://xplorestaging.ieee.org/author/37088750424), [Jun Li](https://grzy.cug.edu.cn/lijun1/en/index.htm), [Lin He](https://ieeexplore.ieee.org/author/37588872700), and [Antonio Plaza](https://www2.umbc.edu/rssipl/people/aplaza/).
***

Code for the paper: [DeGaborNet: Decoupled Gabor Network for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/11667704), IEEE Transactions on Circuits and Systems for Video Technology, 2026.

<div align=center><img src="/Figures/DeGaborNet.png" width="90%" height="90%"></div>
<div align=center>(a)</div>
<div align=center><img src="/Figures/De-GCM.png" width="90%" height="90%"></div>
<div align=center>(b)</div>
Fig. 1. Overview of the proposed DeGaborNet. (a) Decoupled Gabor Network. (b) Decoupled Gabor Convolution Module (De-GCM).


### **Abstract**
---
Recent advances in convolutional neural networks (CNNs) have propelled deep learning (DL)-based hyperspectral image (HSI) classification to the forefront of remote sensing research. However, the spatially coupled convolution in CNNs incurs high computational cost and excessive parameters, posing challenges for high-dimensional and large-sized scenarios and severely limiting training/inference efficiency on resource-constrained hardware. To address this issue, we propose a novel decoupled Gabor network (DeGaborNet) with lightweight structure through optimized convolution operations. We first derive a decoupled Gabor (De-Gabor) filter and prove its statistical maximal information consistency with the naive Gabor filter, while spanning a broader function space. Based on this filter, we further develop a decoupled Gabor convolutional module (De-GCM) by decomposing the De-Gabor filter into two learnable 1D Gabor kernels along orthogonal spatial axes, thus performing accelerated directional convolutions with lower inference complexity. Besides, our De-GCM employs the channel sharing and scattering combination strategy for parameters, offering two main advantages: 1) it reduces complexity and redundancy by eliminating most channel-specific parameters, and 2) it enhances parameter diversity and feature robustness through scattering. Thus, the proposed DeGaborNet, constructed with cascaded De-GCMs, are capable of rapidly extracting bio-inspired generalized features at multiple center frequencies and scales. Experimental results on large-sized HSI datasets demonstrate that DeGaborNet significantly reduces model parameters while improving both training and inference efficiency.


<div align=center><img src="/Figures/channel-sharing.png" width="90%" height="90%"></div>
Fig. 2. Channel-shared parameter mechanism of De-GCM.

---

### **The main function of this project**

	Use `Demo_DeGaborNet.py`


Citation
--
The paper is available now at https://ieeexplore.ieee.org/document/11667704

If this work is helpful to you, please cite our paper as follows:

L. Yu et al., "DeGaborNet: Decoupled Gabor Network for Hyperspectral Image Classification," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2026.3727779.

#### BibTeX:
```
@ARTICLE{11667704,
  author={Yu, Long and Huo, Haoze and Chen, Jia and Zeng, Zhaozhao and Li, Jun and He, Lin and Plaza, Antonio},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DeGaborNet: Decoupled Gabor Network for Hyperspectral Image Classification}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Convolutional neural networks;Kernel;Convolution;Gabor filters;Filtering;Educational institutions;Training;Scattering;Image classification;Hyperspectral image classification;convolutional neural networks (CNNs);decoupled Gabor filter;decoupled Gabor network (DeGaborNet)},
  doi={10.1109/TCSVT.2026.3727779}}
```

