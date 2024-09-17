# DeepVenom: Persistent DNN Backdoors Exploiting Transient Weight Perturbations in Memories

[![GitHub contributors](https://img.shields.io/github/contributors/casrl/deepvenom.svg)](https://GitHub.com/casrl/deepvenom/graphs/contributors/) [![Linux](https://badgen.net/static/os/linux/red)](https://badgen.net/static/os/Linux/red) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://GitHub.com/casrl/deepvenom/graphs/commit-activity) [![Language](https://img.shields.io/badge/Made%20with-C/C++,Python-1f425f.svg)]([https://isocpp.org/std/the-standard](https://img.shields.io/badge/Made%20with-C/C++,Python-1f425f.svg)) [![GitHub license](https://badgen.net/github/license/casrl/deepvenom)](https://github.com/casrl/deepvenom/blob/master/LICENSE) 

[![DOI:10.1109/SP54263.2024.00182](https://zenodo.org/badge/DOI/10.1109/SP54263.2024.00182.svg)](https://casrl.ece.ucf.edu/) [![Paper](https://img.shields.io/badge/Paper%20in-IEEE%20S&P%202024-red.svg)](https://casrl.ece.ucf.edu/wp-content/uploads/2024/03/2024-sp-deepvenom.pdf) 


DeepVenom demonstrates the first **end-to-end** hardware-based DNN backdoor attack during **victim model training**. Particularly, DeepVenom can insert a targeted backdoor persistently at the victim model fine-tuning runtime through transient faults in model weight memory (via rowhammer). The attack manifests in two main steps: i) an **offline step** that identifies **weight bit flips transferable to the victim model** using an ensemble-based local model bit search algorithm, and ii) an **online stage** that integrates advanced system-level techniques to efficiently massage weight tensors for **precise rowhammer-based bit flips**. 
DeepVenom further employs a novel iterative backdoor boosting mechanism that performs multiple rounds of weight perturbations to stabilize the backdoor. We implement an __end-to-end DeepVenom attack in real systems with DDR3/DDR4 memories__, and evaluate it using state-of-the-art **Convolutional Neural Network and Vision Transformer models**. The results show that DeepVenom can effectively generate backdoors in victim’s fine-tuned models with upto 99.8% attack success rate (97.8% on average) using as few as 11 total weight bit flips (maximum 49). The evaluation further demonstrates that DeepVenom is successful under varying victim fine-tuning hyperparameter settings, and is highly robust against catastrophic forgetting. Our work highlights the practicality of training-time backdoors through hardware-based weight perturbation, which represents a new dimension in adversarial machine learning.

<!--- This repository contains code for the DeepVenom __algorithm__ and __system-level__ exploits. tools to profile DRAMs to identify memory cells (bits) in DRAM DIMMS that are highly reliable and leakable (using rowhammer as a side channel). The repo also contains a proof-of-concept code example that can be used to leak ML model weight bits (HammerLeak) as proposed in the paper. --->




## Environment
- **Operating System:** Ubuntu 20.04.4 LTS
- **Kernel:** 5.13.0-40-generic
- **GCC:** Ubuntu 9.4.0-1ubuntu1~20.04.1
- **Python:** 3.9.7 (Anaconda distribution)
- **Pytorch:** Source compiled (Tag: v1.7.1-rc3)


## Repository overview  
- **attack_algorithm**
  - deepvenom_kernel.py
    - deepvenom offline search algorithm to identify trigger and bit flips
  - remote_fine_tuning.py
    - remote victim process for regular fine-tuning
  - remote_fault_injection.py
    - online attacker process for applying set of bit flip groups (BFG) via simulation
- **system_exploits**
  - contains a set of tools to carry out the system-level attack (including the Weight Tensor side channel and rowhammer utility functions) 


More information about DeepVenom can be found in our [paper](https://casrl.ece.ucf.edu/wp-content/uploads/2024/03/2024-sp-deepvenom.pdf) in IEEE Symposium on Security and Privacy, 2024. Our work can be cited using the following information.

## Citing our paper  
```bibtex
@inproceedings{deepvenom,
  title={DeepVenom: Persistent DNN Backdoors Exploiting Transient Weight Perturbations in Memories},
  author={Cai, Kunbei and Chowdhuryy, Md Hafizul Islam and Zhang, Zhenkai and Yao, Fan},
  booktitle={IEEE Symposium on Security and Privacy (S&P)},
  year={2024},
  organization={IEEE}
}
```
