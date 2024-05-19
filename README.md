# DeepVenom: training-time DNN backdoors exploiting transient memory faults in model weights

[![GitHub contributors](https://img.shields.io/github/contributors/casrl/deepvenom.svg)](https://GitHub.com/casrl/deepvenom/graphs/contributors/) [![Linux](https://badgen.net/static/os/linux/red)](https://badgen.net/static/os/Linux/red) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://GitHub.com/casrl/deepvenom/graphs/commit-activity) [![Language](https://img.shields.io/badge/Made%20with-C/C++,Python-1f425f.svg)]([https://isocpp.org/std/the-standard](https://img.shields.io/badge/Made%20with-C/C++,Python-1f425f.svg)) [![GitHub license](https://badgen.net/github/license/casrl/deepvenom)](https://github.com/casrl/deepvenom/blob/master/LICENSE)

[![DOI:10.1109/SP54263.2024.00182](https://zenodo.org/badge/DOI/10.1109/SP54263.2024.00182.svg)](https://casrl.ece.ucf.edu/) [![Paper](https://img.shields.io/badge/Paper%20in-IEEE%20S&P%202024-red.svg)](https://casrl.ece.ucf.edu/wp-content/uploads/2024/03/2024-sp-deepvenom.pdf)


DeepVenom demonstrates the first end-to-end fault-based DNN backdoor attack **during model training**. The attack manifests in victim's training runtime that involves fine-tuning a public pre-trained model (PTM), which is a widely popular approach for the deployment of modern ML services. Specifically, DeepVenom induces multi-round weight bit flips during victim's training using memory transient faults, which manages to insert a **targeted backdoor** in victim's fine-tuned model at the end. The bit-level weight perturbations interfere with the weight learning/updating process, transforming the transient faults into persistent backdoor implanted in the victim's model. DeepVenom is a blackbox attack, as the victim model's weights are periodically updated and unknown to the attacker throughout training. We implement an __end-to-end DeepVenom attack__ in real systems with DDR3/DDR4 memories using rowhammer, and evaluate it using state-of-the-art **Convolutional Neural Network and Vision Transformer models**. The results show that DeepVenom can effectively generate backdoors in victim’s fine-tuned models with high attack success rate (ASR) using minimal bit flips (as few as 11 bits). This study highlights the practicality of training-time backdoors through hardware-based weight perturbation, which represents a new dimension in adversarial machine learning.


## Environment
- **Operating System:** Ubuntu 20.04.4 LTS
- **Kernel:** 5.13.0-40-generic
- **GCC:** Ubuntu 9.4.0-1ubuntu1~20.04.1
- **Python:** 3.9.7 (Anaconda distribution)
- **Pytorch:** Source compiled (Tag: v1.7.1-rc3)

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
