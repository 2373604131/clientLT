# CAPT: Class-Aware Prompt Tuning for Federated Long-Tailed Learning with Vision-Language Model





## :ballot_box_with_check: Supported Methods

| Method              | Paper                                                        |
| ------------------- | ------------------------------------------------------------ |
| CAPT                | Our method                                                   |
| PromptFL (Baseline) | [TMC 2023](https://ieeexplore.ieee.org/document/10210127)    |
| CLIP-LoRA           | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024W/PV/html/Zanella_Low-Rank_Few-Shot_Adaptation_of_Vision-Language_Models_CVPRW_2024_paper.html) |
| FedTPG              | [ICLR 2024](https://openreview.net/forum?id=NW31gAylIm)      |
| FedOTP              | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Global_and_Local_Prompts_Cooperation_via_Optimal_Transport_for_Federated_CVPR_2024_paper.html) |
| FedPGP              | [ICML 2024](https://dl.acm.org/doi/10.5555/3692070.3692451)  |
| FedCLIP             | [IEEE Data Engineering Bulletin 2023](https://arxiv.org/pdf/2302.13485v2) |
| MaPLe               | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.html) |
| KgCoOp              | [CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yao_Visual-Language_Prompt_Tuning_With_Knowledge-Guided_Context_Optimization_CVPR_2023_paper.html) |
| Co-CoOp             | [CVPR 2022](http://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.html) |
| CoOp                | [IJCV 2022](https://link.springer.com/article/10.1007/s11263-022-01653-1) |

<hr />

## Requirements 

- Python 3.8+
- Pytorch 1.10.0+

For installation and other package requirements:

```
pip install -r requirements.txt
```

## Data Preparation
For CIFAR10, CIFAR100, Fashion-MNIST datasets, please download and unzip data under `DATA/` file catalog. Or simply run experiments with CIFAR10/CIFAR100/Fashion-MNIST dataset, the program will download data automatically.

For ImageNet-1k datasets：

- Create a folder named `imagenet/` under `DATA/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/`. The directory structure should look like

```
imagenet/
|–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|–– val/
```

- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the training and validation sets to `$DATA/imagenet/images`.


## ⚡ Quick Start 
CAPT and other methods can be easily trained using simple scripts. 

For example, to train CAPT on ImageNet-LT, run the following command:

```
sh scripts/capt.sh
```

<hr />

## References

Our code is based on [PEILab-Federated-Learning/PromptFL](https://github.com/PEILab-Federated-Learning/PromptFL)

<hr />
