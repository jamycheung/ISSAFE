# ISSAFE & EDCNet

## Introduction

This is the implementation of our papers below, including the code and dataset (DADA-seg).

Improving Semantic Segmentation in Accidents by Fusing Event-based Data, IROS 2021, [[paper](arxiv.org/pdf/2008.08974.pdf)].

Exploring Event-Driven Dynamic Context for Accident Scene Segmentation, T-ITS 2021, [[paper](arxiv.org/pdf/2112.05006.pdf)].


![issafe_demo](demo/issafe.gif)



## Installation

The requirements are listed in the `requirement.txt` file. To create your own environment, an example is:

```bash
conda create -n issafe python=3.7
conda activate issafe
cd /path/to/ISSAFE
pip install -r requirement.txt
```



## Datasets

For the basic setting of this work, please prepare datasets of [Cityscapes](https://www.cityscapes-dataset.com/), and DADA-seg. 

Our proposed DADA-seg dataset is a subset from [DADA-2000](https://github.com/JWFangit/LOTVS-DADA). Our annotations have the same labeling rule as Cityscapes. 

The DADA-seg dataset is now available in [Google Drive]().

The event generation can be found in [EventGAN](https://github.com/alexzzhu/EventGAN). The anchor and its previous frames are needed for event generation. The generated event volume is saved as `.npy` format for this work. 

A structure of dataset is following:

 ```
dataset
├── Cityscapes
│   ├── event
│   │   ├── train
│   │   │   ├─aachen
│   │   │   │  ├─aachen_000000_000019_gtFine_event.npy	# event volume
│   │   └── val
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   ├─leftImg8bit_prev # for event synthesic
│   │   ├─train
│   │   │  ├─aachen
│   │   │  │  ├─aachen_000000_000019_leftImg8bit_prev.png
│   │   └─val
│   ├── leftImg8bit
│   │   ├── train
│   └── └── val
└── DADA_seg
    ├── dof
    │   └── val
    ├── event
    │   └── val
    ├── gtFine
    │   └── val
    └── leftImg8bit
        ├── train
        └── val

 ```

(option) other sources: [BDD3K](https://bdd-data.berkeley.edu/), [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/), [ApolloScape](http://apolloscape.auto/).

(option) other modalities: dense optical flow.



## Training 

The model of EDCNet can be found at `models/edcnet.py`.

Before run the training script, please modify your own path configurations at `mypath.py`.

The training configurations can be adjusted at `train.py`.

An example of training is `python train.py`



## Evaluation

The evaluation configurations can be adjusted at `eval.py`.

To achieve the evaluation result of EDCNet in D2S mode with 2 event time bins, the weights can be downloaded in [Google Drive](https://drive.google.com/drive/folders/19hUd8Mfj6K76G48AT9txq-PX9bHQN0qs?usp=sharing).

Put the weight at `run/cityscapesevent/test_EDCNet_r18/model_best.pth`.

An example of evaluation of the EDCNet at `B=2` event time bins is `python eval.py`.



## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.



## Citation

If you are interested in this work, please cite the following work:

```
@INPROCEEDINGS{zhang2021issafe,
  author={Zhang, Jiaming and Yang, Kailun and Stiefelhagen, Rainer},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={ISSAFE: Improving Semantic Segmentation in Accidents by Fusing Event-based Data}, 
  year={2021},
  pages={1132-1139},
  doi={10.1109/IROS51168.2021.9636109}}
  
@ARTICLE{zhang2021edcnet,
  author={Zhang, Jiaming and Yang, Kailun and Stiefelhagen, Rainer},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Exploring Event-Driven Dynamic Context for Accident Scene Segmentation}, 
  year={2021},
  pages={1-17},
  doi={10.1109/TITS.2021.3134828}}
```

