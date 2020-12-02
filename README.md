

# **Bottom-Up Temporal Action Localization with Mutual Regularization (ECCV2020)** [pdf](https://arxiv.org/pdf/2002.07358.pdf)
![avatar](/framework.png)

# Update

2020-12-02 We also provide a pytorch implementation for proposed Mutual Regularization losses in [Mutual_Regularization_Loss.py](https://github.com/PeisenZhao/Bottom-Up-TAL-with-MR/blob/master/Mutual_Regularization_Loss.py).



# Environment Configuration

1. The code is based on tensorflow 1.5 and python3.5
2. Some required python packages:
	tqdm, matplotlib, pickle, json, 

# Data Preparation

We use the features provided by paper **CMCS-Temporal-Action-Localization** [1].
Download and use *merge_feature.py* in *./data* folder to pre-process the features.

[(I3D Features)](https://github.com/Finspire13/CMCS-Temporal-Action-Localization)

[1] Liu D, Jiang T, Wang Y. Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 1298-1307.

We also provide another feature download links：

**THUMOS14**
link: https://jbox.sjtu.edu.cn/l/pn3mvh
pw: ibhg

**ActivityNet1.3**
link: https://jbox.sjtu.edu.cn/l/vuB3WW
pw: yqgt

# Training and Testing

**step 1**: Obtain the proposal results w/o additional proposal scoring.

```
python main.py
```

**step 2**: Obtain the proposal results w/ additional proposal scoring.

```
python main_pem.py
```

**step 3**: Obtain the detection results.

```
python main_detection.py
```


# Citation

Please cite our paper if you use this code in your research:


```
@inproceedings{zhao2020bottom,
  title={Bottom-up temporal action localization with mutual regularization},
  author={Zhao, Peisen and Xie, Lingxi and Ju, Chen and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
  booktitle={European Conference on Computer Vision},
  pages={539--555},
  year={2020},
  organization={Springer}
}
```


