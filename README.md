

**Bottom-Up Temporal Action Localization with Mutual Regularization (ECCV2020)**



# Environment Configuration

1. The code is based on tensorflow 1.5 and python3.5
2. Some required python packages:
	tqdm, matplotlib, pickle, json, 

# Data Preparation

We use the features provided by paper **CMCS-Temporal-Action-Localization** [1].
Download and use *merge_feature.py* in *./data* folder to pre-process the features.

[(I3D Features)](https://github.com/Finspire13/CMCS-Temporal-Action-Localization)

[1] Liu D, Jiang T, Wang Y. Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 1298-1307.

# Training and Testing

**step 1**: Obtain the results w/o additional proposal scoring.

```
python main.py
```

**step 2**: Obtain the results w/ additional proposal scoring.

```
python main_pem.py
```

**step 3**: Obtain the detection results.

```
python main_detection.py
```


# Citation

Please cite our paper if you use this code in your research:





