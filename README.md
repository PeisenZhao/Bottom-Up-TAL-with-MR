

Code for paper **Bottom-Up Temporal Action Localization with Mutual Regularization (ID:622)**



# Environment Configuration

1. The code is based on tensorflow 1.5
2. Some required python packages:
	tqdm, matplotlib, pickle, json, 

# Data Preparation

We use the features provided by paper **CMCS-Temporal-Action-Localization** [1].
Download and use *merge_feature.py* in *./data* folder to pre-process the features.

[Features](https://github.com/Finspire13/CMCS-Temporal-Action-Localization)


# Training and Testing

**step 1**: Obtain the results w/o additional proposal scoring.

*python main.py 0 0*

**step 2**: Obtain the results w/ additional proposal scoring.

*python main_pem.py 0 0*

**step 3**: Obtain the detection results.

*python main_detection.py 0*




[1] Liu D, Jiang T, Wang Y. Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 1298-1307.