# VD-Matcher: A Very Deep Local Feature Matcher with Weight Recycling and Keypoint Detection
This is the PyTorch implementation of our paper "VD-Matcher: A Very Deep Local Feature Matcher with Weight Recycling and Keypoint Detection".

![overall](https://github.com/mooncake199809/DSAP/blob/main/assets/overall.png)


# Get Started Demo
![demo_img](https://github.com/mooncake199809/DSAP/blob/main/demo/img_res.jpg)

We provide a demo to directly visualize the matching results of DSAP.
You can directly modify your images path to test your own images.
```bash
cd demo
python demo_dsap.py
```

# Installation
Our project is built upon the official code of LoFTR and trained on the ScanNet and MegaDepth dataset.
Please follow [LoFTR](https://github.com/zju3dv/LoFTR) to install the environment and MegaDepth dataset.
Please modify the dataset path in /configs/data when training and testing.

# Training
Please follow the official code of LoFTR to train VD-Matcher.

# Evaluation
The pre-trained models can be downloaded from [DSAP_Mega](https://drive.google.com/drive/folders/1FU8GZ_VdUdbBhPw7m00JNr5Nzd7ZEDg4).
Then, we run the following code to test DSAP on the ScanNet and MegaDepth datasets.
```bash
# Testing VDMatcher-S on the ScanNet dataset
# Results are
# 'auc@10': 0.4847393547946092,   'auc@20': 0.6423085814767878,   'auc@5': 0.2985870969186877
bash scripts/reproduce_test/indoor_small.sh
```
```bash
# Testing VDMatcher-L on the ScanNet dataset
# 'auc@10': 0.4656007069532574,   'auc@20': 0.6284919440707899,   'auc@5': 0.2761418949994144
bash scripts/reproduce_test/indoor_large.sh
```
```bash
# Testing VDMatcher-S on the MegeDepth dataset
bash scripts/reproduce_test/outdoor_small.sh
```
```bash
# Testing VDMatcher-L on the MegeDepth dataset
bash scripts/reproduce_test/outdoor_large.sh
```

# Acknowledgements
We appreciate the previous open-source repository [LoFTR](https://github.com/zju3dv/LoFTR).
We thank for the excellent contribution of LoFTR.

