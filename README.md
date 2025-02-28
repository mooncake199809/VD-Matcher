# VD-Matcher: A Very Deep Local Feature Matcher with Weight Recycling and Keypoint Detection
This is the PyTorch implementation of our paper "VD-Matcher: A Very Deep Local Feature Matcher with Weight Recycling and Keypoint Detection".

![overall](https://github.com/mooncake199809/VD-Matcher/blob/main/imgs/Overall.png)


# Get Started Demo
![demo_img](https://github.com/mooncake199809/VD-Matcher/blob/main/imgs/Matches.png)

We provide a demo to directly visualize the matching results of DSAP.
You can directly modify your images path to test your own images. The coarse matching threshold is set to 0.7 in this demo. You can change it in src/loftr/utils/cvpr_ds_config.py.
```bash
cd demo
python demo.py
```

# Installation
Our project is built upon the official code of LoFTR and trained on the ScanNet and MegaDepth dataset.
Please follow [LoFTR](https://github.com/zju3dv/LoFTR) to install the environment and MegaDepth dataset.
Please modify the dataset path in configs/data when training and testing.

# Training
Please follow the official code of [LoFTR](https://github.com/zju3dv/LoFTR) to train VD-Matcher.

# Evaluation
The pre-trained models can be downloaded from [VD-Matcher_Weights](https://drive.google.com/drive/folders/1FU8GZ_VdUdbBhPw7m00JNr5Nzd7ZEDg4).

Then, we can simply run the following code provided in scripts/reproduce_test to test VD-Matcher on the ScanNet and MegaDepth datasets.
Please change the "ckpt_path" in scripts/reproduce_test.
Taking an example, we can run the following code to test VD-Matcher-S on the ScanNet and MegeDepth dataset.
```bash
# Testing VD-Matcher-S on the ScanNet dataset
# Results are
# 'auc@10': 0.4656007069532574,   'auc@20': 0.6284919440707899,   'auc@5': 0.2761418949994144
bash scripts/reproduce_test/indoor_small.sh
```
```bash
# Testing VD-Matcher-S on the MegeDepth dataset
# Results are
# 'auc@10': 0.7334482280642276,   'auc@20': 0.8454705717469903,   'auc@5': 0.5713914477259984
bash scripts/reproduce_test/outdoor_small.sh
```

# Acknowledgements
We appreciate the previous open-source repository [LoFTR](https://github.com/zju3dv/LoFTR).
We thank for the excellent contribution of LoFTR.

