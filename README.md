# Defocus-Deepfake-Detection

This repository provides the official implementation of **Defocus-based Deepfake Detection**.  
We leverage physically-grounded **defocus blur cues** for discriminating between real and synthetic images.  

## ⚙️ Requirements
We recommend using **Python 3.8+** and creating a virtual environment.

```bash
# Create and activate environment
conda create -n defocus python=3.8 -y
conda activate defocus

# Install dependencies
pip install torch torchvision torchaudio
pip install timm
pip install numpy matplotlib scikit-learn
pip install jupyter seaborn
```
## Training with Defocus Map Estimation Module

We provide a training script with the defocusb module integrated.
Run the following command to train with defocus-based features:

```bash
python train_backbone_gt.py --backbone="legacy xception" --fake_type="Deepfakes"

```

## Pretrained Weights

Pretrained weights are provided for reproduction and evaluation.
To use them, run the same training script but set epochs=0:

```bash
python train_backbone_gt.py --backbone="legacy xception" --fake_type="Deepfakes" --epochs=0

```

## Feature-level Analysis 
We analyze defocus maps at the feature level, including:

1. Binary Mask Generation

2. Local Variance Heatmap

3. t-SNE Visualization

```bash
/visualize
   ├── Binary_Mask/
   │   └── Binary Mask.ipynb
   │
   ├── Local Variance/
   │   └── Local Variance.ipynb
   │
   ├── t-SNE/
   │   └── t-SNE.ipynb
   │
   ├── Shapley Save/
   │   └── Shapley Save.ipynb
   │
   └── Shapley Alignment Score
       └── Shap AlignmentScore.ipynb

```

## Defocus-based Analysis Workflow
To perform feature/model-level analysis, Shapley values and defocus maps must first be saved.

1. Run Shapley Save.ipynb: Save defocus maps and Shapley values as .npz

2. After saving, you can compute either Feature-level Analysis or Model-level Analysis.
  
3. Load the .npz files for Feature-level Analysis (Binary mask, Local Variance, or t-SNE analysis). 

4. Load the .npz files for Model-level Analysis (Shapley Value Visualization, Shapley alignment scores).

## Citation
@inproceedings{10.1145/3746252.3761260,
author = {Jeon, Minsun and Woo, Simon S.},
title = {Seeing Through the Blur: Unlocking Defocus Maps for Deepfake Detection},
year = {2025},
isbn = {9798400720406},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746252.3761260},
doi = {10.1145/3746252.3761260},
abstract = {The rapid advancement of generative AI has enabled the mass production of photorealistic synthetic images, blurring the boundary between authentic and fabricated visual content. This challenge is particularly evident in deepfake scenarios involving facial manipulation, but also extends to broader AI-generated content (AIGC) cases involving fully synthesized scenes. As such content becomes increasingly difficult to distinguish from reality, the integrity of visual media is under threat. To address this issue, we propose a physically interpretable deepfake detection framework and demonstrate that defocus blur can serve as an effective forensic signal. Defocus blur is a depth-dependent optical phenomenon that naturally occurs in camera-captured images due to lens focus and scene geometry. In contrast, synthetic images often lack realistic depth-of-field (DoF) characteristics. To capture these discrepancies, we construct a defocus blur map and use it as a discriminative feature for detecting manipulated content. Unlike RGB textures or frequency-domain signals, defocus blur arises universally from optical imaging principles and encodes physical scene structure. This makes it a robust and generalizable forensic cue. Our approach is supported by three in-depth feature analyses, and experimental results confirm that defocus blur provides a reliable and interpretable cue for identifying synthetic images. We aim for our defocus-based detection pipeline and interpretability tools to contribute meaningfully to ongoing research in media forensics. The implementation is publicly available at: https://github.com/irissun9602/Defocus-Deepfake-Detection},
booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
pages = {1091–1102},
numpages = {12},
keywords = {deepfake detection, defocus blur, image forensics},
location = {Seoul, Republic of Korea},
series = {CIKM '25}
}

