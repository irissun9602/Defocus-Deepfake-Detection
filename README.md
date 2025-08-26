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

## Shapley-based Analysis Workflow
To perform feature-level analysis, Shapley values and defocus maps must first be saved.

1. Run Shapley Save
  Save defocus maps and Shapley values as .npz

2. Shapley Alignment or Feature-level Analysis
  After saving, you can either:

3. Compute Shapley alignment scores

4. Or load the .npz files for binary mask, variance, or t-SNE analysis.

## Citation
@inproceedings{sun2025defocus,
  author    = {Jeon, Minsun and Woo, Simon S.},
  title     = {Seeing Through the Blur: Unlocking Defocus Maps for Deepfake Detection},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year      = {2025},
  ...
}

