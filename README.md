# TOPIQ Standalone

TOPIQ (TOP-down approach for Image Quality assessment) is a state-of-the-art image quality assessment (IQA) model that uses a cross-scale feature attention network to analyze image quality.

This standalone implementation allows you to use TOPIQ without installing the full IQA-PyTorch package.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/TOPIQ-Standalone.git
   cd TOPIQ-Standalone
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Available Models

TOPIQ has several variants for different purposes:

- **topiq_nr**: No-reference IQA for general purpose
- **topiq_nr-face**: No-reference IQA specialized for face images
- **topiq_nr-flive**: No-reference IQA tuned on the FLIVE dataset
- **topiq_nr-spaq**: No-reference IQA tuned on the SPAQ dataset
- **topiq_fr**: Full-reference IQA comparing distorted to reference image
- **topiq_iaa**: Image aesthetic assessment (returns scores from 1-10)

## Usage

### Basic Usage

```python
import torch
from topiq_model import create_topiq, load_image

# Create TOPIQ model (No-reference by default)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_topiq(model_name="topiq_nr", device=device)

# Load and process an image
image_path = "path/to/your/image.jpg"
image = load_image(image_path).to(device)

# Get quality score
with torch.no_grad():
    score = model(image)
    
print(f"Image quality score: {score.item():.4f}")
```

### Full-Reference Usage

```python
# Create full-reference TOPIQ model
model = create_topiq(model_name="topiq_fr", device=device)

# Load distorted and reference images
dist_img = load_image("distorted.jpg").to(device)
ref_img = load_image("reference.jpg").to(device)

# Get quality score
with torch.no_grad():
    score = model(dist_img, ref_img)
    
print(f"Image quality score: {score.item():.4f}")
```

### Command-line Example

Use the provided example script to quickly assess image quality:

```bash
# No-reference quality assessment
python example.py --image path/to/image.jpg --model topiq_nr

# Full-reference quality assessment
python example.py --image path/to/distorted.jpg --reference path/to/reference.jpg --model topiq_fr

# Face image quality assessment
python example.py --image path/to/face.jpg --model topiq_nr-face

# Aesthetic quality assessment
python example.py --image path/to/image.jpg --model topiq_iaa
```

## Pre-trained Weights

The pre-trained weights will be automatically downloaded when you first run the model. They will be stored in `~/.cache/topiq/` directory.

## Citation

If you use TOPIQ in your research, please cite the original paper:

```bibtex
@article{chen2023topiq,
  title={TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment},
  author={Chen, Chaofeng and Mo, Jiadi and Hou, Jingwen and Wu, Haoning and Liao, Liang and Sun, Wenxiu and Yan, Qiong and Lin, Weisi},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```
