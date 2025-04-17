# SemFuse: Semantic-Aware Infrared and Visible Image Fusion with Transformer Guidance

This repository contains the implementation of "SemFuse: Semantic-Aware Infrared and Visible Image Fusion with Transformer Guidance," which extends transformer-based image fusion with semantic guidance from the Segment Anything Model (SAM).

## Abstract

Infrared and visible image fusion aims to integrate complementary information from different types of images into one image. While existing methods have made significant progress, they often lack semantic awareness to preserve important regions in the final fused image. We propose SemFuse, a novel approach that incorporates high-level semantic guidance from the Segment Anything Model (SAM) to enhance the fusion process. Our method leverages both the global context capabilities of transformers and the semantic understanding of pre-trained segmentation models to generate fused images that better preserve semantically meaningful regions while maintaining optimal detail transfer from both modalities.

## Architecture

SemFuse has five main components:
1. **Dual CNN Feature Extractor**: Extracts low-level features from infrared and visible images
2. **SAM Semantic Guidance Module**: Provides semantic segmentation maps to guide the fusion process
3. **Transformer Module**: Captures global relationships in features using channel and spatial transformer blocks
4. **Semantic Feature Enhancement**: Uses semantic maps to enhance features in important regions
5. **Fusion and Reconstruction Module**: Combines and reconstructs the final fused image

## Requirements

```
Python 3.8+
PyTorch 1.8+
segment-anything
numpy
scipy
torchvision
tqdm
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/username/SemFuse.git
cd SemFuse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SAM model checkpoint:
```bash
# Download the SAM checkpoint (ViT-H version)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Dataset Preparation

Prepare your infrared and visible image pairs in separate directories. The dataset directory structure should be:
```
dataset/
├── rgb/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── ir/
│   ├── img1.png
│   ├── img2.png
│   └── ...
```

## Training

To train SemFuse on your dataset:

```bash
python train.py \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --semantic_loss_weight 0.3 \
  --epochs 30 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --dataset_path /path/to/dataset/rgb \
  --save_model_path /path/to/save/models \
  --train_num 7000 \
  --image_height 256 \
  --image_width 256
```

## Testing

To test the trained model:

```bash
python test.py \
  --model_path /path/to/saved/model.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --vi_path /path/to/visible/image.png \
  --ir_path /path/to/infrared/image.png \
  --output_path /path/to/output/directory \
  --num_tests 1
```

## Model Performance

SemFuse achieves state-of-the-art performance by:
- Preserving semantically important regions in the fused image
- Maintaining complementary information from both modalities
- Leveraging the global context understanding of transformers
- Adding semantic guidance from pre-trained segmentation models

## Citation

If you find our work useful in your research, please consider citing:

```
@article{semfuse2023,
  title={SemFuse: Semantic-Aware Infrared and Visible Image Fusion with Transformer Guidance},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## Acknowledgements

This work builds upon the original transformer-based fusion network. We thank the authors of Segment Anything Model (SAM) for making their code publicly available.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
