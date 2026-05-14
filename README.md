# DETR Variants: Scalable Detection Transformers

This repository contains implementations of three scalable variants of DETR (Detection Transformer) for object detection with dynamic depth and width configurations.

## Introduction

Object detection is applied across a wide range of domains, and each application introduces specific computational constraints and requirements. To accommo-
date these constraints, the most common practice is to train multiple models from scratch, resulting in high computational costs and excessive memory usage. To reduce computational and memory demands, we propose scalable Detection Transformers (DETR) that inherently support subnets such that increasing the complexity smoothly improves the model performance. As a result, a single model can be used in different environments without defining the constraints in advance. We introduce
two variants of DETR, namely Depth-Scalable DETR (DS-DETR) and Width-Scalable DETR (WS-DETR). Our experiments demonstrate that both DS-DETR and WS-DETR provide multiple subnets whose performance improves smoothly with increasing model complexity. Furthermore, DS-DETR achieves a maximum performance of AP = 0.374 with a supported range of 74.4GMAC to 77.4GMAC with six subnets under budget-constrained training. For intermediate configurations, DS-DETR even reaches near baseline-level performance when trained with an extended number of epochs. Additionally, the results show that DS-DETR consistently achieves higher precision than WS-DETR across the model complexity range with a maximum difference of ΔAP = 0.113, while providing more subnets.

## Models Overview

### DS-DETR (Depth Scalable DETR)
**Depth Scalable DETR for dynamic depth** provides a flexible transformer-based detector that can adjust the depth (number of transformer encoder and decoder layers) dynamically. This variant allows efficient scaling by controlling computational complexity through the number of transformer layers.

### WS-DETR (Width Scalable DETR)
**Width Scalable DETR with dynamic width based on weight matrix slicing** enables width scaling through  weight matrix slicing. The model can dynamically adjust the width (number of attention heads).


## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on single GPU setup)
- **GPU Memory**: ~1GB for inference, 2-4GB recommended for training
- **CUDA**: Compatible with CUDA 12.8+

### Software
- **Python**: 3.8+
- **PyTorch**: 2.8.0+cu128
- **TorchVision**: 0.13.0+
- **CUDA**: 12.8 or later

### Environment
```bash
PyTorch version: 2.8.0+cu128
CUDA available: True
GPU count: 1
```

---

## Installation

### Prerequisites
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Clone repository
git clone https://github.com/denisekhuu/scalable_detr
cd scalable-detr
```

### Dependencies
```bash
pip install -r requirements.txt
```

- torch==2.8.0
- torchvision>=0.13.0
- numpy
- pycocotools (for evaluation)

---

## Performance Benchmarks

### Width Scaling Comparison (Heads)

The following table compares performance across different numbers of attention heads:

| Heads | Baseline AP | Sliced AP | Sliced-600 AP | Baseline GMACs | Sliced GMACs |
|-------|-------------|-----------|---------------|----------------|--------------|
| 1     | 0.254       | 0.141     | 0.152         | 74.1           | 74.1         | 
| 2     | 0.343       | 0.220     | 0.231         | 74.9           | 74.9         |
| 3     | 0.377       | 0.345     | 0.353         | 76.1           | 76.1         |
| 4     | 0.388       | 0.349     | 0.357         | 77.4           | 77.4         |

- Best performance at 4 heads (AP: 0.388)
- Width slicing introduces performance degradation (Δ up to -0.123)
- Computational cost (GMACs) remains similar across configurations
- 600-epoch training improves performance across all head counts

### Depth Scaling Comparison (Layers)

The following table shows performance with increasing transformer depth:

| Layers | Layer AP | Layer-600 AP |  Layer GMACs | 
|--------|----------|--------------|--------------|
| 1      | 0.233    | 0.244        | 74.4         | 
| 2      | 0.333    | 0.344        | 75.0         |
| 3      | 0.359    | 0.368        | 75.6         |
| 4      | 0.364    | 0.373        | 76.2         | 
| 5      | 0.366    | 0.374        | 76.8         |
| 6      | 0.365    | 0.374        | 77.4         | 

---

## Architecture Details

### Backbone
- ResNet-50 pre-trained on ImageNet

### Encoder
- Multi-scale feature processing
- Positional encoding (sine-based)

### Decoder
- Transformer decoder with configurable depth
- Multi-head attention with dynamic width
- Iterative bounding box refinement

### Head
- Object classification head
- Bounding box regression head
- Auxiliary heads for intermediate layer supervision

---

## Known Limitations

1. **Width Slicing Performance**: WS-DETR shows consistent AP degradation (-0.1 to -0.12) compared to baseline
2. **Depth Saturation**: Performance plateaus beyond 5 layers with minimal improvement
3. **Training Time**: 600-epoch models require significantly longer training

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contributing

Contributions are welcome! Please follow the coding standards and ensure all tests pass before submitting pull requests.

---

---
