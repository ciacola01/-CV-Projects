"""
MobileNetV3-Small for the breeds

Keypoints:
- pretrained weights ImageNet-1K
- Substitution of the last layer of classifier (from 1000 to 37 outputs)
- Possibility to freeze the feature extractor (2nd approach)
- utility to count and list trainable parameters

Structure of the network
- model.features: convolutive part
- model.avgpool: feature map to vector of 576
- model.classifier: fully connected part
    [0] Linear (576 -> 1024)
    [1] Hardwish
    [2] Dropout(0.2)
    [3] Linear(1024 -> 1000)
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from config import NUM_CLASSES

# Built the network with ImageNet weights ans a new last layer
def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        freeze_feature_extractor(model)
    return model

# Freeze the convolutive part and remain trainabile all the classifier
def freeze_feature_extractor(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

# Set the model in train mode respecting the freezing
def set_train_mode(model, freeze_backbone: bool) -> None:
    model.train()
    if freeze_backbone:
        model.features.eval()

# Return trainable and total parameters with relative names
def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def trainable_layer_names(model) -> list:
    return [name for name, p in model.named_parameters() if p.requires_grad]

def build_feature_extractor():
    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    extractor = nn.Sequential(base.features, base.avgpool, nn.Flatten())
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()
    return extractor

# Test of building two models
if __name__ == "__main__":
    for label, freeze in [("Approach 1 (full fine-tuning)", False), ("Approach 2 (only classifier)", True),]:
        m = build_model(freeze_backbone=freeze)
        tr, tot = count_trainable_params(m)
        print(f"\n=== {label} ===")
        print(f"Trainable parameters: {tr:,} on {tot:,}")
        if freeze:
            print("Trainable layers", trainable_layer_names(m))
    m =build_model()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    print("\nOutput form:", tuple(out.shape), "(expected (2, 37))")
    assert out.shape == (2, NUM_CLASSES), f"Unexpected form: {out.shape}"
    print("Correct forward pass")