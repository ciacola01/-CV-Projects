"""
Checks that the environment works and save the versions
"""

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from utils import save_versions, set_seed

def main() -> None:
    set_seed()
    info = save_versions()
    for k, v in info.items():
        print(f"{k:28s} {v}")

    # Load MobileNetV3-Small with ImageNet weights
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    print("\nUsed weights:", weights)

    # The last layer produces 1000 outputs and we substitute with 37
    print("\n Original classifier: \n", model.classifier)

    # Check with fake image
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    print("\nOutput form", tuple(out.shape), " (expected: (1, 1000))")

if __name__ == "__main__":
    main()