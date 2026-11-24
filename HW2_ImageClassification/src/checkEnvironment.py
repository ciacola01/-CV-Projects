"""
Saves the versions of Python, PyTorch, TorchVision, scikit-learn and the device used
in results/environment.json (needed for reproducibility)
"""

from utils import save_versions, set_seed

def main() -> None:
    set_seed()
    info = save_versions()
    for k, v in info.items():
        print(f"{k:28s} {v}")

if __name__ == "__main__":
    main()