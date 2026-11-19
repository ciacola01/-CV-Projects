from torchvision.datasets import OxfordIIITPet
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

# Download the dataset OxfordIIITPet from torchvision. 
def download():
    print("Downloading trainval, it could require some minutes...")
    trainval = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="category", # this is the label for the breeds
        download=True
    )
    print(f"trainval downloaded: {len(trainval)} images")

    print("Downloading test...")
    test=OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types="category",
        download=True
    )
    print(f"test downloaded: {len(test)} images")

    # Print the breeds' name to verify
    print("\nList of 37 breeds:")
    for i, name in enumerate(trainval.classes):
        print(f"    {i}: {name}")

    print(f"Saved datas in: {os.path.abspath(DATA_ROOT)}")

if __name__ == "__main__":
    download()