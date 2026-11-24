# Homework 2 – Image Classification

Transfer learning with **MobileNetV3-Small** (ImageNet-1K weights) on the **Oxford-IIIT Pet** dataset, 37 breed classes. Three strategies are compared with the same data, splits, preprocessing and metrics:
1. **Full fine-tuning**: replace the last layer, train the whole network.
2. **Classifier-head training**: replace the last layer, freeze the convolutional `features`, train `classifier.0` and `classifier.3` only.
3. **Feature extraction + scikit-learn**: frozen 576-d features after the pooling layer, `StandardScaler` fitted on train, then logistic regression and random forest.

## Dataset
Oxford-IIIT Pet Dataset: O. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, _Cats and Dogs_, CVPR 2012. Website: https://www.robots.ox.ac.uk/~vgg/data/pets/ · License: **CC BY-SA 4.0**.

The images are **not** included in this repository. `prepareData.py` downloads them automatically with `torchvision.datasets.OxfordIIITPet(download=True)` into `data/` (about 800 MB, internet connection needed the first time).

Protocol: the official `trainval` partition (3,680 images) is split 80/20 with stratification by breed (seed 42) into train (2,944) and validation (736). The official `test` partition (3,669 images) is used only once, for the final evaluation. The split indexes are saved in `splits/split.json`; per-breed counts are in `results/class_counts.csv`.

## Environment
Versions used for the reported results (also saved in `results/environment.json`):

|Package|Version|
|---|---|
|Python|3.15.2|
|PyTorch|2.15.0|
|TorchVision|0.29.2|
|scikit-learn|1.9.1|
|NumPy|2.4.6|
|Device|Apple Silicon GPU (`mps`), macOS 27.2.1|

Other dependencies: pandas, matplotlib, joblib.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r HW2_ImageClassification/requirements.txt
cd HW2_ImageClassification/src
```

The code runs on CUDA, MPS or CPU (chosen automatically in `utils.get_device()`). 

## Repository structure
```
README.md
requirements.txt
src/            all the Python code (run the commands from this folder)
data/           dataset (downloaded, not in the repository)
splits/         split.json: train/validation indexes
features/       features.npz: frozen 576-d features (approach 3)
checkpoints/    selected weights (approaches 1 and 2) and scikit-learn pipelines (approach 3)
results/        metrics (json/csv), tables and figures
```

|File in `src/`|Purpose|
|---|---|
|`config.py`|seed, paths, image size, batch size, default hyperparameters|
|`utils.py`|seeds, determinism, device, versions|
|`data.py`|transforms, dataset, stratified split, DataLoaders|
|`model.py`|MobileNetV3-Small with 37 outputs, freezing, feature extractor|
|`engine.py`|training epoch and evaluation (shared by approaches 1 and 2)|
|`checkEnvironment.py`|saves package versions in `results/environment.json`|
|`prepareData.py`|downloads the dataset, creates and checks the split, class counts|
|`trainNetwork.py`|training of approach 1 (`--mode full`) and approach 2 (`--mode head`)|
|`extractFeatures.py`|feature extraction for approach 3|
|`trainSklearn.py`|logistic regression and random forest on the features|
|`evaluateTest.py`|final evaluation on the test set (run once)|
|`makeResults.py`|results table and all figures|

## How to reproduce
From the `HW2_ImageClassification/src/` folder, in this order:

```bash
python checkEnvironment.py                 # versions -> results/environment.json
python prepareData.py                      # download, split, class counts

python trainNetwork.py --mode full         # approach 1 (lr 1e-4, 10 epochs)
python trainNetwork.py --mode head         # approach 2 (lr 1e-3, 15 epochs)

python extractFeatures.py                  # approach 3: features
python trainSklearn.py                     # approach 3: logreg + random forest

python evaluateTest.py                     # test set, ONCE
python makeResults.py                      # table and figures
```
