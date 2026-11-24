---
tags: ComputerVision
cssclasses:
  - img-grid
---
<center>
	<b>Recchia Luca - VR563083</b> <br>
	<a href="https://github.com/ciacola01/-CV-Projects.git">Repository GitHub</a>
</center>

---

# Introduction
This report compares three ways of adapting an ImageNet-pretrained MobileNetV3-Small to the 37-breed Oxford-IIIT Pet classification task:
1. full fine-tuning
2. training only the fully connected classifier head on a frozen convolutional backbone
3. extracting frozen features and training a scikit-learn classifier on them. 

Data, splits, preprocessing are identical across the three approaches, so that differences in performance and cost can be attributed to how much of the network is adapted. 

---
# Dataset and split protocol
**Source and license**: Oxford-IIIT Pet dataset (Parkhi, Vedaldi, Zisserman, Jawahar) https://www.robots.ox.ac.uk/~vgg/data/pets/, license CC BY-SA 4.0. 
The images are not redistributed: the code downloads them with `torchvision.datasets.OxfordIIITPet(download=True)` into `data/`

**Split**: The official **trainval** partition ($3\ 680$ images) is split $80 / 20$ with stratification by breed (`train_test_split`, seed 42) into train $(2 \ 944)$ and validation $(736)$. 
The official **test** partition ($3\ 669$ images) is never used for training or model selection. 
The split indexes are saved in `splits/split.json`. 

**Class counts**: per-breed counts for every split are in `results/class_counts.csv`.  
The table shows for every class id the relative breed and the partitioned in train, val and test with range $[74, 80]$ in train, $[19, 20]$ in val and $[88, 100]$. 
The dataset is nearly balanced, so no class-balancing technique (weighted loss, resampling) was applied, and macro-F1 is reported next to accuracy. 

| class_id | breed             | train | val | test | total |
| -------- | ----------------- | ----- | --- | ---- | ----- |
| 0        | Abyssinian        | 80    | 20  | 98   | 198   |
| 1        | American Bulldog  | 80    | 20  | 100  | 200   |
| ...      | ...               | ...   | ... | ...  | 200   |
| 35       | Wheaten Terrier   | 80    | 20  | 100  | 200   |
| 36       | Yorkshire Terrier | 80    | 20  | 100  | 200   |


**Preprocessing** Evaluation transform (validation and test, taken from `MobileNet_V3_Small_Weights.DEFAULT.transforms()`): resize of the short side to 256, center crop to $224 \times 224$, conversion to tensor, normalization with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). 
Training transform: RandomResizedCrop(224, scale 0.7-1.0), random horizontal flip (p = 0.5), ColorJitter (brightness, contrast, saturation 0.2), then the same tensor conversion and normalization. Augmentation is applied only to the training set. 

---
# Model and training setup
**Architecture and weights** MobileNetV3-Small from `torchvision.models.mobilenet_v3_small` with `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet-1K). The network is made of `features` (convolutional part, 576 output channels), `avgpool` (global average pooling, producing a 576-dimensional vector) and `classifier`: Linear(576-1024), Hardswish, Dropout(0.2), Linear(1024-1000). 
The last layer is replaced by Linear $(1024 \to 37)$ in all approaches.

| **Approach**               | **What is trained**                                                                                                                                          | **Trainable parameters**                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| 1. Full fine-tuning        | All layers (convolutional features and classifier)                                                                                                           | $1\ 555\ 781$                                                                    |
| 2. Classifier head         | Only `classifier.0` (576->1024) and `classifier.3` (1024->37); the whole `features` block is frozen, including its BatchNorm layers, which stay in eval mode | $628\ 773$                                                                       |
| 3. Features + scikit-learn | Nothing in the network. Frozen 576-dimensional vector after the pooling layer, standardized with a `StandardScaler` fitted on train only, then a classifier  | Logistic regression $21\ 349$; random forest: no weights ($432\ 642$ tree nodes) |

**Setup per approach**

|                     | Full fine-tuning                       | Head Training                          | Features + sklearn                          |
| ------------------- | -------------------------------------- | -------------------------------------- | ------------------------------------------- |
| Loss                | Cross-entropy                          | Cross-entropy                          | Logistic loss (logreg) / Gini (RF)          |
| Optimizer           | AdamW (weight decay 1e-4)              | AdamW (weight decay 1e-4)              | lbfgs (LogReg), default RF                  |
| Learning rate       | 1e-4                                   | 1e-3                                   | n/a                                         |
| Batch size / epochs | 32 / 10                                | 32 / 15                                | n/a                                         |
| Selection rule      | epoch with highest validation macro-F1 | epoch with highest validation macro-F1 | classifier with highest validation macro-F1 |

Hyperparameters of the third approach: logistic regression `C=1`, `max_iter=2000`; random forest 300 trees.
Learning rates and epochs are the defaults in `config.py`; the reported checkpoint is the epoch with the highest validation macro-F1. 

**Environment and reproducibility.** Python 3.15.2, PyTorch 2.15.0, TorchVision 0.29.2, scikit-learn 1.9.1, NumPy 2.4.6, device mps (macOS 27.2.1, arm64) (see `results/environment.json`). Seed 42 is set for Python, NumPy and PyTorch; the DataLoader uses a seeded generator and seeded workers; `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled. Remaining sources of variation: some MPS kernels are not bit-wise deterministic, so reruns or other machines can differ slightly; all results come from a single run (one seed).

---
# Strategies and trainable parameters
**Approach 1 - Full fine-tuning**: All parameters are updated, including the convolutional layers and the BatchNorm statistics. A small learning rate is used to avoid destroying the pretrained features. 

**Approach 2 - classifier-head training**: The convolutional feature extractor is frozen (`requires_grad=False`, BatchNorm kept in eval mode). The trainable tensors are `classifier.0.weight`, `classifier.0.bias`, `classifier.3.weight`, `classifier.3.bias` that is equivalent to $576 \cdot 1024 +1024 = 590\ 848$ parameters in the hidden layer plus $1024 \cdot 37 + 37 = 37 \ 925$ in the new output layer, for a total of $628\ 773$ trainable parameters. 

**Approach 3 - feature extraction + traditional classifier**: The network is used only once as a fixed feature extractor. Logistic Regression has $37 \cdot 576 + 37 = 21\ 349$ parameters; the Random Forest has $432\ 642$ tree nodes in total and no weights. The features are the same ones that approach 2 feeds to its classifier head, so approaches 2 and 3 differ in the classifier, in the training protocol and in the augmentation. 

---

# Results
Each selected model was evaluated once on the test set. Training time for the third approach is feature extraction on train+val plus classifier fitting. Times were measured on Apple Silicon (MPS) and include the first epoch warm-up. 
Selected checkpoints: epoch 9 of 10 (approach 1) and epoch 13 of 15 (approach 2); for approach 3 the random forest has the higher validation macro-F1 (86.12% vs 85.81% for logistic regression), so it is the validation-selected classifier, and both are reported. 

| Approach                                 | Trainable params            | Training time (s) | Train acc (%) | Val acc (%) | Test acc (%) | Test macro-F1 (%) |
| ---------------------------------------- | --------------------------- | ----------------- | ------------- | ----------- | ------------ | ----------------- |
| 1. Full fine-tuning                      | $1 \ 555 \ 781$             | 591               | 97.42         | 86.82       | 82.12        | 81.96             |
| 2. Classifier-head training              | $628 \ 773$                 | 488               | 98.95         | 89.67       | 85.42        | 85.19             |
| 3. Features + logreg (classifier=logreg) | $21 \ 349$                  | 60                | 100.00        | 85.73       | 84.74        | 84.58             |
| 3. Features + rf (classifier=rf)         | n/a ($432\ 642$ tree nodes) | 68                | 100.00        | 86.14       | 82.69        | 82.22             |

Training curves: loss and accuracy per epoch, train vs validation with the selected epoch marked

![[training_curves.png|400]]

The confusion matrix and the relative example for every approach is available on the `results table`

| **Confusion approach 1 full**     | **Confusion approach 2 head**     | **Confusion approach 3 Random Forest** | **Confusion approach 3 Logistic Regression** |
| --------------------------------- | --------------------------------- | -------------------------------------- | -------------------------------------------- |
| ![[confusion_approach1_full.png]] | ![[confusion_approach2_head.png]] | ![[confusion_approach3_rf.png]]        | ![[confusion_approach3_logreg.png]]          |


Then some example for every approach with the correct (top) and incorrect (bottom) predictions

| **Examples Approach 1 full**     | **Examples approach 2 head**     | **Examples approach 3 Random Forest** | **Examples approach 3 Logistic Regression** |
| -------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------------- |
| ![[examples_approach1_full.png]] | ![[examples_approach2_head.png]] | ![[examples_approach3_rf.png]]        | ![[examples_approach3_logreg.png]]          |


--- 

# Discussion
**Performance**: Classifier-head training  is best on the test set ($85.42\%$ accuracy and $85.19\%$ macro-F1), followed by Logistic Regression $(84.74\% / 84.58\%)$ and the Random Forest $(82.69\% / 82.22 \%)$; while the full fine-tuning has $82.12\%$ for the test accuracy and $81.96\%$ for the macro-F1. 
The $95\%$ binomial interval for an accuracy near $0.85$ on $3\ 669$ test images is about $\pm 1.2$ percentage points. 
$$
1.96 \cdot \sqrt{\frac{p\cdot(1-p)}{n} }
$$
The $0.7$ point difference between head training and Logistic Regression is not meaningful, while the $2.7$ point gap to the Random Forest exceeds this interval; all results come from one seed and one split. 
Finally the gap between head and full fine-tuning is by $3.3$ points. 
Accuracy and macro-F1 differ by less than $0.5$ points in every row, consistent with a near balanced test set. 
The Random Forest was selected on validation, although Logistic Regression scored higher on test. 

**Computational cost**: the first approach updates all the parameters and takes $591\ s$; the second approach trains $628\ 773$ parameters and it took $488 \ s$ for $15$ epochs, about $45\%$  less time per epoch, because every image still goes through the frozen backbone and the augmentation pipeline. 
The third approach needs a single forward pass over train and validation images and then a classifier fit of $0.2\ s$ (logistic regression) or $8.0\ s$ (Random Forest), with no backpropagation 60 s and 68 s in total, about 7–10 times less than approaches 1 and 2. Logistic regression gives up 0.7 test-accuracy points relative to head training (within noise) for about an eighth of the time.

**Errors** Most mistakes are between visually similar breeds, and two groups of errors recur in all four models: American Pit Bull Terrier ↔ Staffordshire Bull Terrier / American Bulldog and Birman ↔ Ragdoll. Other confusions are model-specific.. The weakest classes by recall are American Pit Bull Terrier (36–53% depending on the model) and Staffordshire Bull Terrier (51–64% in approaches 1 and 3). These breeds differ in subtle traits (head shape, coat) and there are only about 80 training images per breed. Errors across species are a minority but not absent: 57–94 of 535–656 mistakes (10–14%), for example Chihuahua predicted as Sphynx by the random forest (12 images).

**Train–test gap.** Approach 1: train 97.42% vs test 82.12% (15.3 points). Approach 2: 98.95% vs 85.42% (13.5 points). Logistic regression: 100.00% vs 84.74% (15.3 points). Random forest: 100.00% vs 82.69% (17.3 points). Every model fits the 2,944 training images almost perfectly and loses 13–17 points on new images, which is overfitting to a small training set (about 80 images per breed); the random forest, which memorizes the training set, has the largest gap and the lowest test score among the approach 3 models. Training accuracy is measured without augmentation. Validation accuracy is also 1–5 points above test for all models (for example 89.67% vs 85.42% in approach 2); part of this is selection on validation, and the rest may come from the difference between the validation split and the official test split, which we did not test. ImageNet-1K already contains many dog breeds and some cats, so source and target domains are similar: frozen features are already discriminative, and head training and logistic regression reach about 85% without changing the backbone. Full fine-tuning did not improve on them in our run (lr 1e-4, 10 epochs).

**Limitations.** One seed and one split, so no variance estimate. Hyperparameters were tuned only lightly (a single learning rate per approach (1e-4 full, 1e-3 head) and no sweep, so the lower score of full fine-tuning may depend on this choice). Timing depends on the hardware and on MPS warm-up, so it is comparable across approaches but not across machines. Only the small MobileNetV3 variant and one dataset were tested; conclusions on the value of fine-tuning may differ with larger domain shift.

---

# References
1.  O. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. Cats and Dogs. CVPR 2012.
2.  A. Howard et al. Searching for MobileNetV3. ICCV 2019.
3.  A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.
4.  F. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 2011.