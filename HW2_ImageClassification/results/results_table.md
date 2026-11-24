| Approach | Trainable params | Training time (s) | Train acc (%) | Val acc (%) | Test acc (%) | Test macro-F1 (%) |
|---|---|---|---|---|---|---|
| 1. Full fine-tuning | 1,555,781 | 591 | 97.42 | 86.82 | 82.12 | 81.96 |
| 2. Classifier-head training | 628,773 | 488 | 98.95 | 89.67 | 85.42 | 85.19 |
| 3. Features + logreg (classifier=logreg) | 21,349 | 60 | 100.00 | 85.73 | 84.74 | 84.58 |
| 3. Features + rf (classifier=rf) | n/a (432,642 tree nodes) | 68 | 100.00 | 86.14 | 82.69 | 82.22 |
