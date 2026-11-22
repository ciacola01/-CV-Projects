| Approach | Trainable params | Training time (s) | Train acc (%) | Val acc (%) | Test acc (%) | Test macro-F1 (%) |
|---|---|---|---|---|---|---|
| 2. Classifier-head training | 628,773 | 1288 | 98.68 | 89.40 | 85.58 | 85.51 |
| 3. Features + logreg (classifier=logreg) | 21,349 | 113 | 100.00 | 85.73 | 84.74 | 84.58 |
| 3. Features + rf (classifier=rf) | n/a (432,660 tree nodes) | 115 | 100.00 | 85.87 | 82.75 | 82.27 |
