| Chapter<br>Number | Chapter<br>Name               | Status             |
| ----------------- | ----------------------------- | ------------------ |
| 9                 | Logistic Regression           | :white_check_mark: |
| 10                | Support Vector Machines (SVM) | :warning:          |
| 11                | Evaluation Metrics            | :warning:          |
| 12 + 13           | Neural Networks               | :warning:          |


# Lecture 9: Logistic Regression
## Examples to try (from slides)
- Linear Decision Boundary Example⚠️
- Non-Linear Decision Boundary (Circle)⚠️
- Multi-Class Classification (Run LDB example with an extra input set)⚠️
- Regularized LogReg⚠️

## Logistic Regression
### Trivia
- Odds function -> $\frac{p}{1-p}$
- Logit -> $\log({\frac{p}{1-p}})$
- if y = 1 -> $-\log({h_\theta})$
- if y = 0 -> $-\log({1 - h_\theta})$

### Decision Boundary
- Can be non-linear, for e.g. a circular boundary
	- $h_{\theta(x)} = g({\theta_{0}}+{\theta_{1}x_{1}}+{\theta_{2}x_{2}}+{\theta_{3}x_{3}}^{2}+{\theta_{4}x_{4}}^{2})$

### Regularization

---

# Lecture 10: Support Vector Machines (SVM)
###  Intro to SVM
###  Linear Separation using SVM
###  Classifier Margin (SVM)
###  Mathematics for SVM
###  Hypothesis for SVM
###  Objective Function
###  Numerical Example: SVM
###  Hard-Margin SVM
###  Soft-Margin SVM
###  Non-linear Decision Boundary and Kernel Trick

---

# Lecture 11: Evaluation Metrics (Classification)

---

# Lecture 12+13: Neural Networks
## Lecture 12 Content
#### Intro to Neural Networks
### Building blocks of neural network
#### Perceptron
#### Activation Function
##### Example
### Stepping towards neural networks
#### Application of neural network

## Lecture 13 Content
### Neural network in practice
### Training a neural network
### Optimization
### Overfitting