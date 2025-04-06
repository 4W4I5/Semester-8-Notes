| Chapter<br>Number | Chapter<br>Name               | Status             |
| ----------------- | ----------------------------- | ------------------ |
| 9                 | Logistic Regression           | :white_check_mark: |
| 10                | Support Vector Machines (SVM) | :warning:          |
| 11                | Evaluation Metrics            | :warning:          |
| 12 + 13           | Neural Networks               | :warning:          |

# Lecture 9: Logistic Regression
## Classification
- Supervised machine learning, goal is to predict a category/class label based on input features
- Discrete output, the input feature set is mapped to one discrete class/category
- Types
	- Binary
		- LogiReg, predict for 1 of 2 possible classes
	- Multi-Class
		- LogiReg/SVM, predict for 1 of Y possible classes
	- Multi-Label
		- Input set is matched to X of Y possible classes
			- A photo that contains both cats + dogs
## Examples to try (from slides)
- Linear Decision Boundary Example✅
- Non-Linear Decision Boundary (Circle)❌
- Multi-Class Classification (Run LDB example with an extra input set)✅
- Regularized LogReg⚠️

## Logistic Regression
### Trivia
- Odds function -> $\frac{p}{1-p}$
- Logit -> $\log({\frac{p}{1-p}})$

### Decision Boundary
- Can be non-linear, for e.g. a circular boundary
	- $h_{\theta(x)} = g({\theta_{0}}+{\theta_{1}x_{1}}+{\theta_{2}x_{2}}+{\theta_{3}x_{3}}^{2}+{\theta_{4}x_{4}}^{2})$

### Cost function
- $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$
	- The first hypothesis calc can be simplified by the following ruleset
		- if y = 1 -> $-\log({h_\theta})$
		- if y = 0 -> $-\log({1 - h_\theta})$

### Gradient Descent
- $\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$

## Regularized Logistic regression (Regularization)
### Cost function, modified
- $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$
	- Y conditions are same as before (the first hypothesis calc can be simplified by the following ruleset)
		- if y = 1 -> $-\log({h_\theta})$
		- if y = 0 -> $-\log({1 - h_\theta})$

### Gradient Descent, modified
- $\theta_{0} := \theta_{0} - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_{0}^{(i)}\right)$
	- Theta 0 is treated differently
- $\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right)$
	- A term is added to the end, $\frac{\lambda}{m} \theta_j$

---

# Lecture 10: Support Vector Machines (SVM)
##  Intro to SVM
##  Linear Separation using SVM
##  Classifier Margin (SVM)
##  Mathematics for SVM
##  Hypothesis for SVM
##  Objective Function
##  Numerical Example: SVM
##  Hard-Margin SVM
##  Soft-Margin SVM
##  Non-linear Decision Boundary and Kernel Trick

---

# Lecture 11: Evaluation Metrics (Classification)

---

# Lecture 12+13: Neural Networks
## Lecture 12 Content
### Intro to Neural Networks
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