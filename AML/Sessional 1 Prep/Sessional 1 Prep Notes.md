| Chapter<br>Number | Chapter<br>Name               | Status             |
| ----------------- | ----------------------------- | ------------------ |
| 1                 | Introduction                  | :white_check_mark: |
| 2                 | Machine Learning              | :white_check_mark: |
| 3 + 4             | Data Views                    | :white_check_mark: |
| 5                 | Feature Engineering           | :white_check_mark: |
| 6                 | High Dimensional Data & PCA   | :white_check_mark: |
| 6A                | Exploratory Data Analysis     | :warning:          |
| 7                 | Linear Regression             | :white_check_mark: |
| 8                 | Evaluation Parameters+Metrics | :white_check_mark: |

> [!WARNING]
> made a mistake, the focus for this course is entirely on the math instead of the concept

# Lecture 1: Introduction
## **Cybersecurity**

| **Concept**              | **Description**                                                                                                                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cybersecurity**        | Protecting computer systems, networks, and data from unauthorized access, cyber threats, and attacks.                                                                                                     |
| **Network Security**     | Protecting data in transit from interception or tampering.                                                                                                                                                |
| **Application Security** | Ensuring software is secure from vulnerabilities.                                                                                                                                                         |
| **Information Security** | Safeguarding data from breaches and leaks.                                                                                                                                                                |
| **Operational Security** | Managing access controls and monitoring user activity.                                                                                                                                                    |
| **Cyber Threats**        | - **Malware**: Malicious software (viruses, worms, trojans) <br>- **Phishing**: Tricking users into revealing credentials <br>- **DDoS**: Overloading networks <br>- **MitM**: Intercepting communication |

## **Dataset**

| **Type**         | **Description**                                                       |
| ---------------- | --------------------------------------------------------------------- |
| **Structured**   | Organized data in rows/columns (e.g., SQL databases).                 |
| **Unstructured** | Data such as text, images, and logs that require preprocessing.       |
| **Labeled**      | Data with both input and output labels (used in supervised learning). |
| **Unlabeled**    | Raw input data (used in unsupervised learning).                       |

## **Big Data**

| **5 V's**    | **Description**                                                     |
| ------------ | ------------------------------------------------------------------- |
| **Volume**   | Large amounts of data generated.                                    |
| **Velocity** | Speed at which data is collected and processed.                     |
| **Variety**  | Different data formats (structured, unstructured, semi-structured). |
| **Veracity** | Ensuring data quality and accuracy.                                 |
| **Value**    | Extracting meaningful insights from raw data.                       |

## **Data Analytics**

| **Phase**                       | **Description**                                         |
| ------------------------------- | ------------------------------------------------------- |
| **Identify**                    | Define the problem or goal.                             |
| **Data Collection**             | Gather relevant data from various sources.              |
| **Data Preprocessing**          | Clean, filter, and normalize data.                      |
| **Data Exploration**            | Analyze patterns and trends in the data.                |
| **Data Transformation**         | Convert data into a usable format.                      |
| **Data Modeling**               | Apply ML algorithms for predictions or classifications. |
| **Data Interpretation**         | Understand the results and extract insights.            |
| **Data Visualization**          | Present findings in charts, graphs, or reports.         |
| **Reporting & Decision-Making** | Use insights to make informed security decisions.       |

## **AI & Machine Learning in Cybersecurity**

| **Category**                     | **Description**                                                                                                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Artificial Intelligence (AI)** | AI in cybersecurity includes: <br> - **Automated Threat Detection** <br> - **Fraud Prevention** <br> - **Incident Response**                                                                                                                         |
| **Machine Learning (ML)**        | Key ML applications in cybersecurity: <br>- **Predictive Analysis** <br>- **Anomaly Detection** <br>- **Pattern Recognition**                                                                                                                     |
| **Deep Learning (DL)**           | Specialized branch of ML that uses neural networks with multiple layers for tasks like: <br>- **Image Recognition** <br>- **Natural Language Processing (NLP)** <br>- **Behavioral Analysis**                                                     |
| **Data Mining**                  | Extracting patterns from large datasets using techniques like classification and clustering for: <br>- **Intrusion Detection** <br>- **Malware Classification** <br>- **Phishing and Fraud Detection** <br>- **Log Analysis for Threat Intelligence** |

## **ML in Cybersecurity**

| **Use Case**                                | **Description**                                                               |
| ------------------------------------------- | ----------------------------------------------------------------------------- |
| **Intrusion Detection & Prevention**        | Identifying and blocking unauthorized access attempts.                        |
| **Malware/Phishing/Spam Detection**         | Detecting malicious emails, domains, or software using ML models.             |
| **Fraud Detection & Threat Intelligence**   | Analyzing transaction patterns to prevent fraud and detect potential threats. |
| **User & Entity Behavior Analytics (UEBA)** | Detecting anomalies in user behavior to flag insider threats.                 |
| **Automated Incident Response**             | Using AI to respond to security incidents without human intervention.         |

## **ML Algorithms in Cybersecurity**

| **Learning Type**         | **Algorithms**                   | **Description**                                                                          |
| ------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------- |
| **Supervised Learning**   | **Linear Regression**            | Predicts numerical values (e.g., attack likelihood).                                     |
|                           | **Logistic Regression**          | Classifies threats as safe or malicious.                                                 |
|                           | **Support Vector Machine (SVM)** | Separates data into distinct categories (e.g., spam vs. non-spam).                       |
|                           | **Decision Trees**               | Creates rules for classifying security incidents.                                        |
|                           | **Random Forest**                | Uses multiple decision trees for improved accuracy.                                      |
| **Unsupervised Learning** | **K-Means Clustering**           | Groups similar data points (e.g., detecting botnets in network traffic).                 |
| **Deep Learning**         | **Neural Networks**              | Used for complex cybersecurity tasks, such as malware detection and behavioral analysis. |

## **Tools & Libraries**

| **Library**                | **Purpose**                              |
| -------------------------- | ---------------------------------------- |
| **Numpy**                  | Numerical operations.                    |
| **Pandas**                 | Data manipulation and preprocessing.     |
| **Matplotlib & Seaborn**   | Data visualization.                      |
| **Scikit-learn (Sklearn)** | Standard ML algorithms.                  |
| **TensorFlow/PyTorch**     | Deep learning frameworks.                |
| **Keras**                  | Simplified deep learning model building. |

## **Companies Using ML in Cybersecurity**

| **Tech Giants**  | **Applications**                                           |
| ---------------- | ---------------------------------------------------------- |
| **Microsoft**    | AI-driven threat intelligence in Defender & Sentinel.      |
| **Google**       | Uses ML for spam filtering and malware detection in Gmail. |
| **Amazon (AWS)** | AI-powered security monitoring.                            |
| **Apple**        | Face ID and anomaly detection in security logs.            |

| **Cybersecurity Companies** | **Applications**                                      |
| --------------------------- | ----------------------------------------------------- |
| **FireEye (Mandiant)**      | Uses AI to detect APTs (Advanced Persistent Threats). |
| **Palo Alto Networks**      | AI-driven intrusion prevention systems (IPS).         |
| **Vectra AI**               | Uses ML for real-time attack detection.               |
| **Sophos**                  | AI-powered endpoint security solutions.               |

---

# Lecture 2: Machine Learning

## **Rising Cybersecurity Problems**

| **Problem**                                        | **Description**                                                        |
| -------------------------------------------------- | ---------------------------------------------------------------------- |
| **Intrusion Detection and Prevention**             | Identifying unauthorized access attempts in real-time.                 |
| **Vulnerability Management**                       | Discovering and patching weaknesses in software and systems.           |
| **Malware Detection and Classification**           | Using ML to recognize new and evolving malware threats.                |
| **Phishing Detection**                             | Identifying deceptive emails and websites designed to steal user data. |
| **Spam and Botnet Detection**                      | Filtering out malicious automated activity in networks.                |
| **Fraud Detection**                                | Analyzing transaction patterns to prevent financial fraud.             |
| **Threat Intelligence**                            | Predicting cyberattacks based on collected threat data.                |
| **User and Entity Behavior Analytics (UEBA)**      | Detecting unusual behavior in user activity logs.                      |
| **Automated Incident Response**                    | Using AI to automatically respond to cyber threats.                    |
| **Data Loss Prevention (DLP)**                     | Preventing unauthorized access to sensitive data.                      |
| **Detection of Advanced Persistent Threats (APT)** | Identifying prolonged, targeted cyberattacks.                          |
| **Detection of Hidden Channels**                   | Finding covert communication methods used by attackers.                |
| **Detection of Software Vulnerabilities**          | Predicting and mitigating software flaws before exploitation.          |

## **Machine Learning (ML) in Cybersecurity**

| **Concept**                      | **Description**                                                                                                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Machine Learning**             | ML is a subset of AI that enables computers to learn from data and make decisions without explicit programming.                                                                |
| **Why Use ML in Cybersecurity?** | - **Scalability**: Process vast amounts of data faster than humans <br>- **Feature Extraction**: Identifies key data attributes <br>- **Adaptability**: ML systems evolve with threats |
|                                  | - **Pattern Recognition**: Detects anomalies to indicate threats <br>- **Complex Problem-Solving**: Handles intricate, dynamic datasets and scenarios                              |

## **Fundamental Machine Learning Concepts**

| **Concept**                      | **Description**                                                                                                                                             |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Defining the Task (T)**        | The problem the system is solving, such as classifying emails (Classification), detecting fraud (Anomaly Detection), or predicting intrusions (Regression). |
| **Defining the Experience (E)**  | The dataset used for training, consisting of labeled (supervised) or unlabeled (unsupervised) data.                                                         |
| **Defining the Performance (P)** | Performance is evaluated using metrics like Accuracy, Precision & Recall, and F1-Score.                                                                     |

## **Machine Learning Pipeline**

| **Stage**                 | **Description**                                                              |
| ------------------------- | ---------------------------------------------------------------------------- |
| **Data Collection**       | Gathering relevant data from logs, network traffic, or security events.      |
| **Data Preprocessing**    | Cleaning, normalizing, and transforming raw data into usable formats.        |
| **Feature Engineering**   | Extracting key attributes that help in model training.                       |
| **Model Selection**       | Choosing the appropriate ML algorithm for the task.                          |
| **Training & Testing**    | Splitting data into training and testing sets to evaluate model performance. |
| **Model Optimization**    | Adjusting parameters to improve accuracy.                                    |
| **Deployment**            | Integrating the model into a live security system for real-time detection.   |
| **Continuous Monitoring** | Updating the model as new threats emerge.                                    |

## **Types of Machine Learning**

| **Type**                     | **Description**                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Supervised Learning**      | Uses labeled data for prediction (e.g., spam classification, regression for attack severity).                        |
| **Unsupervised Learning**    | Finds patterns in unlabeled data (e.g., anomaly detection, clustering).                                              |
| **Semi-Supervised Learning** | A mix of labeled and unlabeled data for tasks where labeling is expensive.                                           |
| **Reinforcement Learning**   | Learns by interacting with an environment and receiving rewards or penalties (e.g., AI-driven intrusion prevention). |

## **Learning Approaches in ML**

| **Approach**                | **Description**                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Batch Learning**          | Model is trained using the entire dataset at once and does not update itself automatically.                               |
| **Online Learning**         | The model continuously learns from incoming data, ideal for real-time monitoring (e.g., live threat detection).           |
| **Instance-Based Learning** | Memorizes specific instances and makes decisions based on similarity (e.g., K-Nearest Neighbors for intrusion detection). |
| **Model-Based Learning**    | Builds a general model and makes predictions based on training data (e.g., Support Vector Machines for classification).   |

## **Supervised vs. Unsupervised ML Pipelines**

| **Pipeline Type**            | **Description**                                                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Supervised ML Pipeline**   | - Collect labeled cybersecurity data (e.g., logs with attack labels) <br>- Preprocess and clean the data <br>- Train a classification model (e.g., Decision Trees, SVM) |
|                              | - Evaluate accuracy using Precision, Recall, F1-Score <br>- Deploy model for real-time threat detection.                                                            |
| **Unsupervised ML Pipeline** | - Collect raw, unlabeled data (e.g., network traffic) <br>- Apply clustering or anomaly detection (e.g., K-Means)                                                   |
|                              | - Identify patterns of unusual activity for proactive threat hunting.                                                                                           |

## **Example: Cybersecurity Application of ML**

| **Use Case**                   | **Description**                                                   |
| ------------------------------ | ----------------------------------------------------------------- |
| **Intrusion Detection System** | Analyzes network packets to classify traffic as normal or attack. |
| **Phishing Detection**         | Analyzes email content to flag suspicious messages.               |
| **AI-Driven Firewalls**        | Continuously learn from attack patterns to adapt to new threats.  |

---

# Lecture 3 + 4: Data Views
# Analytic View
## **Data Matrix (A Dataset Representation)**
- A **data matrix** is a structured representation of data in an **n × d** format:
	- **Rows (n)** : Also called **instances, records, transactions, feature vectors, objects, tuples**. Represents the number of observations.
	- **Columns (d)** : Also called **attributes, features, dimensions, variables, properties**. Represents the number of data features.
- Types of Data Matrices:
	- **Univariate** : Data contains a single variable.
	- **Bivariate** : Data involves two variables.
	- **Multivariate** : Data consists of multiple variables (common in ML).
- ### **Forms of Datasets**
	- Not all datasets exist in matrix form. Common types include:
		- **Sequential Data** : E.g., DNA sequences, protein sequences.
		- **Text Data** : E.g., emails, logs, documents.
		- **Time-Series Data** : E.g., stock prices, sensor readings, cyber attack logs.
		- **Image Data** : E.g., face recognition datasets, CAPTCHA images.
		- **Audio & Video Streams** – E.g., voice command recognition, surveillance footage.
	- **Raw data is transformed into structured datasets using feature extraction techniques.**

## **Attributes, Analytics, and Machine Learning**

Attributes are **variables or features** used for **data analysis and machine learning.** They can be classified as:

## **Nominal Attributes (Categorical & Unordered)**

| **Aspect**                | **Details**                                                                                                                                                                                                                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**           | Nominal attributes are categorical without inherent order.                                                                                                                                                                                                                                                     |
| **Examples**              | - Colors: {Red, Green, Blue} <br>- Gender: {Male, Female, Non-binary} <br>- Countries: {USA, Canada, Mexico}                                                                                                                                                                                                           |
| **Analytics**             | - **Frequency Distribution**: Shows frequency of each category. <br>- **Mode**: Identifies most common category (useful for missing values). <br>- **Cross-Tabulation (Contingency Tables)**: Analyzes relationships between categories. <br>- **Chi-Square Test**: Determines associations between categorical variables. |
| **Encoding for ML**       | - **Label Encoding**: Converts categories to numerical values. <br>- **One-Hot Encoding**: Converts categories into binary columns. <br>- **Binary Encoding**: Combines label encoding and one-hot encoding. <br>- **Target Encoding**: Replaces categories with the mean of the target variable.                          |
| **Feature Engineering**   | - **Combining Categories**: Merging similar groups (e.g., “Bachelor’s” & “Master’s” into “Higher Education”). <br>- **Creating Interaction Features**: Deriving new features from existing nominal attributes.                                                                                                     |
| **Visualization Methods** | - **Bar Charts**: Displays category distribution. <br>- **Pie Charts**: Displays proportions of each category.                                                                                                                                                                                                     |
| **Insights**              | - **Category-Specific Statistics**: Understanding major classes. <br>- **Predictive Power**: Evaluating importance for ML models. <br>- **Anomaly Detection**: Identifying outliers in categorical data.                                                                                                               |

## **Ordinal Attributes (Categorical & Ordered)**

| **Aspect**                | **Details**                                                                                                                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**           | Ordinal attributes have a defined order but unequal differences between values.                                                                                                                                         |
| **Examples**              | - Education Levels: {High School, Bachelor’s, Master’s, PhD} <br>- Customer Satisfaction: {Very Unsatisfied, Neutral, Satisfied, Very Satisfied} <br>- Movie Ratings: {1 star, 2 stars, 3 stars, 4 stars, 5 stars}              |
| **Analytics**             | - **Frequency Distribution**: Identifies dominant categories. <br>- **Mode, Median, Percentiles**: Measures central tendency. <br>- **Spearman Rank Correlation**: Analyzes relationships between ordinal & numeric attributes. |
| **Encoding for ML**       | - **Ordinal Encoding**: Assigns numeric values based on order. <br>- **Target Encoding**: Uses mean of target variable for each category.                                                                                   |
| **Feature Engineering**   | - **Binning/Grouping**: Merging similar categories. <br>- **Interaction Features**: Creating new variables based on ordinal data.                                                                                           |
| **Visualization Methods** | - **Bar Charts**: Displays distribution of ordinal attributes. <br>- **Histograms**: Shows frequency distribution of ordinal values.                                                                                        |
| **Insights**              | - **Trend Analysis**: Evaluates patterns over time. <br>- **Correlation with Target Variable**: Helps in feature selection.                                                                                                 |

## **Interval-Scaled Attributes (Continuous & No True Zero)**

| **Aspect**                   | **Details**                                                                                                                                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Description**              | Interval-scaled attributes have equal differences between values but no absolute zero.                                                                                                                                                                 |
| **Examples**                 | - Temperature in Celsius/Fahrenheit <br>- Dates & Time Intervals                                                                                                                                                                                           |
| **Analytics**                | - **Mean, Median, Standard Deviation**: Measure central tendency & spread. <br>- **Pearson Correlation Coefficient**: Assesses relationships between interval attributes. <br>- **Covariance**: Indicates directional relationships.                           |
| **Feature Scaling for ML**   | - **Z-score Normalization**: Converts values to a mean of 0 and standard deviation of 1. <br>- **Min-Max Scaling**: Rescales values between [0,1]. <br>- **IQR Method**: Identifies outliers using interquartile range.                                        |
| **Feature Engineering**      | - **Polynomial Features**: Deriving new features using powers of existing attributes. <br>- **Interaction Terms**: Capturing dependencies between attributes.                                                                                              |
| **Dimensionality Reduction** | - **PCA (Principal Component Analysis)**: Reduces feature dimensions while preserving variance.                                                                                                                                                        |
| **Visualization Methods**    | - **Histograms**: Displays the frequency of data distribution. <br>- **Box Plots**: Visualizes data spread and detects outliers. <br>- **Scatter Plots**: Shows relationships between two interval attributes. <br>- **Line Plots**: Tracks data trends over time. |
| **Insights**                 | - **Trends & Patterns**: Identifies shifts in cybersecurity data. <br>- **Outliers & Anomalies**: Detects security breaches.                                                                                                                               |

## **Ratio-Scaled Attributes (Continuous & True Zero Exists)**

| **Aspect**                 | **Details**                                                                                                                                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**            | Ratio-scaled attributes have a true zero point, meaning zero represents no existence of the measured quantity.                                                                                                                          |
| **Examples**               | - Height <br>- Weight <br>- Age <br>- Income                                                                                                                                                                                                        |
| **Analytics**              | - **Mean, Median, Standard Deviation**: Measure central tendency & dispersion. <br>- **Correlation & Covariance**: Identifies dependencies between attributes.                                                                              |
| **Feature Scaling for ML** | - **Z-score Normalization, Min-Max Scaling, IQR Method**: Applied similarly to interval-scaled attributes.                                                                                                                              |
| **Feature Engineering**    | - **Polynomial & Interaction Features**: Used for improving ML models.                                                                                                                                                                  |
| **Predictive Modeling**    | - **Regression Analysis (Linear & Logistic)**: Uses ratio attributes for prediction.                                                                                                                                                    |
| **Anomaly Detection**      | - **Identifying Outliers**: Detects unusual data points (e.g., fraudulent transactions).                                                                                                                                                |
| **Visualization Methods**  | - **Histograms**: Displays distribution and frequency. <br>- **Box Plots**: Visualizes data spread and detects outliers. <br>- **Scatter Plots**: Shows relationships between ratio attributes. <br>- **Line Plots**: Tracks data trends over time. |
| **Insights**               | - **Relative Measures & Ratios**: Useful for comparative analysis. <br>- **Descriptive & Predictive Insights**: Provides valuable trends in cybersecurity.                                                                                  |

# **Algebraic & Geometric View of Data**
## **Key Concepts:**

| **Aspect**                                      | **Details**                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vector & Matrix Representation**              | - Data is represented as vectors and matrices, allowing easier mathematical transformations and optimizations.                                                                                                                                                                                                                                                                                                                |
| **Linear Algebra Operations**                   | - **Matrix multiplication**: Applied in neural networks, dimensionality reduction, and data transformations. <br>- **Matrix inversion**: Solves systems of equations, particularly in regression. <br>- **Feature transformations**: Involves scaling, rotation, and projection of data for enhanced model performance.                                                                                                               |
| **Dimensionality Reduction**                    | - **Principal Component Analysis (PCA)**: Projects high-dimensional data into lower dimensions while retaining maximum variance. <br>- **Singular Value Decomposition (SVD)**: Factorizes a matrix into three matrices to extract important features. <br>- **Latent Semantic Analysis (LSA)**: Used in NLP to analyze relationships between words and documents.                                                                     |
| **Feature Scaling & Normalization**             | - **Min-Max Scaling**: Rescales values between a fixed range [0,1] or [-1,1] to improve model performance. <br>- **Z-score Normalization**: Converts data to a normal distribution with mean 0 and standard deviation 1. <br>- **Logarithmic Scaling**: Used for skewed data to reduce the impact of extreme values.                                                                                                                  |
| **Optimization & Model Training**               | - **Gradient Descent**: Optimizes model parameters by iteratively reducing the error. <br>- **Stochastic Gradient Descent (SGD)**: A variant of gradient descent that updates parameters for each training instance, useful for large datasets. <br>- **Newton’s Method**: Used in convex optimization problems for rapid convergence. <br>- **Least Squares Optimization**: Minimizes the sum of squared residuals in regression models. |
| **Distance Measures**                           | - **Euclidean Distance**: Measures straight-line distance between two points. <br>- **Manhattan Distance**: Measures distance along axis-aligned paths, useful for grid-like structures. <br>- **Mahalanobis Distance**: Accounts for correlations between variables and scales accordingly. <br>- **Cosine Similarity**: Measures angular similarity between vectors, often used in text mining and NLP.                                 |
| **Orthogonal Projection & Linear Independence** | - **Orthogonality**: Ensures features are independent of each other, improving interpretability and reducing redundancy. <br>- **Linear Independence**: Ensures that no feature can be represented as a linear combination of other features.                                                                                                                                                                                     |
| **Kernel Methods & Feature Transformations**    | - **Kernel Trick**: Projects data into higher-dimensional space for non-linear classification. <br>- **Polynomial & Radial Basis Function (RBF) Kernels**: Transformations that allow support vector machines to work effectively with non-linear data.                                                                                                                                                                           |

## Formulae

For two vectors $A = [A_1, A_2, ..., A_n]$ and $B = [B_1, B_2, ..., B_n]$

- ### 1. **Dot Product (Scalar Product)**
	- $A \cdot B = A_{1} \cdot B_1 + A_{2}\cdot B_{2} +...+A_{n}\cdot B_{n}$
	- This results in a scalar value.
- ### 2. **Length (Euclidean Norm) of a Vector**
	- $\|\mathbf{A}\| = \sqrt{\sum_{i=1}^{m}a_{i}^2}$
	- It represents the magnitude or length of the vector **A**.
- ### 3. **Euclidean Distance Between Two Vectors**
	- $\|\mathbf{A - B}\| = \sqrt{\sum_{i=1}^{m}(a_{i}-b_{i})^2}$
	- This gives the straight-line distance between the two points in the space.
- ### 4. **Angle (Similarity and Orthogonality)**
	- #### **Cosine Similarity** (Angle between two vectors):
		- $\text{Cosine Similarity}=A⋅B∥A∥⋅∥B∥$
		- $\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}$
			- If the cosine similarity is **1**, the vectors are in the same direction.
			- If it is **0**, the vectors are orthogonal (perpendicular).
			- If it is **-1**, the vectors are in opposite directions.
	- #### **Angle between two vectors** (in radians):
		- $θ=\cos⁡−1(A⋅B∥A∥⋅∥B∥)\theta = \cos^{-1}\left(\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}\right)$
- ### 5. **Mean of a Dataset**
	- For a dataset **X** = $[x_1, x_2, ..., x_n]$, the mean is:
		- $\mathbf{\mu} = \frac{1}{n}|\sum_{i=1}^{n}\mathbf{X}_{i}$
	- Where $\mu$ is the average value of all data points.
- ### 6. **Total Variance**
	- For a dataset **X** = $[x_1, x_2, ..., x_n]$, the total variance $\sigma^2$ is:
		- $\mathbf{\sigma^2} = \frac{1}{n}\sum_{i=1}^{n}{\|{X}_{i}-\mu\|}^2$
	- Where:
		- $\mu$ is the mean of the dataset.
		- $x_i$ is each individual data point.
		- $n$ is the number of data points.
- ### 7. **Centered Data Matrix**
	- For a dataset **X** of shape **n × d** (n samples, d features), the centered data matrix **X_centered** is obtained by subtracting the mean of each feature (column) from each data point:
		- $Xcentered=X−μX_{\text{centered}} = X - \mu$
	- Where:
		- **X** is the original data matrix.
			- μ\mu is the vector of means of each feature (column).
		- For each column jj:
			- $μj=1n∑i=1nXij\mu_j = \frac{1}{n} \sum_{i=1}^{n} X_{ij}$
		- Where $XijX_{ij}$ is the value in the **i-th** row and **j-th** column of the dataset.


## **Probabilistic View of Data**

| **Concept**                                         | **Description**                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data as a Random Variable**                       | Attributes modeled as random variables enable statistical inferences about populations from sampled data. Used in Bayesian networks and probabilistic graphical models.                                                                                                                                                                         |
| **Distribution Modeling**                           | **Normal Distribution (Gaussian)** – Used for statistical modeling and ML algorithms like Naive Bayes. **Poisson Distribution** – Used for event-based modeling (e.g., cybersecurity attack frequency prediction). **Exponential Distribution** – Useful in failure rate analysis. **Dirichlet Distribution** – Used in topic modeling for NLP. |
| **Feature Selection & Regularization**              | **Mutual Information** – Measures dependency between variables. **Information Gain** – Used in decision trees and entropy-based models. **L1 & L2 Regularization** – Lasso and Ridge regression techniques that prevent overfitting.                                                                                                            |
| **Overfitting Prevention**                          | **Bayesian Regularization** – Introduces prior distributions to limit model complexity. **Cross-Validation** – Ensures generalization by testing model performance on unseen data. **Dropout in Neural Networks** – Prevents co-adaptation of neurons, reducing overfitting.                                                                    |
| **Decision Theory & Risk Management**               | **Probability-based decision-making** – Bayesian inference applied to cybersecurity threat detection. **Markov Decision Processes (MDP)** – Used in reinforcement learning for decision-making in uncertain environments. **Handling Uncertainty** – Using probabilistic confidence intervals and Monte Carlo simulations.                      |
| **Handling Missing Data**                           | **Mean/Median Imputation** – Replacing missing values using statistical averages. **Expectation-Maximization (EM) Algorithm** – Probabilistic technique to handle missing data and clustering.                                                                                                                                                  |
| **Bayesian Inference & Uncertainty Quantification** | **Bayesian Theorem** – Updates the probability of a hypothesis as more evidence is provided. **Monte Carlo Methods** – Used in probabilistic sampling to model uncertainty in predictions.                                                                                                                                                      |

## **Graph View of Data**

| **Concept**                       | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Graph Representation**          | **Nodes (Vertices):** Represent entities such as users, IP addresses, or devices. **Edges (Links):** Represent relationships between nodes (e.g., network connections). **Adjacency Matrix:** Matrix representation of a graph capturing connectivity between nodes.                                                                                                                                                                                                                                                    |
| **Graph Metrics**                 | **Degree Distribution** – Measures the number of connections per node. **Shortest Path & Betweenness Centrality** – Determines influential nodes in a network. **Clustering Coefficient** – Measures the tendency of nodes to form tightly-knit groups. **PageRank** – Used in search engine ranking and network analysis.                                                                                                                                                                                              |
| **Machine Learning Applications** | **Graph Neural Networks (GNNs)** – Learn embeddings for nodes in a graph. **Social Network Analysis** – Understanding connections and detecting fake accounts. **Community Detection** – Identifies clusters of related nodes (e.g., fraud rings). **Anomaly Detection in Networks** – Identifies unusual behaviors using graph structures. **Link Prediction** – Predicts future connections (e.g., friend suggestions on social media). **Recommendation Systems** – Suggests items based on user interaction graphs. |

## **Benchmark Datasets for Cybersecurity**

Benchmark datasets are commonly used for training and evaluating machine learning models in cybersecurity.

- ### **Intrusion Detection and Prevention**
	- **KDD Cup 1999** – [https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
	- **NSL-KDD** – [https://www.kaggle.com/datasets/hassan06/nslkdd](https://www.kaggle.com/datasets/hassan06/nslkdd)
	- **CICIDS 2017** – [https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
- ### **Malware Detection and Classification**
	- **MalwareBazaar** – [https://bazaar.abuse.ch/](https://bazaar.abuse.ch/)
	- **Ember Dataset** – [https://github.com/elastic/ember](https://github.com/elastic/ember)
	- **Malware Traffic Analysis** – [https://www.malware-traffic-analysis.net](https://www.malware-traffic-analysis.net/)
- ### **Phishing Detection**
	- **PhishTank Dataset** – [https://www.phishtank.com/](https://www.phishtank.com/)
	- **Labeled Phishing URLs Dataset** – [https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
	- **APWG eCrime Exchange (eCX)** – [https://apwg.org/the-apwg-ecrime-exchange-ecx/](https://apwg.org/the-apwg-ecrime-exchange-ecx/)
- ### **Fraud Detection**
	- **Credit Card Fraud Detection Dataset** – [https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
	- **IEEE-CIS Fraud Detection** – [https://www.kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)
- ### **User and Entity Behavior Analytics (UEBA)**
	- **CERT Insider Threat Dataset** – [https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research](https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research)
	- **LANL User Authentication Dataset** – [https://csr.lanl.gov/data/](https://csr.lanl.gov/data/)
- ### **Spam and Botnet Detection**
	- **Enron Email Dataset** – [https://www.kaggle.com/datasets/wcukierski/enron-email-dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
	- **Botnet Dataset** – [https://www.stratosphereips.org/datasets-ctu13](https://www.stratosphereips.org/datasets-ctu13)
	- **SpamAssassin Public Corpus** – [https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus](https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus)

---


# Lecture 5: Feature Engineering

**Feature Engineering – Outliers, Missing Values, Duplicates, and Bias-Variance Tradeoff**

## **Feature Engineering**

Feature engineering is the process of transforming raw data into meaningful features that improve model performance.

- ### **Importance of Feature Engineering**
	- **Improves Model Performance** : Enhances predictive accuracy.
	- **Reduces Training Time** : Well-engineered features allow faster training.
	- **Handles Data Complexity** : Helps models generalize better.
	- **Prevents Overfitting/Underfitting** : Ensures better model robustness.
- ### **Steps in Feature Engineering**
	1. **Feature Selection** : Identifying the most relevant features.
		- **Filter Methods** (e.g., correlation, mutual information)
		- **Wrapper Methods** (e.g., recursive feature elimination)
		- **Embedded Methods** (e.g., LASSO regression)
	2. **Feature Transformation** : Modifying existing features.
		- Normalization, log transformation, encoding categorical variables.
	3. **Feature Extraction** : Creating new features.
		- Dimensionality reduction, text/image feature extraction.
	4. **Feature Creation** : Generating domain-specific features.
		- Polynomial features, date-based features.
	5. **Handling Outliers, Missing Values, and Duplicates.**

## **Outliers**

| **Topic**                        | **Details**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Outliers – What?**             | Outliers are data points that differ significantly from the rest of the dataset. They are extreme high or low values that usually do not follow the general trend of the data.                                                                                                                                                                                                                                                                                                                                          |
| **Causes of Outliers**           | - **Variability of Data**: Natural or rare events. <br> -**Measurement Error**: Human or instrument error. <br> -**Data Pre-processing Issues**: Errors during data cleaning. <br> -**Experimental Design**: Small sample sizes, sampling errors, or out-of-scope sampling. <br> -**Changes Over Time**: Trends, seasonal effects. <br> -**Intentional Factors**: Fraud, extreme behavior. <br> -**External Influences**: Environmental or external factors.                                                                                    |
| **Impact on Model Performance**  | - **Bias and Variance**: Outliers can distort the relationship between features and target variables, leading to bias and increased variance. This makes the model less generalizable. <br> -**Prediction Accuracy**: Outliers can skew the model training, affecting predictions, especially in regression models, where they may disproportionately affect the slope.                                                                                                                                                     |
| **Effect on Algorithms**         | - **Linear Models**: Outliers heavily influence model parameters. <br> -**Distance-Based Algorithms**: Algorithms like K-Nearest Neighbors and K-Means can have distorted clusters or neighbors due to outliers. <br> -**Support Vector Machines**: Outliers near the decision boundary can significantly shift the boundary.                                                                                                                                                                                                   |
| **Training Time & Complexity**   | - **Increased Complexity**: Models may overfit to outliers, increasing the number of parameters and training time. <br> -**Longer Training Time**: Algorithms such as K-means or SVM may take longer to converge or fail to converge at all.                                                                                                                                                                                                                                                                                |
| **Impact on Evaluation Metrics** | Outliers can disproportionately influence metrics such as Mean Squared Error (MSE) or Mean Absolute Error (MAE).                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Interpretability**             | Outliers can lead to misleading insights or incorrect feature importance rankings in interpretive models like linear regression or decision trees.                                                                                                                                                                                                                                                                                                                                                                      |
| **Outlier Detection (Methods)**  | - **Z-Score Method**: Identifies outliers based on how many standard deviations a data point is away from the mean. <br> -**IQR Method**: Identifies outliers as data points outside the range [Q1 − 1.5 × IQR , Q3 + 1.5 × IQR]. <br> -**Visualization**: Boxplots and scatter plots can help detect outliers. <br> -**Isolation Forest**: A machine learning method that isolates anomalies by randomly selecting features and splitting data points. |
| **Handling Outliers**            | - **Removing Outliers**: Outliers may be removed, particularly if they are due to errors. <br> -**Transforming Data**: Applying transformations like log or square root can reduce the impact of outliers. <br> -**Using Robust Algorithms**: Algorithms like decision trees or random forests are less affected by outliers. <br> -**Capping or Imputation**: Outliers can be replaced with a threshold value (capping) or imputed with more reasonable values.                                                                    |

### Outlier Formulae
#### Z-Standardization
$\mathbf{Z} = \frac{(X - \mu)}{\sigma}$
- Requires
	- Precalculated $\mu$ and $\sigma$
#### IQR
- Already know this, its based off of the medians
## **Missing Values**

| **Category**                 | **Details**                                                                                                                                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What are Missing Values?** | Missing values are data points that are not recorded or unavailable in a dataset.                                                                                                                  |
|                              | - Missing values occur when information that should be present for certain observations is absent.                                                                                                 |
| **Causes of Missing Values** | Missing values can occur due to the following reasons:                                                                                                                                             |
|                              | - **Data Collection Issues**: Issues during the process of gathering data.                                                                                                                         |
|                              | - **Measurement Error**: Human or instrument errors causing missing data.                                                                                                                          |
|                              | - **Data Processing Issues**: Problems that occur during the data processing phase.                                                                                                                |
|                              | - **Irrelevant Information**: Data is intentionally omitted because it is deemed unnecessary.                                                                                                      |
|                              | - **Survey or Interview Refusals**: Missing data from respondents refusing to answer certain questions.                                                                                            |
| **Impact of Missing Values** | Missing values can significantly affect data analysis and model performance.                                                                                                                       |
|                              | - **Incompatibility with Algorithms**: Many algorithms (e.g., linear regression, logistic regression, KNN) cannot handle null values directly and require imputation or removal of missing values. |
|                              | - **Bias in Model Predictions**: Incorrect imputation methods (e.g., using the mean without considering other factors) can lead to biased predictions.                                             |
|                              | - **Loss of Data**: Removing rows or columns with missing values can result in a significant loss of data, reducing the size of the training dataset and potentially leading to overfitting.       |
|                              | - **Misleading Insights**: If missing values are not handled appropriately, they can distort interpretations of the data.                                                                          |
|                              | - **Incomplete Data and Loss of Context**: Missing data can lead to incomplete datasets, impacting context and analysis.                                                                           |
|                              | - **Distorted Correlation and Covariance**: Missing values can skew correlations or covariances between variables, misrepresenting their true relationship.                                        |
|                              | - **Breaks in Continuity of Time-Series Data**: Missing values can disrupt the continuity of time-series data, causing issues in forecasting or trend analysis.                                    |
| **Handling Missing Values**  | Various methods to handle missing values:                                                                                                                                                          |
|                              | - **Removing Null Values**: Removing rows or columns that contain missing values, though this may lead to significant data loss.                                                                   |
|                              | - **Imputation**: Filling in missing values using estimates or statistical methods:                                                                                                                |
|                              | - **Mean/Median Imputation**: Replace missing values with the mean or median of the feature.                                                                                                       |
|                              | - **Mode Imputation**: For categorical variables, replace missing values with the mode (most frequent value).                                                                                      |
|                              | - **K-Nearest Neighbors (KNN) Imputation**: Impute missing values using the average (or weighted average) of the nearest neighbors.                                                                |
|                              | - **Regression Imputation**: Predict missing values using regression models based on other features in the dataset.                                                                                |
|                              | - **Forward/Backward Fill**: In time-series data, impute missing values using the previous or next observed value.                                                                                 |
|                              | - **Flagging Missing Data**: Create binary features that flag whether a value is missing, allowing models to account for missing data patterns.                                                    |


## **Duplicate Values**

| **Category**             | **Details**                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Causes of Duplicates** | Duplicates occur when identical entries exist in the dataset.                                                                                         |
|                          | - **Data Entry Errors**: Mistakes made during manual or automated data entry.                                                                         |
|                          | - **Data Merging Issues**: Duplicates can arise when datasets are merged incorrectly, often due to overlapping or similar entries.                    |
|                          | - **Web Scraping Artifacts**: Scraping errors when collecting data from websites lead to duplicate entries.                                           |
|                          | - **System Errors**: Bugs or issues in data processing systems can generate duplicates.                                                               |
| **Impact of Duplicates** | Duplicates can severely affect data analysis and machine learning model performance.                                                                  |
|                          | - **Distorted Summary Statistics**: Skewed statistics like mean, median, standard deviation, and percentiles.                                         |
|                          | - **Misleading Insights**: Duplicates can lead to incorrect or overly simplistic interpretations of data.                                             |
|                          | - **Bias in Model Training**: Disproportionate influence on model predictions, leading to less accurate models.                                       |
|                          | - **Misleading Accuracy Metrics**: Inflated evaluation metrics (accuracy, precision, recall, F1 score).                                               |
|                          | - **Data Redundancy**: Increased dataset size without adding new information, leading to slower processing and higher storage costs.                  |
|                          | - **Inconsistent Results**: Analysis or reporting inconsistencies, especially in metrics like sales or revenue, due to duplication.                   |
|                          | - **Data Cleaning Effort**: Presence of duplicates suggests poor data quality, leading to a greater need for cleaning and preprocessing.              |
|                          | - **Less Diverse Training Data**: Reduced diversity, limiting model generalization.                                                                   |
|                          | - **Incorrect or Overlapping Clusters**: In clustering algorithms (e.g., K-means, DBSCAN), duplicates distort clusters, causing inaccurate groupings. |
|                          | - **Disrupted Time-Series Analysis**: Duplicates can disrupt the temporal sequence, leading to faulty forecasts.                                      |
|                          | - **Effects on Business and Decision-Making**: Misrepresentation of KPIs, leading to incorrect resource allocation and business strategies.           |
|                          | - **Misleading and False Associations**: Duplicates can cause false associations in classification and forecasting models.                            |
|                          | - **Increased Computational and Storage Cost**: Handling duplicates increases computational and storage requirements.                                 |
| **Duplicate Detection**  | Methods for detecting duplicate values in a dataset include:                                                                                          |
|                          | - **Direct Inspection**: Visually inspecting the data for repeated or identical entries.                                                              |
|                          | - **Summary Statistics**: Using statistical measures (e.g., counts and frequency distributions) to identify duplicates.                               |
|                          | - **Visualizations**: Creating visualizations (e.g., histograms, scatter plots) to highlight duplicates and data anomalies.                           |
| **Handling Duplicates**  | Techniques for handling duplicates:                                                                                                                   |
|                          | - **Keep First/Last Occurrence**: Retain only the first or last occurrence of each duplicate.                                                         |
|                          | - **Fuzzy Matching**: Use fuzzy matching algorithms (like Levenshtein distance) to detect and remove slight variations in data.                       |
|                          | - **Remove Exact Duplicates**: Identify and drop exact duplicate records, leaving only unique entries.                                                |
|                          | - **Remove Near-Duplicates**: Use fuzzy matching to remove near-identical records (e.g., spelling errors or slight variations).                       |
|                          | - **Aggregate Data**: Instead of removing duplicates, summarize the data, such as by averaging or summing values to represent them as a single entry. |

## **Bias vs Variance**

- ### **Bias**
	- Difference between predicted and actual values.
	- High bias -> underfitting (oversimplified models).
	- Low bias -> makes fewer assumptions, captures complex patterns in data
- ### **Variance**
	- Model sensitivity to small fluctuations in training data.
	- High variance -> overfitting (too complex models).
	- Low variance -> more stable, generalizes well to unseen data
- ### **Bias-Variance Tradeoff**
	- **High Bias, Low Variance** : Underfits data (e.g., linear regression).
	- **Low Bias, High Variance** : Overfits data (e.g., deep decision trees).
	- **Goal** : Find a balance where the model generalizes well to unseen data.
		- A model must be low bias enough to capture complex data patterns but in moderation as to not overfit the training data
			- Both variance and bias must be low. Set bias to a high enough point where variance is not too high

---

# Lecture 6: High Dimensional Data & PCA

## **High Dimensional Data**

High-dimensional data refers to datasets with a large number of attributes (features), often exceeding the number of observations. This presents unique challenges in data analysis, visualization, and machine learning.

- ### **Characteristics of High-Dimensional Data**
	- **High Feature Count** : Datasets often contain hundreds or thousands of attributes, making them complex to analyze. More columns(D) than rows (n)
	- **Computational Complexity** : More features lead to higher processing time and memory usage.
	- **Feature Redundancy** : Many attributes may be correlated, adding unnecessary complexity.
	- **Data Sparsity** : High-dimensional data often contains mostly zero or missing values, making meaningful patterns difficult to extract.
- ### **Challenges in High-Dimensional Data**
	1. **Curse of Dimensionality** : As dimensions increase, data points become sparsely distributed, reducing model effectiveness.
	2. **Overfitting** : Models trained on too many features tend to memorize noise instead of learning useful patterns.
	3. **High Computational Costs** : Processing and storing high-dimensional datasets require substantial computing resources.
	4. **Difficulty in Visualization** : Human interpretability decreases as dimensions increase.
	5. **Feature Irrelevance** : Many features may be redundant or uninformative, lowering model efficiency.
- ### **Techniques for Handling High-Dimensional Data**
	- #### **Dimensionality Reduction Techniques**
		- **Principal Component Analysis (PCA)** : Converts correlated features into uncorrelated principal components.
		- **Linear Discriminant Analysis (LDA)** : Focuses on maximizing class separability in classification problems.
		- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** : Non-linear technique for high-dimensional visualization.
		- **Autoencoders** : Deep learning-based compression technique for feature extraction.
	- #### **Feature Selection Methods**
		- **Filter Methods** : Select features based on statistical measures (e.g., correlation, mutual information).
		- **Wrapper Methods** : Iteratively evaluate subsets of features using machine learning models.
		- **Embedded Methods** : Feature selection occurs as part of model training (e.g., LASSO regression).
	- #### **Regularization & Optimization Techniques**
		- **L1 & L2 Regularization** : Shrinks less important feature coefficients (e.g., Ridge, LASSO regression).
		- **Ensemble Methods** : Combine multiple models to improve generalization.
		- **Sparse Models** : Focus on the most essential nonzero attributes.
		- **Visualization Techniques** : Utilize heatmaps, scatter plots, and dimensionality reduction to interpret data.

## **Principal Component Analysis (PCA)**

PCA is a statistical technique used to reduce the dimensionality of data while preserving as much variance as possible.

- ### **Why Use PCA?**
	- Reduces computational complexity and improves efficiency.
	- Helps in visualizing high-dimensional data in lower dimensions.
	- Mitigates overfitting by eliminating redundant features.
	- Converts correlated features into uncorrelated principal components.
- ### **How PCA Works**
	1. **Standardization** : Scale features so they contribute equally to the analysis.
	2. **Compute Covariance Matrix** : Identifies relationships between variables.
	3. **Compute Eigenvectors & Eigenvalues** : Determines the principal components.
	4. **Create Feature Vector** : Selects principal components that explain the most variance.
	5. **Recast Data Along Principal Component Axes** : Transforms data into a lower-dimensional space.
- ### **Detailed Steps in PCA**
	- #### **1. Standardization**
		- Ensures all features contribute equally by rescaling them to a mean of 0 and unit variance.
		- Prevents certain features from dominating due to their larger scale.
		- Use of Z-Standardization $\mathbf{Z}=\frac{X - \mu}{\sigma}$
	- #### **2. Compute Covariance Matrix**
		- Measures the variance and relationship between each pair of features.
		- Helps determine whether features are correlated and can be combined.
			- $\begin{bmatrix}Cov(X,X) & Cov(X,Y) \\ Cov(Y,X) & Cov(Y,Y)\end{bmatrix}$
		- $Cov(-,-)$ Function
			- $\mathbf{Covariance} = \frac{\sum(X - \bar{X})\times(Y-\bar{Y})}{N}$
	- #### **3. Compute Eigenvectors and Eigenvalues**
		- Eigenvectors define the principal component directions.
		- Eigenvalues represent the amount of variance captured by each principal component.
		- Done by solving $\text{det}(A-\lambda I) = 0$
			- Will result in a quadratic equation that provides the required eigenvalues. Choose the non-negative one
		-
	- #### **4. Create Feature Vector**
		- Selects the top principal components that retain most of the dataset’s variance.
		- Reduces dimensionality while preserving the most important information.
	- #### **5. Recast Data Along Principal Components**
		- Projects the original data onto the selected principal components.
		- Results in a new dataset with fewer dimensions but minimal information loss.
		- Done by multiplying the two
			- Take transpose of X to ensure that it can be multiplied by the eigenvector


## **Applications of PCA**

1. **Dimensionality Reduction** : Reduces feature count while retaining essential patterns.
2. **Data Visualization** : Enables plotting high-dimensional data in 2D or 3D.
3. **Noise Reduction** : Eliminates less significant variations, improving model robustness.
4. **Preprocessing for Machine Learning** : Reduces feature redundancy, leading to more efficient models.


- ### **Advantages of PCA**
	- ✔ **Reduces Dimensionality** : Simplifies models, making them faster and more efficient.
	- ✔ **Minimizes Overfitting** : Removes irrelevant features, reducing the risk of learning noise.
	- ✔ **Captures Important Information** : Focuses on components that explain the most variance.
	- ✔ **Creates Uncorrelated Features** : Helps improve the performance of algorithms relying on independent features.
	- ✔ **Useful for High-Dimensional Data** : Essential when working with large datasets containing many features.
- ### **Disadvantages of PCA**
	- ❌ **Loss of Interpretability** : Principal components are linear combinations of original features, making them harder to interpret.
	- ❌ **Assumes Linearity** : PCA works best with datasets where variables exhibit linear relationships.
	- ❌ **Loss of Information** : Some variance is lost when reducing dimensions, potentially affecting model accuracy.
	- ❌ **Sensitive to Scaling** : Requires proper feature standardization to work effectively.

---


# Lecture 6A: Exploratory Data Analysis

## **Exploratory Data Analysis (EDA)**

EDA is the process of analyzing datasets to summarize their main characteristics, often using visual methods. It helps in identifying patterns, relationships, and anomalies in data.

### **Objectives of EDA**

- Understand data structure and distribution.
- Detect missing values, outliers, and inconsistencies.
- Identify patterns, trends, and correlations.
- Provide insights for feature engineering and model selection.

### **Types of EDA**

1. **Univariate Analysis** : Examines individual variables.

	- Summary statistics (mean, median, mode, variance, skewness, kurtosis).
	- Frequency distributions and histograms.
	- Box plots for outlier detection.
2. **Bivariate Analysis** : Examines relationships between two variables.

	- Scatter plots (continuous vs continuous data).
	- Correlation matrices and heatmaps.
	- Chi-square tests for categorical relationships.
3. **Multivariate Analysis** : Examines interactions between multiple variables.

	- Pair plots and Principal Component Analysis (PCA).
	- Cluster analysis for segmentation.
	- Regression analysis for dependency modeling.

### **Common EDA Techniques**

4. **Descriptive Statistics** : Mean, median, standard deviation, quartiles.
5. **Data Visualization** : Histograms, boxplots, violin plots, scatter plots, heatmaps.
6. **Correlation Analysis** : Pearson/Spearman correlation coefficients.
7. **Dimensionality Reduction** : PCA, t-SNE, UMAP for visualizing high-dimensional data.

### **Handling Missing Data**

- **Deletion Methods:**
	- Listwise deletion (removing rows with missing values).
	- Pairwise deletion (using available data without removing entire rows).
- **Imputation Methods:**
	- Mean/median/mode substitution.
	- KNN-based imputation.
	- Multiple imputation using regression models.

### **Outlier Detection in EDA**

- **Statistical Methods:**
	- Z-score (values beyond 3 standard deviations from the mean).
	- IQR method (values outside Q1 - 1.5_IQR and Q3 + 1.5_IQR).
- **Visualization-Based:**
	- Boxplots, scatter plots, density plots.
- **Machine Learning-Based:**
	- Isolation Forests, One-Class SVM, DBSCAN clustering.

---

## **Feature Engineering**

Feature engineering involves creating, modifying, and selecting the best features to improve model performance.

### **Feature Selection Methods**

8. **Filter Methods:**
	- Uses statistical techniques to rank features (e.g., correlation, mutual information).
9. **Wrapper Methods:**
	- Uses machine learning models to iteratively evaluate subsets of features (e.g., recursive feature elimination).
10. **Embedded Methods:**
	- Feature selection is integrated into the model training process (e.g., LASSO regression, decision trees).

### **Feature Transformation Techniques**

1. **Scaling & Normalization:**
	- Min-Max Scaling (scales values between 0 and 1).
	- Z-score Standardization (centers data around mean 0 with standard deviation 1).
2. **Encoding Categorical Variables:**
	- One-hot encoding (for nominal variables).
	- Label encoding (for ordinal variables).
	- Target encoding (replaces categories with mean target value).
3. **Polynomial Features:**
	- Creating interaction terms for non-linear relationships.
4. **Log Transformation:**
	- Reduces skewness in highly skewed data.

### **Feature Extraction Methods**

- **PCA** : Reduces dimensionality by transforming correlated variables into uncorrelated principal components.
- **t-SNE & UMAP** : Nonlinear methods for visualization and feature reduction.
- **Text Feature Extraction** : TF-IDF, word embeddings (Word2Vec, GloVe).
- **Image Feature Extraction** : Convolutional Neural Networks (CNNs).

### **Handling Feature Redundancy**

- **Variance Thresholding:**
	- Removes features with low variance.
- **Correlation Analysis:**
	- Drops highly correlated features to avoid multicollinearity.
- **Recursive Feature Elimination (RFE):**
	- Iteratively removes less important features.

---

## **Applications of EDA & Feature Engineering**

1. **Fraud Detection** : Identifying suspicious transactions based on behavioral patterns.
2. **Cybersecurity** : Detecting network intrusions using anomaly detection techniques.
3. **Healthcare** : Predicting diseases based on patient data and clinical features.
4. **Finance** : Analyzing stock market trends and risk assessment.
5. **Natural Language Processing (NLP)** : Extracting meaningful features from text.

---

# Lecture 7: Linear Regression (LinReg)
## Univariate /Multivariate LinReg
- ### Cost function
	- $J_{\theta} = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_{i} - y_{i})^2$
- ### Gradient Descent (Param Update function)
	- $\theta_{j} = \theta_{j} - \alpha \times \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^i)-y^{i})\times{x^i}$
- ### Procedure (Fitting)
	- Init Vectors for theta as parameters and Input Features as X
		- Note that $x_0$ is always set to 1
		- $m$ is known as the members of the dataset
	- Repeat the following until the hyperparameter `iterations` stops the fitting process
		- Compute hypothesis i.e. run the function for the given values of X. The result is called a hypothesis
		- Run the Cost function $J(\theta)$
			- Note that this includes all thetas
		- Update params via Gradient Descent
			- Use a learning rate ($\alpha$) of `0.01` to `0.1`
---

# Lecture 8: Evaluation Parameters+Metrics
## Evaluation Parameters
- ### In-sample
	- Evaluates how well the model has learned the training dataset
	- **Adv:** Helps detect underfitting
		- Provides a measure of model fitting
	- **Disadv:** Easy to overestimate model performance
		- Does not indicate how well the model generalizes new unseen data
- ### Out-sample
	- Evaluates how well the model generalizes based on unseen data i.e. data it was not trained on
	- **Adv:** Helps detect overfitting
		- Provides a realistic measure of model performance
	- **Disadv:** Performance is highly-dependent on test data
		- Small datasets are not representative of real-world performance

## Evaluation Metrics
- ### Mean Absolute Error (MAE)
	- $\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}|y_{i}- \hat{y_i}|$
- ### Mean Squared Error (MSE)
	- $\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}(y_{i}- \hat{y_i})^2$
	- Sensitive to larger errors
	- Provides a scaled up value of the error
- ### Root Mean Squared Error (RMSE)
	- $\text{RMSE} = \sqrt(MSE)$
	- Scales MSE back down to original scale of data
- ### R-Squared (R2)
	- $R^{2}= 1 - \frac{\sum_(y_{i}- \hat{y}_i)^2}{\sum_(y_{i}- \bar{y}_i)^2}$
	- Proportion of variance in the dependent variable that is predictable from the independent variables
- ### Adjusted R-Squared (R2adj)
	- $R_{adj}^{2} = 1 - (1 - R^{2}) \times \frac{n - 1}{n - k -1}$
	- Where:
		- $R^2$ is the R2 value
		- $n$ is the number of observations
		- $k$ is the number of predictors i.e. independent variables in the model
	- Adjusts R2 for number of predictors
- ### Mean Absolute Percentage Error (MAPE)
	- $MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$
	- Where:
		- $y_i$ is the actual value (observed value) for the i-th data point,
		- $\hat{y}_i$ is the predicted value for the i-th data point,
		- $n$ is the total number of observations (data points).