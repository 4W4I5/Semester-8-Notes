| Chapter<br>Number | Chapter<br>Name                                     | Status             |
| ----------------- | --------------------------------------------------- | ------------------ |
| 1                 | Introduction                                        | :white_check_mark: |
| 2                 | Machine Learning                                    | :white_check_mark: |
| 3 + 4             | Data Views                                          | :white_check_mark: |
| 5                 | Feature Engineering                                 | :warning:          |
| 6                 | High Dimensional Data                               | :warning:          |
| 6A                | Exploratory Data Analysis                           | :warning:          |
| 7                 | Univariate Linear Regression                        | :warning:          |
| 8                 | Multivariate Linear Regression & Evaluation Metrics | :warning:          |

> [!WARNING]
> made a mistake, the focus for this course is entirely on the math instead of the concept

# Lecture 1: Introduction
## **Cybersecurity**

|**Concept**|**Description**|
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cybersecurity**        | Protecting computer systems, networks, and data from unauthorized access, cyber threats, and attacks.                                                                                         |
| **Network Security**     | Protecting data in transit from interception or tampering.                                                                                                                                    |
| **Application Security** | Ensuring software is secure from vulnerabilities.                                                                                                                                             |
| **Information Security** | Safeguarding data from breaches and leaks.                                                                                                                                                    |
| **Operational Security** | Managing access controls and monitoring user activity.                                                                                                                                        |
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
	- **Rows (n)** – Also called **instances, records, transactions, feature vectors, objects, tuples**. Represents the number of observations.
	- **Columns (d)** – Also called **attributes, features, dimensions, variables, properties**. Represents the number of data features.
- Types of Data Matrices:
	- **Univariate** – Data contains a single variable.
	- **Bivariate** – Data involves two variables.
	- **Multivariate** – Data consists of multiple variables (common in ML).
- ### **Forms of Datasets**
	- Not all datasets exist in matrix form. Common types include:
		- **Sequential Data** – E.g., DNA sequences, protein sequences.
		- **Text Data** – E.g., emails, logs, documents.
		- **Time-Series Data** – E.g., stock prices, sensor readings, cyber attack logs.
		- **Image Data** – E.g., face recognition datasets, CAPTCHA images.
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

For two vectors $A = [A1,A2,...,An][A_1, A_2, ..., A_n]$ and $B = [B1,B2,...,Bn][B_1, B_2, ..., B_n]$

- ### 1. **Dot Product (Scalar Product)**
	- $A⋅B=A1⋅B1+A2⋅B2+...+An⋅Bn\mathbf{A} \cdot \mathbf{B} = A_1 \cdot B_1 + A_2 \cdot B_2 + ... + A_n \cdot B_n$
	- This results in a scalar value.
- ### 2. **Length (Euclidean Norm) of a Vector**
	- $∥A∥=A12+A22+...+An2\|\mathbf{A}\| = \sqrt{A_1^2 + A_2^2 + ... + A_n^2}$
	- It represents the magnitude or length of the vector **A**.
- ### 3. **Euclidean Distance Between Two Vectors**
	- $Distance(A,B)=(A1−B1)2+(A2−B2)2+...+(An−Bn)2\text{Distance}(\mathbf{A}, \mathbf{B}) = \sqrt{(A_1 - B_1)^2 + (A_2 - B_2)^2 + ... + (A_n - B_n)^2}$
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
	- For a dataset **X** = [x1,x2,...,xn][x_1, x_2, ..., x_n], the mean is:
		- $μ=1n∑i=1nxi\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$
	- Where $μ\mu$ is the average value of all data points.
- ### 6. **Total Variance**
	- For a dataset **X** = [x1,x2,...,xn][x_1, x_2, ..., x_n], the total variance $σ2\sigma^2$ is:
		- $σ2=1n∑i=1n(xi−μ)2\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$
	- Where:
		- $μ\mu$ is the mean of the dataset.
		- $xix_i$ is each individual data point.
		- $nn$ is the number of data points.
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

---

## **Feature Engineering**

Feature engineering is the process of transforming raw data into meaningful features that improve model performance.

### **Importance of Feature Engineering**

- **Improves Model Performance** – Enhances predictive accuracy.
- **Reduces Training Time** – Well-engineered features allow faster training.
- **Handles Data Complexity** – Helps models generalize better.
- **Prevents Overfitting/Underfitting** – Ensures better model robustness.

### **Steps in Feature Engineering**

1. **Feature Selection** – Identifying the most relevant features.
	- **Filter Methods** (e.g., correlation, mutual information)
	- **Wrapper Methods** (e.g., recursive feature elimination)
	- **Embedded Methods** (e.g., LASSO regression)
2. **Feature Transformation** – Modifying existing features.
	- Normalization, log transformation, encoding categorical variables.
3. **Feature Extraction** – Creating new features.
	- Dimensionality reduction, text/image feature extraction.
4. **Feature Creation** – Generating domain-specific features.
	- Polynomial features, date-based features.
5. **Handling Outliers, Missing Values, and Duplicates.**

---

## **Outliers**

Outliers are data points that significantly deviate from the overall dataset pattern.

### **Causes of Outliers**

- **Natural Variability** – Genuine rare occurrences in data.
- **Measurement Error** – Human or instrument error.
- **Data Processing Issues** – Incorrect data formatting.
- **Experimental Design Issues** – Sampling errors.
- **External Influences** – Fraud, extreme behaviors, or seasonal trends.

### **Impact of Outliers**

6. **Model Performance:**
	- Can bias and distort machine learning models.
	- Causes inaccurate regression coefficients.
7. **Effect on Algorithms:**
	- Distance-based methods (KNN, K-Means) are highly affected.
	- SVM decision boundaries can shift significantly.
8. **Increased Complexity:**
	- Causes overfitting and longer training times.
9. **Misleading Insights:**
	- Distorts interpretability of statistical models.

### **Outlier Detection Methods**

10. **Z-Score Method** – Detects data points that deviate from the mean by more than 3 standard deviations.
11. **Interquartile Range (IQR) Method** – Identifies values outside [Q1 - 1.5 × IQR, Q3 + 1.5 × IQR].
12. **Visualization Techniques:**
	- Boxplots, scatter plots.
13. **Machine Learning Methods:**
	- Isolation Forests for anomaly detection.

### **Handling Outliers**

- **Remove Outliers** – If they are due to errors.
- **Transform Data** – Apply log or square root transformation.
- **Use Robust Models** – Decision trees and random forests handle outliers better.
- **Cap Extreme Values** – Replace outliers with thresholds.

---

## **Missing Values**

Missing values occur when data points are unavailable or not recorded.

### **Causes of Missing Data**

- **Data Collection Issues**
- **Measurement Errors**
- **Survey Non-Responses**
- **Data Processing Errors**

### **Impact of Missing Data**

- **Incompatibility with Algorithms** – Many ML models cannot handle missing values directly.
- **Bias in Model Predictions** – Can skew results if not handled properly.
- **Loss of Data** – Removing missing values can shrink the dataset.
- **Distorted Correlations** – Can misrepresent variable relationships.

### **Handling Missing Values**

14. **Remove Missing Data** – If the percentage is small.
15. **Imputation Methods:**
	- Mean/Median/Mode Imputation.
	- KNN Imputation (Nearest Neighbors).
	- Regression-based Imputation.
	- Forward/Backward Fill (for time-series data).
16. **Flagging Missing Data** – Create a binary feature indicating missing values.

---

## **Duplicate Values**

Duplicate values occur when identical entries exist in the dataset.

### **Causes of Duplicates**

- **Data Entry Errors**
- **Data Merging Issues**
- **Web Scraping Artifacts**
- **System Errors**

### **Impact of Duplicates**

- **Misleading Statistics** – Affects mean, variance, and distributions.
- **Bias in Model Training** – Over-represents certain instances.
- **Increased Computational Costs** – Redundant processing power and storage.
- **Incorrect Clustering & Classification** – Distorts machine learning models.

### **Handling Duplicates**

- **Remove Exact Duplicates** – Identify and drop exact matches.
- **Remove Near-Duplicates** – Use fuzzy matching techniques.
- **Aggregate Data** – Summarize duplicated information.

---

## **Bias vs Variance**

### **Bias**

- Difference between predicted and actual values.
- High bias → underfitting (oversimplified models).

### **Variance**

- Model sensitivity to small fluctuations in training data.
- High variance → overfitting (too complex models).

### **Bias-Variance Tradeoff**

- **High Bias, Low Variance** – Underfits data (e.g., linear regression).
- **Low Bias, High Variance** – Overfits data (e.g., deep decision trees).
- **Goal** – Find a balance where the model generalizes well to unseen data.


---

# Lecture 6: High Dimensional Data

## **High Dimensional Data**

High-dimensional data refers to datasets with a large number of attributes (features), often exceeding the number of observations. This presents unique challenges in data analysis, visualization, and machine learning.

### **Characteristics of High-Dimensional Data**

- **High Feature Count** – Datasets often contain hundreds or thousands of attributes, making them complex to analyze.
- **Computational Complexity** – More features lead to higher processing time and memory usage.
- **Feature Redundancy** – Many attributes may be correlated, adding unnecessary complexity.
- **Data Sparsity** – High-dimensional data often contains mostly zero or missing values, making meaningful patterns difficult to extract.

### **Challenges in High-Dimensional Data**

1. **Curse of Dimensionality** – As dimensions increase, data points become sparsely distributed, reducing model effectiveness.
2. **Overfitting** – Models trained on too many features tend to memorize noise instead of learning useful patterns.
3. **High Computational Costs** – Processing and storing high-dimensional datasets require substantial computing resources.
4. **Difficulty in Visualization** – Human interpretability decreases as dimensions increase.
5. **Feature Irrelevance** – Many features may be redundant or uninformative, lowering model efficiency.

### **Techniques for Handling High-Dimensional Data**

#### **Dimensionality Reduction Techniques**

- **Principal Component Analysis (PCA)** – Converts correlated features into uncorrelated principal components.
- **Linear Discriminant Analysis (LDA)** – Focuses on maximizing class separability in classification problems.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** – Non-linear technique for high-dimensional visualization.
- **Autoencoders** – Deep learning-based compression technique for feature extraction.

#### **Feature Selection Methods**

- **Filter Methods** – Select features based on statistical measures (e.g., correlation, mutual information).
- **Wrapper Methods** – Iteratively evaluate subsets of features using machine learning models.
- **Embedded Methods** – Feature selection occurs as part of model training (e.g., LASSO regression).

#### **Regularization & Optimization Techniques**

- **L1 & L2 Regularization** – Shrinks less important feature coefficients (e.g., Ridge, LASSO regression).
- **Ensemble Methods** – Combine multiple models to improve generalization.
- **Sparse Models** – Focus on the most essential nonzero attributes.
- **Visualization Techniques** – Utilize heatmaps, scatter plots, and dimensionality reduction to interpret data.

---

## **Principal Component Analysis (PCA)**

PCA is a statistical technique used to reduce the dimensionality of data while preserving as much variance as possible.

### **Why Use PCA?**

- Reduces computational complexity and improves efficiency.
- Helps in visualizing high-dimensional data in lower dimensions.
- Mitigates overfitting by eliminating redundant features.
- Converts correlated features into uncorrelated principal components.

### **How PCA Works**

6. **Standardization** – Scale features so they contribute equally to the analysis.
7. **Compute Covariance Matrix** – Identifies relationships between variables.
8. **Compute Eigenvectors & Eigenvalues** – Determines the principal components.
9. **Create Feature Vector** – Selects principal components that explain the most variance.
10. **Recast Data Along Principal Component Axes** – Transforms data into a lower-dimensional space.

### **Detailed Steps in PCA**

#### **1. Standardization**

- Ensures all features contribute equally by rescaling them to a mean of 0 and unit variance.
- Prevents certain features from dominating due to their larger scale.

#### **2. Compute Covariance Matrix**

- Measures the variance and relationship between each pair of features.
- Helps determine whether features are correlated and can be combined.

#### **3. Compute Eigenvectors and Eigenvalues**

- Eigenvectors define the principal component directions.
- Eigenvalues represent the amount of variance captured by each principal component.

#### **4. Create Feature Vector**

- Selects the top principal components that retain most of the dataset’s variance.
- Reduces dimensionality while preserving the most important information.

#### **5. Recast Data Along Principal Components**

- Projects the original data onto the selected principal components.
- Results in a new dataset with fewer dimensions but minimal information loss.

---

## **Applications of PCA**

1. **Dimensionality Reduction** – Reduces feature count while retaining essential patterns.
2. **Data Visualization** – Enables plotting high-dimensional data in 2D or 3D.
3. **Noise Reduction** – Eliminates less significant variations, improving model robustness.
4. **Preprocessing for Machine Learning** – Reduces feature redundancy, leading to more efficient models.

---

## **Advantages of PCA**

✔ **Reduces Dimensionality** – Simplifies models, making them faster and more efficient. ✔ **Minimizes Overfitting** – Removes irrelevant features, reducing the risk of learning noise. ✔ **Captures Important Information** – Focuses on components that explain the most variance. ✔ **Creates Uncorrelated Features** – Helps improve the performance of algorithms relying on independent features. ✔ **Useful for High-Dimensional Data** – Essential when working with large datasets containing many features.

## **Disadvantages of PCA**

❌ **Loss of Interpretability** – Principal components are linear combinations of original features, making them harder to interpret. ❌ **Assumes Linearity** – PCA works best with datasets where variables exhibit linear relationships. ❌ **Loss of Information** – Some variance is lost when reducing dimensions, potentially affecting model accuracy. ❌ **Sensitive to Scaling** – Requires proper feature standardization to work effectively.

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

1. **Univariate Analysis** – Examines individual variables.

	- Summary statistics (mean, median, mode, variance, skewness, kurtosis).
	- Frequency distributions and histograms.
	- Box plots for outlier detection.
2. **Bivariate Analysis** – Examines relationships between two variables.

	- Scatter plots (continuous vs continuous data).
	- Correlation matrices and heatmaps.
	- Chi-square tests for categorical relationships.
3. **Multivariate Analysis** – Examines interactions between multiple variables.

	- Pair plots and Principal Component Analysis (PCA).
	- Cluster analysis for segmentation.
	- Regression analysis for dependency modeling.

### **Common EDA Techniques**

4. **Descriptive Statistics** – Mean, median, standard deviation, quartiles.
5. **Data Visualization** – Histograms, boxplots, violin plots, scatter plots, heatmaps.
6. **Correlation Analysis** – Pearson/Spearman correlation coefficients.
7. **Dimensionality Reduction** – PCA, t-SNE, UMAP for visualizing high-dimensional data.

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

- **PCA** – Reduces dimensionality by transforming correlated variables into uncorrelated principal components.
- **t-SNE & UMAP** – Nonlinear methods for visualization and feature reduction.
- **Text Feature Extraction** – TF-IDF, word embeddings (Word2Vec, GloVe).
- **Image Feature Extraction** – Convolutional Neural Networks (CNNs).

### **Handling Feature Redundancy**

- **Variance Thresholding:**
	- Removes features with low variance.
- **Correlation Analysis:**
	- Drops highly correlated features to avoid multicollinearity.
- **Recursive Feature Elimination (RFE):**
	- Iteratively removes less important features.

---

## **Applications of EDA & Feature Engineering**

1. **Fraud Detection** – Identifying suspicious transactions based on behavioral patterns.
2. **Cybersecurity** – Detecting network intrusions using anomaly detection techniques.
3. **Healthcare** – Predicting diseases based on patient data and clinical features.
4. **Finance** – Analyzing stock market trends and risk assessment.
5. **Natural Language Processing (NLP)** – Extracting meaningful features from text.

---

# Lecture 7: Univariate Linear Regression

---

# Lecture 8: Multivariate Linear Regression & Evaluation Metrics