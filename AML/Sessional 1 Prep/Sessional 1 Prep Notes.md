| Chapter<br>Number | Chapter<br>Name           | Status             |
| ----------------- | ------------------------- | ------------------ |
| 1                 | Introduction              | :white_check_mark: |
| 2                 | Machine Learning          | :white_check_mark: |
| 3 + 4             | Data Views                | :warning:          |
| 5                 | Feature Engineering       | :warning:          |
| 6                 | High Dimensional Data     | :warning:          |
| 7                 | Exploratory Data Analysis | :warning:          |

# Lecture 1: Introduction
## **Key Terms & Concepts**
- ### **Cybersecurity**
	- Cybersecurity refers to the practice of protecting computer systems, networks, and data from unauthorized access, cyber threats, and attacks. It includes multiple domains:
		- **Network Security** – Protecting data in transit from interception or tampering.
		- **Application Security** – Ensuring software is secure from vulnerabilities.
		- **Information Security** – Safeguarding data from breaches and leaks.
		- **Operational Security** – Managing access controls and monitoring user activity.
	- Common **cyber threats** include:
		- **Malware** – Malicious software like viruses, worms, and trojans.
		- **Phishing** – Social engineering attacks that trick users into revealing credentials.
		- **DDoS (Distributed Denial of Service)** – Overloading a network to disrupt services.
		- **MitM (Man-in-the-Middle)** – Intercepting communication between two parties.
- ### **Dataset**
	- A dataset is a structured collection of related data points used for analysis, research, or decision-making. In cybersecurity, datasets contain logs, network traffic, or malware samples used for training ML models.
	- Types of datasets:
		- **Structured** – Organized in rows/columns (e.g., SQL databases).
		- **Unstructured** – Text, images, and logs that require preprocessing.
		- **Labeled** – Contain both input and output labels for supervised learning.
		- **Unlabeled** – Only raw input data, used in unsupervised learning.
- ### **Big Data**
	- Big Data refers to extremely large and complex datasets that require specialized tools to process. It is defined by the **5 V’s**:
		1. **Volume** – Massive amounts of data generated.
		2. **Velocity** – The speed at which data is collected and processed.
		3. **Variety** – Different data formats (structured, unstructured, semi-structured).
		4. **Veracity** – Ensuring data quality and accuracy.
		5. **Value** – Extracting meaningful insights from raw data.
	- Big Data is crucial for cybersecurity, as it enables large-scale threat detection, anomaly detection, and fraud prevention.
- ### **Data Analytics**
	- Data analytics is the process of extracting useful insights from raw data. It involves multiple phases:
		1. **Identify** – Define the problem or goal.
		2. **Data Collection** – Gather relevant data from various sources.
		3. **Data Preprocessing** – Clean, filter, and normalize data.
		4. **Data Exploration** – Analyze patterns and trends in the data.
		5. **Data Transformation** – Convert data into a usable format.
		6. **Data Modeling** – Apply ML algorithms for predictions or classifications.
		7. **Data Interpretation** – Understand the results and extract insights.
		8. **Data Visualization** – Present findings in charts, graphs, or reports.
		9. **Reporting & Decision-Making** – Use insights to make informed security decisions.
## **AI & Machine Learning**
- ### **Artificial Intelligence (AI)**
	- **AI Applications in Cybersecurity:**
		- **Automated Threat Detection** – Identifying cyber threats in real time.
		- **Fraud Prevention** – Recognizing anomalies in transactions.
		- **Incident Response** – Automating security responses to attacks.
- ### **Machine Learning (ML)**
	- **Key ML Applications in Cybersecurity:**
		- **Predictive Analysis** – Forecasting potential security incidents.
		- **Anomaly Detection** – Identifying unusual network behavior.
		- **Pattern Recognition** – Detecting malware signatures or phishing attempts.
- ### **Deep Learning (DL)**
	- DL is a specialized branch of ML that uses neural networks with multiple layers to process complex data. It is effective for:
		- **Image Recognition** – Identifying malicious code structures.
		- **Natural Language Processing (NLP)** – Detecting phishing emails.
		- **Behavioral Analysis** – Profiling attackers based on past activity.
- ### **Data Mining**
	- Data mining involves extracting meaningful patterns from large datasets using statistical and computational techniques.
		- **Steps in Data Mining:**
			1. **Data Exploration** – Understanding the dataset characteristics.
			2. **Pattern Discovery** – Finding correlations and anomalies.
			3. **Applying Algorithms** – Using classification, clustering, regression, etc.
			4. **Model Evaluation** – Measuring accuracy and performance.
			5. **Deployment & Interpretation** – Using insights for real-world applications.
		- **Cybersecurity Applications of Data Mining:**
			- Intrusion detection
			- Malware classification
			- Phishing and fraud detection
			- Log analysis for threat intelligence
## **ML in Cybersecurity**
- ### **Use Cases**
	- **Intrusion Detection & Prevention** – Identifying and blocking unauthorized access attempts.
	- **Malware/Phishing/Spam Detection** – Using ML models to detect malicious emails, domains, or software.
	- **Fraud Detection & Threat Intelligence** – Analyzing transaction patterns to prevent fraud.
	- **User & Entity Behavior Analytics (UEBA)** – Detecting anomalies in user behavior to flag insider threats.
	- **Automated Incident Response** – Using AI to respond to security incidents without human intervention.
## **ML Algorithms**
- ### **Supervised Learning**
	- Algorithms trained with labeled data:
		- **Linear Regression** – Predicts numerical values (e.g., forecasting attack likelihood).
		- **Logistic Regression** – Classifies threats as safe or malicious.
		- **Support Vector Machine (SVM)** – Separates data into distinct categories (e.g., spam vs. non-spam).
		- **Decision Trees** – Creates rules for classifying security incidents.
		- **Random Forest** – Improves accuracy by using multiple decision trees.
- ### **Unsupervised Learning**
	- Algorithms trained with unlabeled data:
		- **K-Means Clustering** – Groups similar data points (e.g., detecting botnets in network traffic).
- ### **Deep Learning**
	- **Neural Networks** – Used for complex cybersecurity tasks, such as malware detection and behavioral analysis.

## **Tools & Libraries**
Python is widely used for ML in cybersecurity. Important libraries include:
- **Numpy** – For numerical operations.
- **Pandas** – Data manipulation and preprocessing.
- **Matplotlib & Seaborn** – Data visualization.
- **Scikit-learn (Sklearn)** – Standard ML algorithms.
- **TensorFlow/PyTorch** – Deep learning frameworks.
- **Keras** – Simplified deep learning model building.

## **Companies Using ML in Cybersecurity**
- ### **Tech Giants Integrating ML for Security**
	- **Microsoft** – AI-driven threat intelligence in Defender & Sentinel.
	- **Google** – Uses ML for spam filtering and malware detection in Gmail.
	- **Amazon (AWS)** – AI-powered security monitoring.
	- **Apple** – Face ID and anomaly detection in security logs.
- ### **Cybersecurity Companies Specializing in ML**
	- **FireEye (Mandiant)** – Uses AI to detect APTs (Advanced Persistent Threats).
	- **Palo Alto Networks** – AI-driven intrusion prevention systems (IPS).
	- **Vectra AI** – Uses ML for real-time attack detection.
	- **Sophos** – AI-powered endpoint security solutions.

---

# Lecture 2: Machine Learning
- ### **Rising Cybersecurity Problems**
	- **Intrusion Detection and Prevention** – Identifying unauthorized access attempts in real-time.
	- **Vulnerability Management** – Discovering and patching weaknesses in software and systems.
	- **Malware Detection and Classification** – Using ML to recognize new and evolving malware threats.
	- **Phishing Detection** – Identifying deceptive emails and websites designed to steal user data.
	- **Spam and Botnet Detection** – Filtering out malicious automated activity in networks.
	- **Fraud Detection** – Analyzing transaction patterns to prevent financial fraud.
	- **Threat Intelligence** – Predicting cyberattacks based on collected threat data.
	- **User and Entity Behavior Analytics (UEBA)** – Detecting unusual behavior in user activity logs.
	- **Automated Incident Response** – Using AI to automatically respond to cyber threats.
	- **Data Loss Prevention (DLP)** – Preventing unauthorized access to sensitive data.
	- **Detection of Advanced Persistent Threats (APT)** – Identifying prolonged, targeted cyberattacks.
	- **Detection of Hidden Channels** – Finding covert communication methods used by attackers.
	- **Detection of Software Vulnerabilities** – Predicting and mitigating software flaws before exploitation.
## **Machine Learning**
Machine learning (ML) is a subset of **Artificial Intelligence (AI)** that enables computers to learn from data and make decisions without explicit programming. Unlike traditional rule-based systems, ML algorithms **improve over time** as they are exposed to more data.
- ### **Why Use Machine Learning in Cybersecurity?**
	1. **Scalability** – ML can process vast amounts of security data much faster than humans.
	2. **Feature Extraction** – Identifies key data attributes useful for security models.
	3. **Limited Human Expertise** – ML helps in areas where human expertise is scarce.
	4. **Adaptability** – Cyber threats evolve, and ML systems can adapt over time.
	5. **Complex Problem-Solving** – ML handles intricate scenarios that involve large datasets and dynamic conditions.
	6. **Pattern Recognition** – Detects anomalies that may indicate cyber threats.
## **Fundamental Machine Learning Concepts**
- ### **Defining the Task (T)**
	- A **task (T)** in ML is the specific problem the system is trying to solve using data. Examples:
		- **Classifying emails as spam or not spam** (Classification)
		- **Detecting fraudulent credit card transactions** (Anomaly Detection)
		- **Predicting network intrusions** (Regression)
- ### **Defining the Experience (E)**
	- The **experience (E)** represents the dataset used for training the ML model. A model **learns patterns** from this dataset to make predictions.
		- **Labeled Data** – Used in supervised learning (e.g., network traffic logs labeled as normal or attack).
		- **Unlabeled Data** – Used in unsupervised learning (e.g., clustering user behaviors without predefined labels).
- ### **Defining the Performance (P)**
	- The **performance (P)** measures how well an ML model is solving its task. Performance is evaluated using standard metrics such as:
		- **Accuracy** – Percentage of correct predictions.
		- **Precision & Recall** – Used in cybersecurity to measure the rate of false positives vs. actual threats detected.
		- **F1-Score** – A balance between precision and recall.

## **Machine Learning Pipeline**
A **machine learning pipeline** is a sequence of steps for building an ML model, from data collection to model deployment.
- ### **Key Stages in the Pipeline**
	1. **Data Collection** – Gathering relevant data from logs, network traffic, or security events.
	2. **Data Preprocessing** – Cleaning, normalizing, and transforming raw data into a usable format.
	3. **Feature Engineering** – Extracting key attributes that will help in model training.
	4. **Model Selection** – Choosing an appropriate ML algorithm.
	5. **Training & Testing** – Splitting the dataset into training and testing sets to evaluate model performance.
	6. **Model Optimization** – Adjusting parameters to improve accuracy.
	7. **Deployment** – Integrating the model into a live security system.
	8. **Continuous Monitoring** – Updating the model as new threats emerge.
## **Types of Machine Learning**
- ### **Supervised Learning** (Requires Labeled Data)
	- The algorithm learns from labeled examples and makes predictions based on past data.
		- **Classification** – Assigning labels to input data (e.g., spam vs. non-spam emails).
		- **Regression** – Predicting a continuous outcome (e.g., estimating attack severity).
- ### **Unsupervised Learning** (No Labeled Data)
	- The model finds patterns in data without predefined labels.
		- **Clustering** – Grouping similar data points (e.g., identifying malicious user behaviors).
		- **Dimensionality Reduction** – Simplifying data without losing critical information.
		- **Anomaly Detection** – Identifying outliers in network traffic or login activity.
		- **Association Rule-Mining** – Finding hidden relationships between security events.
- ### **Semi-Supervised Learning** (Mix of Labeled & Unlabeled Data)
	- Useful when labeling data is expensive, but a small amount of labeled data is available.
- ### **Reinforcement Learning** (Learning from Actions & Rewards)
	- The model improves by interacting with the environment and receiving rewards or penalties.
	- Example: **AI-driven intrusion prevention systems (IPS)** that learn from cyberattack patterns.
## **Learning Approaches in ML**
- ### **Batch Learning**
	- The model is trained using the entire dataset at once.
	- Once trained, it does not update itself automatically.
	- Used for **offline analysis** (e.g., periodic malware analysis).
- ### **Online Learning**
	- The model continuously learns from incoming data.
	- Ideal for **real-time cybersecurity monitoring** (e.g., live network threat detection).
- ### **Instance-Based Learning**
	- The algorithm memorizes specific instances and makes decisions based on similarity.
	- Example: **K-Nearest Neighbors (KNN)** for intrusion detection.
- ### **Model-Based Learning**
	- The algorithm builds a general model and makes predictions based on training data.
	- Example: **Support Vector Machines (SVM)** for classifying security threats.

## **Supervised vs. Unsupervised ML Pipelines**
- ### **Supervised ML Pipeline**
	1. **Collect labeled cybersecurity data** (e.g., logs with attack labels).
	2. **Preprocess and clean the data** (handling missing values, feature selection).
	3. **Split data into training and testing sets.**
	4. **Train a classification model** (e.g., Decision Trees, SVM).
	5. **Evaluate accuracy using metrics like Precision, Recall, F1-Score.**
	6. **Deploy model for real-time threat detection.**
- ### **Unsupervised ML Pipeline**
	1. **Collect raw, unlabeled data** (e.g., network traffic without predefined attack labels).
	2. **Apply clustering or anomaly detection algorithms** (e.g., K-Means, Isolation Forest).
	3. **Identify patterns of unusual activity.**
	4. **Use detected anomalies for proactive threat hunting.**

## **Example: Cybersecurity Application of ML**
- A **machine learning-based intrusion detection system (IDS)** can analyze network packets and classify them as normal or attack traffic.
- **Phishing detection systems** use ML to analyze email content and flag suspicious messages.
- **AI-driven firewalls** adapt to new threats by continuously learning from attack patterns.

---

# Lecture 3 + 4: Data Views
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
- Attributes are **variables or features** used for **data analysis and machine learning.** They can be classified as:
	- **Categorical Attributes** – Discrete values (e.g., Gender, Color).
	- **Numeric Attributes** – Continuous values (e.g., Age, Temperature).
- ### **Nominal Attributes (Categorical & Unordered)**
	- Nominal attributes are categorical attributes **without inherent order.**
		- **Examples:**
		    - **Colors:** {Red, Green, Blue}
		    - **Gender:** {Male, Female, Non-binary}
		    - **Countries:** {USA, Canada, Mexico}
	- #### **Analytics for Nominal Attributes:**
		- **Frequency Distribution** – Shows how often each category appears.
		- **Mode** – Identifies the most common category (useful for handling missing values).
		- **Cross-Tabulation (Contingency Tables)** – Analyzes relationships between categories.
		- **Chi-Square Test** – Determines associations between categorical variables.
	- #### **Encoding for Machine Learning:**
		- **Label Encoding** – Assigns numerical values to each category.
		- **One-Hot Encoding** – Converts categories into binary (0 or 1) columns.
		- **Binary Encoding** – Combines label encoding and one-hot encoding.
		- **Target Encoding** – Replaces categories with the mean target variable.
	- #### **Feature Engineering Techniques:**
		- **Combining Categories** – Merging similar groups (e.g., “Bachelor’s” & “Master’s” into “Higher Education”).
		- **Creating Interaction Features** – Deriving new features from existing nominal attributes.
	- #### **Visualization Methods:**
		- **Bar Charts, Pie Charts** – Used to display category distribution.
	- #### **Insights from Nominal Attributes:**
		- **Category-Specific Statistics** – Understanding major classes.
		- **Predictive Power** – Evaluating importance for ML models.
		- **Anomaly Detection** – Identifying outliers in categorical data.
- ### **Ordinal Attributes (Categorical & Ordered)**
	- Ordinal attributes have a **defined order** but **unequal differences** between values.
	- **Examples:**
	    - **Education Levels:** {High School, Bachelor’s, Master’s, PhD}
	    - **Customer Satisfaction:** {Very Unsatisfied, Neutral, Satisfied, Very Satisfied}
	    - **Movie Ratings:** {1 star, 2 stars, 3 stars, 4 stars, 5 stars}
	- #### **Analytics for Ordinal Attributes:**
		- **Frequency Distribution** – Identifies dominant categories.
		- **Mode, Median, Percentiles** – Measures central tendency.
		- **Spearman Rank Correlation** – Analyzes relationships between ordinal & numeric attributes.
	- #### **Encoding for Machine Learning:**
		- **Ordinal Encoding** – Assigns numeric values based on order.
		- **Target Encoding** – Uses mean of target variable for each category.
	- #### **Feature Engineering Techniques:**
		- **Binning/Grouping** – Merging similar categories.
		- **Interaction Features** – Creating new variables based on ordinal data.
	- #### **Visualization Methods:**
		- **Bar Charts, Histograms** – Display distribution of ordinal attributes.
	- #### **Insights from Ordinal Attributes:**
		- **Trend Analysis** – Evaluates patterns over time.
		- **Correlation with Target Variable** – Helps in feature selection.
- ### **Interval-Scaled Attributes (Continuous & No True Zero)**
	- Interval-scaled attributes have **equal differences** between values, but **no absolute zero.**
		- **Examples:**
		    - **Temperature in Celsius/Fahrenheit**
		    - **Dates & Time Intervals**
	- #### **Analytics for Interval-Scaled Attributes:**
		- **Mean, Median, Standard Deviation** – Measure central tendency & spread.
		- **Pearson Correlation Coefficient** – Assesses relationships between interval attributes.
		- **Covariance** – Indicates directional relationships.
	- #### **Feature Scaling for ML:**
		- **Z-score Normalization** – Converts values to a mean of 0 and standard deviation of 1.
		- **Min-Max Scaling** – Rescales values between [0,1].
		- **IQR Method** – Identifies outliers using interquartile range.
	- #### **Feature Engineering Techniques:**
		- **Polynomial Features** – Deriving new features using powers of existing attributes.
		- **Interaction Terms** – Capturing dependencies between attributes.
	- #### **Dimensionality Reduction Techniques:**
		- **PCA (Principal Component Analysis)** – Reduces feature dimensions while preserving variance.
	- #### **Visualization Methods:**
		- **Histograms, Box Plots, Scatter Plots, Line Plots** – Used for data distribution analysis.
	- #### **Insights from Interval-Scaled Attributes:**
		- **Trends & Patterns** – Identify shifts in cybersecurity data.
		- **Outliers & Anomalies** – Detect security breaches.
- ### **Ratio-Scaled Attributes (Continuous & True Zero Exists)**
	- Ratio-scaled attributes have a **true zero point**, meaning zero represents **no existence** of the measured quantity.
	- **Examples:**
	    - **Height, Weight, Age, Income**
	- #### **Analytics for Ratio-Scaled Attributes:**
		- **Mean, Median, Standard Deviation** – Measure central tendency & dispersion.
		- **Correlation & Covariance** – Identifies dependencies between attributes.
	- #### **Feature Scaling for ML:**
		- **Z-score Normalization, Min-Max Scaling, IQR Method** – Applied similarly to interval-scaled attributes.
	- #### **Feature Engineering Techniques:**
		- **Polynomial & Interaction Features** – Used for ML model improvements.
	- #### **Predictive Modeling:**
		- **Regression Analysis (Linear & Logistic)** – Uses ratio attributes for prediction.
	- #### **Anomaly Detection:**
		- **Identifying Outliers** – Unusual data points (e.g., fraudulent transactions).
	- #### **Visualization Methods:**
		- **Histograms, Box Plots, Scatter Plots, Line Plots** – Used for detecting anomalies.
	- #### **Insights from Ratio-Scaled Attributes:**
		- **Relative Measures & Ratios** – Useful for comparative analysis.
		- **Descriptive & Predictive Insights** – Provides valuable trends in cybersecurity.

## **Algebraic & Geometric View of Data**
### **Key Concepts:**
- **Vector & Matrix Representation** – Data is represented as vectors & matrices, making it easier to apply mathematical transformations and optimizations.
- **Linear Algebra Operations:**
    - **Matrix multiplication** – Used in neural networks, dimensionality reduction, and data transformations.
    - **Matrix inversion** – Helps in solving systems of equations, particularly useful in regression models.
    - **Feature transformations** – Scaling, rotation, and projection of data for better interpretability and model efficiency.
- **Dimensionality Reduction:**
    - **Principal Component Analysis (PCA)** – Projects high-dimensional data into lower dimensions while retaining maximum variance.
    - **Singular Value Decomposition (SVD)** – Factorizes a matrix into three matrices to extract important features.
    - **Latent Semantic Analysis (LSA)** – Used in NLP to analyze relationships between words and documents.
- **Feature Scaling & Normalization:**
    - **Min-Max Scaling** – Rescales values between a fixed range [0,1] or [-1,1] to improve model performance.
    - **Z-score Normalization** – Converts data into a normal distribution with mean 0 and standard deviation 1.
    - **Logarithmic Scaling** – Used for highly skewed data to reduce the impact of extreme values.
- **Optimization & Model Training:**
    - **Gradient Descent** – Optimizes model parameters by iteratively reducing error.
    - **Stochastic Gradient Descent (SGD)** – A variant of gradient descent that updates parameters for each training instance, useful for large datasets.
    - **Newton’s Method** – Used in convex optimization problems for rapid convergence.
    - **Least Squares Optimization** – Minimizes the sum of squared residuals in regression models.
- **Distance Measures:**
    - **Euclidean Distance** – Measures straight-line distance between two points.
    - **Manhattan Distance** – Measures distance along axis-aligned paths, useful for grid-like structures.
    - **Mahalanobis Distance** – Accounts for correlations between variables and scales accordingly.
    - **Cosine Similarity** – Measures angular similarity between vectors, commonly used in text mining and NLP.
- **Orthogonal Projection & Linear Independence:**
    - **Orthogonality** – Ensures data features are independent of each other, aiding interpretability and reducing redundancy.
    - **Linear Independence** – Ensures that no feature can be represented as a linear combination of other features.
- **Kernel Methods & Feature Transformations:**
    - **Kernel Trick** – Projects data into higher-dimensional space for non-linear classification.
    - **Polynomial & Radial Basis Function (RBF) Kernels** – Transformations that allow support vector machines to work effectively with non-linear data.
## **Probabilistic View of Data**
### **Key Concepts:**
- **Data as a Random Variable:**
    - Attributes modeled as random variables enable statistical inferences about populations from sampled data.
    - Used in Bayesian networks and probabilistic graphical models.
- **Distribution Modeling:**
    - **Normal Distribution (Gaussian)** – Used for statistical modeling and ML algorithms like Naive Bayes.
    - **Poisson Distribution** – Used for event-based modeling, e.g., cybersecurity attack frequency prediction.
    - **Exponential Distribution** – Useful in failure rate analysis.
    - **Dirichlet Distribution** – Used in topic modeling for natural language processing.
- **Feature Selection & Regularization:**
    - **Mutual Information** – Measures dependency between variables.
    - **Information Gain** – Used in decision trees and entropy-based models.
    - **L1 & L2 Regularization** – Lasso and Ridge regression techniques that prevent overfitting.
- **Overfitting Prevention:**
    - **Bayesian Regularization** – Introduces prior distributions to limit model complexity.
    - **Cross-Validation** – Ensures generalization by testing model performance on unseen data.
    - **Dropout in Neural Networks** – Prevents co-adaptation of neurons, reducing overfitting.
- **Decision Theory & Risk Management:**
    - **Probability-based decision-making** – Bayesian inference applied to cybersecurity threat detection.
    - **Markov Decision Processes (MDP)** – Used in reinforcement learning for decision-making in uncertain environments.
    - **Handling Uncertainty** – Using probabilistic confidence intervals and Monte Carlo simulations.
- **Handling Missing Data:**
    - **Mean/Median Imputation** – Replacing missing values using statistical averages.
    - **Expectation-Maximization (EM) Algorithm** – Probabilistic technique to handle missing data and clustering.
- **Bayesian Inference & Uncertainty Quantification:**
    - **Bayesian Theorem** – Updates the probability of a hypothesis as more evidence is provided.
    - **Monte Carlo Methods** – Used in probabilistic sampling to model uncertainty in predictions.

---

## **Graph View of Data**
### **Key Concepts:**
- **Graph Representation:**
    - **Nodes (Vertices):** Represent entities such as users, IP addresses, or devices.
    - **Edges (Links):** Represent relationships between nodes, e.g., network connections.
    - **Adjacency Matrix:** Matrix representation of a graph, capturing connectivity between nodes.
- **Graph Metrics:**
    - **Degree Distribution** – Measures the number of connections per node.
    - **Shortest Path & Betweenness Centrality** – Determines influential nodes in a network.
    - **Clustering Coefficient** – Measures the tendency of nodes to form tightly-knit groups.
    - **PageRank** – Used in search engine ranking and network analysis.
- **Machine Learning Applications:**
    - **Graph Neural Networks (GNNs)** – Learn embeddings for nodes in a graph.
    - **Social Network Analysis** – Understanding connections and detecting fake accounts.
    - **Community Detection** – Identifies clusters of related nodes (e.g., fraud rings).
    - **Anomaly Detection in Networks** – Identifies unusual behaviors using graph structures.
    - **Link Prediction** – Predicts future connections (e.g., friend suggestions on social media).
    - **Recommendation Systems** – Suggests items based on user interaction graphs.
---

## **Benchmark Datasets for Cybersecurity**

Benchmark datasets are commonly used for training and evaluating machine learning models in cybersecurity.

### **Intrusion Detection and Prevention**

- **KDD Cup 1999** – [https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- **NSL-KDD** – [https://www.kaggle.com/datasets/hassan06/nslkdd](https://www.kaggle.com/datasets/hassan06/nslkdd)
- **CICIDS 2017** – [https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

### **Malware Detection and Classification**

- **MalwareBazaar** – [https://bazaar.abuse.ch/](https://bazaar.abuse.ch/)
- **Ember Dataset** – [https://github.com/elastic/ember](https://github.com/elastic/ember)
- **Malware Traffic Analysis** – [https://www.malware-traffic-analysis.net](https://www.malware-traffic-analysis.net/)

### **Phishing Detection**

- **PhishTank Dataset** – [https://www.phishtank.com/](https://www.phishtank.com/)
- **Labeled Phishing URLs Dataset** – [https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- **APWG eCrime Exchange (eCX)** – [https://apwg.org/the-apwg-ecrime-exchange-ecx/](https://apwg.org/the-apwg-ecrime-exchange-ecx/)

### **Fraud Detection**

- **Credit Card Fraud Detection Dataset** – [https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- **IEEE-CIS Fraud Detection** – [https://www.kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)

### **User and Entity Behavior Analytics (UEBA)**

- **CERT Insider Threat Dataset** – [https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research](https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research)
- **LANL User Authentication Dataset** – [https://csr.lanl.gov/data/](https://csr.lanl.gov/data/)

### **Spam and Botnet Detection**

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