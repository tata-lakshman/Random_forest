# 🌲 Customer Churn Prediction Using Random Forest

## 📌 Project Overview

This project uses the **Random Forest Classifier** to predict customer churn.

Customer churn refers to whether a customer leaves or stops using a company's service. The objective of this project is to build a machine learning model that predicts customer churn based on customer demographic information, services, contract details, and billing information.

The project includes:

* Data loading and exploration
* Data cleaning
* Handling missing values
* Converting categorical data into numerical values
* Exploratory Data Analysis
* Churn distribution analysis
* Correlation analysis
* Train-test splitting
* Random Forest model training
* Model evaluation
* Hyperparameter tuning using RandomizedSearchCV
* Feature importance analysis
* Classification report analysis

---

## 📊 Dataset

The dataset used in this project is:

```text
customer_churn (3).csv
```

The dataset contains customer information such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges
* Churn

The target variable is:

```text
Churn
```

The model predicts whether a customer is likely to churn.

---

## 🛠️ Technologies and Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

### Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 📥 Loading the Dataset

The dataset is loaded using Pandas.

```python
churn_data = pd.read_csv('customer_churn (3).csv')
```

The dataset was explored using:

```python
churn_data.head()
churn_data.columns
churn_data.shape
churn_data.dtypes
churn_data.info()
```

---

# 🔍 Exploratory Data Analysis

Basic Exploratory Data Analysis was performed to understand the dataset.

The following operations were used:

* Viewing the first rows of the dataset
* Checking column names
* Checking the dataset shape
* Checking data types
* Viewing dataset information
* Generating statistical summaries

For numerical columns:

```python
churn_data.describe().T
```

For categorical columns:

```python
churn_data.describe(include='object').T
```

---

# 🧹 Data Cleaning

## Converting Total Charges to Numeric

The `TotalCharges` column was converted into numeric format.

```python
churn_data['TotalCharges'] = pd.to_numeric(
    churn_data['TotalCharges'],
    errors='coerce'
)
```

Invalid values were converted into missing values.

---

## Handling Missing Values

Missing values were checked using:

```python
churn_data.isnull().sum()
```

Rows containing missing values were removed.

```python
churn_data.dropna(inplace=True)
```

---

## Removing Customer ID

The `customerID` column was removed because it is a unique identifier and is not used as a feature for model training.

```python
churn_data.drop(
    columns=['customerID'],
    inplace=True
)
```

---

# 🔄 Data Preprocessing

The dataset contains categorical values. These values were converted into numerical values using `OrdinalEncoder`.

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
```

Several categorical columns were encoded, including:

* Gender
* Churn
* Contract
* Dependents
* Device Protection
* Internet Service
* Multiple Lines
* Online Backup
* Online Security
* Partner
* Phone Service
* Paperless Billing
* Payment Method
* Streaming Movies
* Streaming TV
* Tech Support

Example:

```python
churn_data['gender'] = encoder.fit_transform(
    churn_data[['gender']]
)

churn_data['Churn'] = encoder.fit_transform(
    churn_data[['Churn']]
)

churn_data['Contract'] = encoder.fit_transform(
    churn_data[['Contract']]
)
```

The notebook also applies `OrdinalEncoder` to additional columns before training the model.

---

# 📊 Churn Distribution Analysis

The percentage distribution of customer churn was calculated using:

```python
churn_data['Churn'].value_counts(
    normalize=True
) * 100
```

A bar chart was created to visualize the churn distribution.

```python
churn_data['Churn'].value_counts(
    normalize=True
).plot(
    kind='bar',
    figsize=(12,8),
    color='pink'
)
```

---

# 🔥 Correlation Analysis

A correlation heatmap was created to analyze relationships between features.

```python
plt.figure(figsize=(10,8))

sns.heatmap(
    churn_data.corr(),
    cmap='Reds',
    annot=True,
    fmt='.1f'
)

plt.show()
```

This visualization helps understand the correlation between the numerical features in the dataset.

---

# 🎯 Feature and Target Selection

The dataset was divided into:

* **X** → Input features
* **y** → Target variable

```python
x = churn_data.iloc[:, :-1]

y = churn_data.iloc[:, -1]
```

The target variable is the customer churn value.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing data.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)
```

* **80%** of the data is used for training.
* **20%** of the data is used for testing.

The `random_state=0` ensures reproducible results.

---

# 🌲 Random Forest Model

A Random Forest Classifier was created and trained.

```python
rf = RandomForestClassifier(
    random_state=0
)

rf1 = rf.fit(
    x_train,
    y_train
)
```

Predictions were generated for both training and testing datasets.

```python
y_train_pred = rf1.predict(x_train)

y_test_pred = rf1.predict(x_test)
```

---

# 📈 Model Evaluation

The model was evaluated using accuracy.

```python
print(
    f'Train Accuracy: '
    f'{accuracy_score(y_train, y_train_pred) * 100}'
)

print(
    f'Test Accuracy: '
    f'{accuracy_score(y_test, y_test_pred) * 100}'
)
```

The project compares:

* Training Accuracy
* Testing Accuracy

This helps evaluate the performance of the model on both training and unseen testing data.

---

# ⚙️ Hyperparameter Tuning

The project uses **RandomizedSearchCV** to search for better Random Forest hyperparameters.

The following parameters were included:

* `n_estimators`
* `max_features`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `bootstrap`

The parameter grid was created as follows:

```python
n_estimators = [
    int(x)
    for x in np.linspace(
        start=200,
        stop=2000,
        num=10
    )
]

max_features = ['auto', 'sqrt']

max_depth = [
    int(x)
    for x in np.linspace(
        10,
        110,
        num=11
    )
]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]
```

A random parameter grid was created:

```python
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
```

---

# 🔎 RandomizedSearchCV

`RandomizedSearchCV` was used to test different combinations of hyperparameters.

```python
rf = RandomForestClassifier(
    random_state=0
)

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2,
    random_state=0,
    n_jobs=-1,
    return_train_score=True
)
```

The model was then trained using:

```python
rf2 = rf_random.fit(
    x_train,
    y_train
)
```

The best hyperparameters were checked using:

```python
rf2.best_params_
```

---

# 🚀 Tuned Random Forest Model

A Random Forest model was created using the selected hyperparameters.

```python
rf2 = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=10,
    bootstrap=False,
    random_state=42
)

rf2.fit(
    x_train,
    y_train
)
```

Predictions were generated again:

```python
y_train_pred = rf2.predict(
    x_train
)

y_test_pred = rf2.predict(
    x_test
)
```

The training and testing accuracy were calculated again after tuning.

---

# 📊 Feature Importance Analysis

Random Forest provides a feature importance score that shows which features contribute more to the model's predictions.

Feature importance was extracted using:

```python
importances = pd.DataFrame(
    data={
        'Attribute': x_train.columns,
        'Importance': rf2.feature_importances_
    }
)
```

The features were sorted by importance:

```python
importances = importances.sort_values(
    by='Importance',
    ascending=False
)
```

A bar chart was created to visualize feature importance.

```python
plt.figure(figsize=(25,8))

sns.barplot(
    x=importances['Attribute'],
    y=importances['Importance'],
    palette='Set1'
)

plt.show()
```

---

# 📋 Classification Report

The model was further evaluated using a classification report.

```python
print(
    classification_report(
        y_test,
        y_test_pred
    )
)
```

The classification report provides metrics such as:

* Precision
* Recall
* F1-score
* Support

The classification report was also generated for the training data.

```python
print(
    classification_report(
        y_train,
        y_train_pred
    )
)
```

---

# 📁 Project Structure

```text
Customer-Churn-Random-Forest/
│
├── Random_Forest (1).ipynb
├── customer_churn (3).csv
└── README.md
```

---

# ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd Customer-Churn-Random-Forest
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebook

Open:

```text
Random_Forest (1).ipynb
```

Make sure the dataset file:

```text
customer_churn (3).csv
```

is available in the same project directory.

---

# 🧠 Key Learnings

Through this project, I learned:

* How to load and explore a dataset
* How to handle missing values
* How to convert data types
* How to remove unnecessary columns
* How to encode categorical variables
* How to analyze customer churn
* How to create a correlation heatmap
* How to split data into training and testing sets
* How Random Forest Classification works
* How to train a Random Forest model
* How to make predictions
* How to evaluate a model using accuracy
* How to perform hyperparameter tuning
* How `RandomizedSearchCV` works
* How cross-validation is used during hyperparameter tuning
* How to identify the best model parameters
* How to analyze feature importance
* How to evaluate a classification model using precision, recall, and F1-score

---

# 🚀 Future Improvements

Possible improvements for this project include:

* Comparing Random Forest with other classification algorithms
* Adding a Confusion Matrix visualization
* Using GridSearchCV for hyperparameter tuning
* Performing more advanced feature engineering
* Handling class imbalance
* Deploying the trained model using Streamlit
* Creating an interactive customer churn prediction application

---

# 👨‍💻 Author

**Tata Lakshman Kumar**

Aspiring AI/ML Engineer
