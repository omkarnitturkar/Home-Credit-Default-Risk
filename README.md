# Employee Performance Analysis and Prediction

## Project Overview
This project involves analyzing employee performance data from the **INX Future Inc Employee Performance dataset**. The goal is to identify key factors influencing performance, predict performance ratings, and provide actionable recommendations to improve overall employee productivity.

---

## Features
- **Data Cleaning and Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Visualizations**: Histogram, Violin Plots, Correlation Heatmap, etc.
- **Outlier Detection and Removal**
- **Predictive Modeling**: Machine Learning Algorithms
- **Hyperparameter Tuning**: GridSearchCV
- **Feature Importance Analysis**
- **Custom Function**: Predicting performance of new employees

---

## Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Visualization Tools**: SweetViz, Violin Plots, Correlation Heatmaps
- **Machine Learning Models**:
  - Random Forest
  - Decision Tree
  - Support Vector Classifier
  - K-Nearest Neighbors
  - Naive Bayes
  - Bagging Classifier
- **Data Processing**: Label Encoding, StandardScaler
- **Evaluation Metrics**: Accuracy, Classification Report

---

## Dataset
The dataset includes employee details such as demographics, job roles, work environment satisfaction, and performance ratings. Data preprocessing includes handling null values, encoding categorical variables, and removing outliers.

---

## Key Steps

### 1. Data Preprocessing
- Removed missing values and handled categorical data using Label Encoding.
- Detected and removed outliers from key features such as:
  - `TotalWorkExperienceInYears`
  - `ExperienceYearsAtThisCompany`
  - `YearsSinceLastPromotion`

---

### 2. Exploratory Data Analysis (EDA)
- Analyzed data distribution using histograms and box plots.
- Created department-wise and job role-based performance visualizations.
- Explored correlations using a heatmap.

#### Correlation Heatmap
![Correlation Heatmap](https://github.com/omkarnitturkar/Home-Credit-Default-Risk/blob/main/heatmap.png)

---

### 3. Model Training and Evaluation
- Trained multiple machine learning models and compared performance.
- Used GridSearchCV for hyperparameter optimization.
- Identified the top 3 most important features affecting performance.

#### Distribution
![Department-wise Performance](https://github.com/omkarnitturkar/Home-Credit-Default-Risk/blob/main/Distribution.png)

---

### 4. Predictive Function
A custom function, `predict_employee_performance`, predicts the performance rating for new hires based on input features.

---

## Results
- Best-performing model: **Random Forest Classifier** with optimized hyperparameters.
- **Accuracy**: Over **85%**.
- **Top 3 Important Factors**:
  1. `YearsSinceLastPromotion`
  2. `EmpEnvironmentSatisfaction`
  3. `EmpLastSalaryHikePercent`

---

## Recommendations
- Improve work-life balance for employees to boost performance.
- Increase training opportunities for underperforming employees.
- Tailor performance incentives based on specific job roles.
---

## How to Run the Project
1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn sweetviz
   ```

2. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

3. Navigate to the project directory and run the Jupyter notebook for analysis:
   ```bash
   jupyter notebook Employee_Performance_Analysis.ipynb
   ```

---

# Additional Application: Home Credit Default Risk Analysis

## Project Overview
This part of the project analyzes and predicts the risk of loan default using the **Home Credit Default Risk Dataset**.

---

## Steps

### 1. Dataset Loading
```python
application_train = pd.read_csv('application_train.csv')
bureau = pd.read_csv('bureau.csv')
bureau_balance = pd.read_csv('bureau_balance.csv')
POS_CASH_balance = pd.read_csv('POS_CASH_balance.csv')
credit_card_balance = pd.read_csv('credit_card_balance.csv')
previous_application = pd.read_csv('previous_application.csv')
installments_payments = pd.read_csv('installments_payments.csv')
```

---

### 2. Exploratory Data Analysis (EDA)

#### Missing Values Table Function
```python
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
    return mis_val_table_ren_columns.sort_values('% of Total Values', ascending=False)
```

#### Target Distribution
```python
application_train['TARGET'].astype(int).plot.hist();
```

---

### 3. Feature Engineering
- Encoded categorical variables using **Label Encoding** and **One-Hot Encoding**.
- Aggregated numerical and categorical features from related tables.
- Generated polynomial features for key predictors.

---

### 4. Model Training
- Trained multiple models including Logistic Regression, Random Forest, and Gradient Boosting.
- Used **GridSearchCV** for hyperparameter tuning.

#### Evaluation Metrics
- **Accuracy**
- **ROC AUC Score**

---

## Visualizations
#### Age Distribution and Default Rates
```python
plt.figure(figsize=(10, 8))
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label='Target == 0')
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label='Target == 1')
plt.title('Age Distribution by Target');
```

#### Correlation Heatmap
```python
plt.figure(figsize=(8, 6))
sns.heatmap(application_train.corr(), cmap='coolwarm', annot=True);
```

---

## Final Dataset
The final processed dataset is exported to `train_final.csv` for model training and evaluation.

```python
train_final.to_csv('train_final.csv')
```

---

## Results
- Best-performing model: **Logistic Regression** with an AUC score of **0.77**.
- Identified key predictors of loan default including **DAYS_BIRTH**, **EXT_SOURCE_2**, and **AMT_CREDIT**.
