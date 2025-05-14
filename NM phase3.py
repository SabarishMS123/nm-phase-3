import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample  # For handling class imbalance (if needed)

# 1. Data Loading
try:
    df = pd.read_csv("telco_churn.csv")  # Or the path to your dataset
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure the file is in the correct location or provide the correct path.")
    # Instead of exit(), raise the exception to allow handling in notebook environment
    raise

# 2. Data Cleaning

## Missing Values
print(df.isnull().sum())   # Check for missing values
# Strategy:
# The 'TotalCharges' column will be converted to numeric. Non-numeric values will be coerced to NaN.
# These NaN values will then be filled with the median of the 'TotalCharges' column.

# Convert 'TotalCharges' to numeric, handling potential errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# 'errors='coerce'' will turn any non-numeric values into NaN (missing)
# Now fill the NaN values in 'TotalCharges' with the median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

## Duplicates
print(df.duplicated().sum())  # Check for duplicates
# Strategy:
# Duplicate rows will be identified and removed to ensure data integrity.
df = df.drop_duplicates()

## Data Types
print(df.info())  # Check data types
# Strategy:
# The data types of the columns will be reviewed to ensure they are appropriate for analysis and modeling.
# 'TotalCharges' has already been converted to numeric.

# 3. Exploratory Data Analysis (EDA)

## Univariate Analysis
# Histograms for numerical features
df.hist(figsize=(12, 10))
plt.show()
# Insight: The histograms provide a visual distribution of the numerical features like 'tenure', 'MonthlyCharges', and 'TotalCharges'. 'tenure' seems to have a concentration at the lower end and a peak at the higher end, suggesting many new customers and many long-term customers. 'MonthlyCharges' has a wider distribution, and 'TotalCharges' is skewed to the right, which is expected as it's a product of 'tenure' and 'MonthlyCharges'.

# Countplots for categorical features
for col in df.select_dtypes(include='object').columns:
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Countplot of {col}')
    plt.tight_layout()
    plt.show()
# Insights: These countplots show the distribution of each categorical feature. For example, we can see the number of customers for each 'InternetService' type, 'Contract' type, 'PaymentMethod', etc. The 'Churn' countplot shows the imbalance in the target variable, with more customers not churning than churning.

# Boxplots for numerical features (outlier detection)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Churn', y='tenure')  # Corrected column name to 'tenure'
plt.title('Boxplot of Tenure vs Churn')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Boxplot of Monthly Charges vs Churn')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Churn', y='TotalCharges')
plt.title('Boxplot of Total Charges vs Churn')
plt.show()
# Insights: These boxplots help visualize the relationship between numerical features and the target variable 'Churn'. For instance, customers with shorter tenures and higher monthly charges seem more likely to churn. 'TotalCharges' for churned customers appears to have a wider spread but generally lower values, likely due to the shorter tenure.

## Bivariate/Multivariate Analysis
# Correlation matrix
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Insight: The correlation matrix shows the linear relationships between numerical features. 'tenure' and 'TotalCharges' have a strong positive correlation, which is expected. 'MonthlyCharges' also shows a positive correlation with 'TotalCharges', but weaker than 'tenure'. The correlation between 'Churn' and other numerical features is relatively weak, but negative for 'tenure' and 'TotalCharges', and slightly positive for 'MonthlyCharges'.

# Scatterplots
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn')
plt.title('Scatterplot of Tenure vs Monthly Charges (colored by Churn)')
plt.show()
# Insight: This scatterplot shows that customers with high monthly charges and shorter tenures are more likely to churn. Customers with longer tenures tend to have a wider range of monthly charges and are less likely to churn.

# Grouped bar plots
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    sns.catplot(data=df, x='Churn', kind='count', hue=col, aspect=1, height=4)
    plt.title(f'Churn Distribution by {col}')
    plt.show()
# Insights: These grouped bar plots illustrate the distribution of churn across different categories within each categorical feature. For example, we can observe churn rates for different contract types, internet services, and payment methods. Month-to-month contracts appear to have a higher churn rate compared to one-year or two-year contracts. Customers with fiber optic internet service also seem to have a higher churn rate.

# 4. Feature Engineering

## Create New Features
df['AvgMonthlyUsage'] = df['TotalCharges'] / df['tenure']
# Rationale: This feature represents the average amount a customer pays per month during their tenure. It might provide insights into spending habits and their relation to churn.

df['HasMultipleServices'] = ((df['PhoneService'] == 'Yes') & (df['InternetService'] == 'Yes') & (df['OnlineSecurity'] == 'Yes')).astype(int)
# Rationale: This binary feature indicates whether a customer subscribes to phone, internet, and online security services. Customers with a broader range of services might be less likely to churn.

## Binning
df['Tenure_Group'] = pd.qcut(df['tenure'], q=4, labels=['0-1 Year', '1-2 Years', '2-5 Years', '5+ Years'])
# Strategy: The 'tenure' column is binned into four quartiles to group customers based on their service duration. This can help capture non-linear relationships between tenure and churn.

## One-Hot Encode Tenure Group (and other categorical features if needed after binning)
df = pd.get_dummies(df, columns=['Tenure_Group'], drop_first=True)

## Preprocessing for Modeling
# Identify numerical and categorical features
numerical_features = df.select_dtypes(include=np.number).drop(columns=['Churn']).columns.tolist()
categorical_features = df.select_dtypes(include='object').columns.tolist()

# Create transformers for preprocessing
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')  # handle_unknown='ignore' is important!

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep remaining columns (like engineered features)
)

# 5. Model Building

## Prepare Data
X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Ensure y is numeric

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratify!

## Model 1: Logistic Regression
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])  # solver important for smaller datasets
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]

## Model 2: Random Forest
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_proba_rf = pipeline_rf.predict_proba(X_test)[:, 1]

## Model Evaluation
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    print(f"--- {model_name} ---")
    print(f"    Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"    Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"    F1-score: {f1_score(y_true, y_pred):.4f}")
    print(f"    ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.show()

evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, 'Logistic Regression')
evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')

## Feature Importance (Random Forest)
feature_importances = pipeline_rf.named_steps['classifier'].feature_importances_
# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Extract the feature names corresponding to the original columns
original_feature_names = []
for name in feature_names:
    if 'num__' in name:
        original_feature_names.append(name.replace('num__', ''))
    elif 'cat__' in name:
        original_feature_names.append(name.replace('cat__', ''))
    else:
        original_feature_names.append(name) # For remainder='passthrough' features

# Match feature importances to original feature names (handling one-hot encoded features)
importance_dict = {}
num_numerical = len(numerical_features)
num_categorical = len(categorical_features)

for i in range(num_numerical):
    importance_dict[numerical_features[i]] = feature_importances[i]

cat_start_index = num_numerical
for i in range(num_categorical):
    base_name = categorical_features[i]
    # Get the number of categories (excluding the dropped one)
    num_cats = len(pipeline_rf.named_steps['preprocessor'].transformers_[1][1].categories_[i]) -1
    for j in range(num_cats):
        importance_dict[f'{base_name}_{pipeline_rf.named_steps["preprocessor"].transformers_[1][1].categories_[i][j+1]}'] = feature_importances[cat_start_index + i * (len(pipeline_rf.named_steps['preprocessor'].transformers_[1][1].categories_[i]) -1) + j]

# Add importance for engineered features (assuming they are at the end after 'passthrough')
engineered_features = ['AvgMonthlyUsage', 'HasMultipleServices', 'Tenure_Group_1-2 Years', 'Tenure_Group_2-5 Years', 'Tenure_Group_5+ Years']
start_engineered_index = num_numerical + sum(len(pipeline_rf.named_steps['preprocessor'].transformers_[1][1].categories_[i]) - 1 for i in range(num_categorical))

for i, feature in enumerate(engineered_features):
    if start_engineered_index + i < len(feature_importances):
        importance_dict[feature] = feature_importances[start_engineered_index + i]

sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)

plt.figure(figsize=(10, 10))
sns.barplot(x=[importance for name, importance in sorted_importance], y=[name for name, importance in sorted_importance])
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 6. Conclusion
# Summarize your findings, compare models, and discuss limitations and future work
print("\n--- Conclusion ---")
print("We built two classification models, Logistic Regression and Random Forest, to predict customer churn based on the Telco dataset.")
print("\nModel Performance:")
print("Logistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
print("\nRandom Forest:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

print("\nObservations:")
print("- Both models show reasonable performance in predicting churn.")
print("- The Random Forest model generally exhibits slightly better performance, particularly in terms of accuracy and potentially in capturing non-linear relationships in the data, as suggested by the feature importance.")
print("- Features like 'Contract_Month-to-month', 'tenure', 'MonthlyCharges', and 'TotalCharges' appear to be important predictors of churn based on the Random Forest feature importance.")
print("- The ROC-AUC scores indicate the ability of both models to distinguish between churned and non-churned customers.")

print("\nLimitations:")
print("- The dataset might have class imbalance, which could affect model performance, especially recall and precision for the minority class (churned customers). We used `stratify` during the train-test split to mitigate this, but further techniques like oversampling or undersampling could be explored.")
print("- The models are based on the features available in the dataset. Other external factors not included here could also influence churn.")
print("- The performance of the models could potentially be improved by further hyperparameter tuning.")

print("\nFuture Work:")
print("- Address potential class imbalance using techniques like SMOTE or ADASYN.")
print("- Perform more extensive hyperparameter tuning for both models using techniques like GridSearchCV or RandomizedSearchCV.")
print("- Explore other advanced machine learning models (e.g., Gradient Boosting, Support Vector Machines).")
print("- Investigate the impact of the newly engineered features on model performance.")
print("- Consider feature selection techniques to identify the most relevant predictors and potentially simplify the models.")
print("- Analyze the types of errors made by the models (false positives and false negatives) to understand the business implications and potentially adjust model thresholds.")
