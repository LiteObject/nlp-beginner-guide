# Data Cleaning Guide: Preparing Data for Analysis and Model Training

Data cleaning is a critical step in the data science workflow. Raw data often contains inconsistencies, missing values, duplicates, and formatting issues that can negatively impact analysis and model performance. This guide covers essential data cleaning techniques.

## Table of Contents
1. [Handling Missing Values](#1-handling-missing-values)
2. [Removing Duplicates](#2-removing-duplicates)
3. [Data Type Conversion](#3-data-type-conversion)
4. [Handling Outliers](#4-handling-outliers)
5. [Text Data Cleaning](#5-text-data-cleaning)
6. [Standardizing and Normalizing](#6-standardizing-and-normalizing)
7. [Removing Irrelevant Features](#7-removing-irrelevant-features)
8. [Data Consistency and Validation](#8-data-consistency-and-validation)

## 1. Handling Missing Values

Missing values (NaN, None, empty strings) can distort analysis and break models.

### Detection
```python
# Check for missing values
print(df.isnull().sum())  # Count missing values per column
print(df.isnull())         # Boolean mask of missing values
```

### Strategies

**Drop Missing Values**
- Use when missing data is < 5% of dataset
```python
df_cleaned = df.dropna()  # Remove rows with any missing value
df_cleaned = df.dropna(subset=['column_name'])  # Drop if specific column is missing
df_cleaned = df.dropna(thresh=5)  # Keep rows with at least 5 non-null values
```

**Forward/Backward Fill** (for time-series data)
```python
df['column'].fillna(method='ffill')  # Forward fill
df['column'].fillna(method='bfill')  # Backward fill
```

**Imputation**
```python
# Fill with mean/median/mode
df['column'].fillna(df['column'].mean(), inplace=True)
df['column'].fillna(df['column'].median(), inplace=True)
df['column'].fillna(df['column'].mode()[0], inplace=True)

# Fill with specific value
df['column'].fillna('Unknown', inplace=True)
```

**Interpolation** (numerical data)
```python
df['column'].interpolate(method='linear', inplace=True)
```

## 2. Removing Duplicates

Duplicate rows inflate the dataset and bias analysis.

### Detection and Removal
```python
# Find duplicates
print(df.duplicated())  # Boolean mask
print(df.duplicated(subset=['column1', 'column2']))  # Check specific columns

# Remove duplicates
df_cleaned = df.drop_duplicates()
df_cleaned = df.drop_duplicates(subset=['column1', 'column2'])  # Based on specific columns
df_cleaned = df.drop_duplicates(keep='first')  # Keep first occurrence
```

## 3. Data Type Conversion

Ensure columns have appropriate data types for processing.

```python
# Convert to numeric
df['column'] = pd.to_numeric(df['column'], errors='coerce')  # Convert invalid values to NaN

# Convert to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Convert to categorical
df['category'] = df['category'].astype('category')

# Convert to string
df['id'] = df['id'].astype(str)

# Check data types
print(df.dtypes)
```

## 4. Handling Outliers

Outliers can skew analysis and affect model training.

### Detection Methods

**Statistical Methods**
```python
# Z-score method (values beyond 3 standard deviations)
from scipy import stats
z_scores = np.abs(stats.zscore(df['column']))
df_cleaned = df[z_scores < 3]

# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df[(df['column'] >= Q1 - 1.5*IQR) & (df['column'] <= Q3 + 1.5*IQR)]
```

### Handling Outliers
```python
# Remove outliers
df_cleaned = df[~((df['column'] > upper_limit) | (df['column'] < lower_limit))]

# Cap/Floor outliers (winsorization)
df['column'] = df['column'].clip(lower=lower_limit, upper=upper_limit)

# Replace with median
median = df['column'].median()
df.loc[outlier_mask, 'column'] = median
```

## 5. Text Data Cleaning

Text data requires specific preprocessing for NLP tasks.

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Convert to lowercase
df['text'] = df['text'].str.lower()

# Remove special characters and punctuation
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

# Remove extra whitespace
df['text'] = df['text'].str.strip()
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

# Remove URLs
df['text'] = df['text'].str.replace(r'http\S+|www\S+', '', regex=True)

# Remove HTML tags
df['text'] = df['text'].str.replace(r'<[^>]+>', '', regex=True)

# Remove numbers (if not needed)
df['text'] = df['text'].str.replace(r'\d+', '', regex=True)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatization/Stemming
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
```

## 6. Standardizing and Normalizing

Scaling numerical features improves model performance.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (Z-score normalization)
scaler = StandardScaler()
df[['column1', 'column2']] = scaler.fit_transform(df[['column1', 'column2']])

# Min-Max Scaling (normalize to 0-1)
scaler = MinMaxScaler()
df[['column1', 'column2']] = scaler.fit_transform(df[['column1', 'column2']])

# Robust Scaling (resistant to outliers)
scaler = RobustScaler()
df[['column1', 'column2']] = scaler.fit_transform(df[['column1', 'column2']])
```

## 7. Removing Irrelevant Features

Remove columns that don't contribute to analysis.

```python
# Remove columns with single value
df_cleaned = df.loc[:, (df != df.iloc[0]).any()]

# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
df_cleaned = df[df.columns[selector.get_support(indices=True)]]

# Drop specific columns
df_cleaned = df.drop(['column1', 'column2'], axis=1)

# Remove columns with high missing percentage
threshold = 0.5  # 50%
df_cleaned = df.dropna(axis=1, thresh=len(df)*(1-threshold))
```

## 8. Data Consistency and Validation

Ensure data quality and consistency.

### Categorical Consistency
```python
# Standardize categorical values
df['category'] = df['category'].str.lower().str.strip()

# Replace inconsistent values
df['category'] = df['category'].replace({'Yes': 1, 'No': 0})

# Check unique values
print(df['category'].unique())
print(df['category'].value_counts())
```

### Range Validation
```python
# Verify values are within expected range
valid_data = df[(df['age'] >= 0) & (df['age'] <= 120)]

# Check for invalid entries
invalid = df[~df['email'].str.contains('@', na=False)]
```

### Format Validation
```python
# Validate email format
import re
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
df['valid_email'] = df['email'].str.match(email_pattern)

# Validate phone format
phone_pattern = r'^\d{3}-\d{3}-\d{4}$'
df['valid_phone'] = df['phone'].str.match(phone_pattern)
```

## Complete Data Cleaning Workflow Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data.csv', encoding='latin-1')

# 1. Handle missing values
df.dropna(subset=['important_column'], inplace=True)
df['numeric_column'].fillna(df['numeric_column'].median(), inplace=True)

# 2. Remove duplicates
df.drop_duplicates(inplace=True)

# 3. Convert data types
df['date'] = pd.to_datetime(df['date'])
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

# 4. Handle outliers
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['amount'] >= Q1 - 1.5*IQR) & (df['amount'] <= Q3 + 1.5*IQR)]

# 5. Text cleaning
df['description'] = df['description'].str.lower().str.strip()

# 6. Normalize numerical features
scaler = StandardScaler()
df[['amount', 'quantity']] = scaler.fit_transform(df[['amount', 'quantity']])

# 7. Remove irrelevant columns
df.drop(['unnecessary_id'], axis=1, inplace=True)

# 8. Validation
print("Data shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Data types:\n", df.dtypes)
print("Summary statistics:\n", df.describe())

# Save cleaned data
df.to_csv('data_cleaned.csv', index=False)
```

## Best Practices

- **Understand your data first** - Explore before cleaning
- **Document your process** - Keep track of transformations applied
- **Create backups** - Save original data before cleaning
- **Validate after each step** - Ensure transformations work as expected
- **Consider domain knowledge** - Cleaning should be context-aware
- **Be cautious with outliers** - They may be legitimate values, not errors
- **Test on small samples** - Verify logic before applying to entire dataset
- **Keep raw data intact** - Use copies for cleaning operations

## Common Pitfalls to Avoid

- Removing too much data (data loss)
- Over-filling missing values without justification
- Ignoring data types and formats
- Not checking for data quality issues
- Inconsistent handling across similar columns
- Losing important information during transformation
- Forgetting to handle categorical variables properly
