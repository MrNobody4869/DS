# Practical No. 1: Data Wrangling I
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# --- Step 1: Load the Dataset ---

# Option A: Fetch from personal GitHub repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/train.csv"
df = pd.read_csv(url)

# Option B: Local CSV (Matches your 'train.csv' file)
# Uncomment below and comment out Option A to use local file:
# df = pd.read_csv("train.csv")

print("--- Initial Data View ---")
print(df.head()) # Shows the first 5 rows

# --- Step 2: Initial Statistics & Variable Descriptions ---

print("\n--- Data Information (Types & Missing Values) ---")
# .info() gives variable types and non-null counts
df.info()  #info(): Instantly shows data types and exactly how many missing (null) values exist.

print("\n--- Summary Statistics ---")
# .describe() gives mean, std, min, max, etc., for numerical columns
print(df.describe())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())  # isnull() flags missing cells, sum() counts them up per column.

print("\n--- Data Dimensions ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}") # .shape: It returns the dimensions (number of rows and columns).

print("\n--- Variable Type Summary ---")
# Manually summarizing types as requested by problem statement
print("Numeric Variables:", df.select_dtypes(include=['number']).columns.tolist())
print("Categorical/Object Variables:", df.select_dtypes(include=['object', 'category']).columns.tolist())

# --- Step 3: Data Formatting & Handling Missing Values ---

# Age has missing values. Let's fill them with the mean (average) age.
# fillna(): Replaces NaN values with a specified value.
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)  #fillna: Safely replaces empty (NaN) cells without deleting valuable rows.

# Cabin has too many missing values, so we drop the column entirely.
# drop(): Removes specified labels from rows or columns.
df = df.drop(columns=['Cabin'])

# Embarked has a few missing values. We fill them with the mode (most frequent value).
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# Convert 'Survived' and 'Pclass' to categorical data types for proper formatting
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# --- Step 4: Data Normalization ---

# We will normalize the 'Fare' column because the values vary wildly.
# MinMaxScaler scales data to a fixed range, usually 0 to 1.
scaler = MinMaxScaler()
# We must pass it as a 2D array, hence [['Fare']]
df['Fare_Normalized'] = scaler.fit_transform(df[['Fare']])  # fit_transform: 'fit' finds the min/max limits, 'transform' mathematically shrinks the data to 0-1.

print("\n--- Fare Before vs After Normalization ---")
print(df[['Fare', 'Fare_Normalized']].head())

# --- Step 5: Categorical to Quantitative Conversion ---

# Approach 1: Label Encoding (e.g., Sex: male -> 1, female -> 0)
# LabelEncoder assigns a unique integer to each category.
le = LabelEncoder()
df['Sex_Encoded'] = le.fit_transform(df['Sex'])

# Approach 2: One-Hot Encoding (e.g., Embarked ports C, Q, S become separate 0/1 columns)
# pd.get_dummies() creates new columns for each category with 1s and 0s.
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  #drop_first=True: Prevents the 'Dummy Variable Trap' by removing redundant duplicate columns.








































# --- Step 6: Academic Explanation Output ---

"""
1. IMPORTING DATA: We utilized 'pandas.read_csv' to fetch the Titanic dataset from an open-source 
   GitHub repository. This allows for reproducible research without requiring local files.
2. PREPROCESSING: We used 'isnull()' to detect missing values. We found gaps in Age and 
   Embarked, which we filled (imputed) using the Mean and Mode respectively to prevent data loss.
3. DATA FORMATTING: We checked the 'dtypes' and used 'astype' to ensure columns like 
   'Survived' and 'Pclass' are treated as categorical factors rather than simple integers.
4. DIMENSIONS: We used 'df.shape' to verify that our dataset consists of 891 rows and 12 columns.
5. NORMALIZATION: We applied 'MinMaxScaler' to the 'Fare' column. This rescales ticket prices 
   to a 0-1 range, ensuring the variable's magnitude doesn't bias future machine learning models.
6. QUANTITATIVE CONVERSION: We converted 'Sex' (text) into 'Sex_Encoded' (binary 0/1) and 
   created Dummy Variables for 'Embarked' ports to turn text categories into numeric features.
"""
"""
--- Code Explanation ---
1. Data Sourcing: We located the Titanic dataset from the Data Science Dojo GitHub repository (a reliable open-source provider similar to Kaggle).
2. pd.read_csv(url): This command 'scrapes' or reads the data directly from the web link into a Python table (DataFrame).
3. df.shape: Returns the (rows, columns) to understand the size of our dataset.
4. df.info() & df.describe(): Summarizes the variable types (int, float, object) and calculates basic statistics (mean, min, max).
5. df.isnull().sum(): Identifies precisely where data is missing so we can fix it.
6. df.fillna(): Replaces empty cells with the average (mean) or most frequent (mode) value so the data is complete.
7. MinMaxScaler: Squashes a wide range of numbers (like Fare prices) into a strict 0.0 to 1.0 range (Normalization).
8. LabelEncoder: Converts text categories into simple numbers (Male/Female -> 1/0).
9. pd.get_dummies: Creates "One-Hot" encoded columns to represent multi-category variables as quantitative 0/1 bits.
_________________________
------------------------- Detailed Theory Notes (Full Concept Guide) -------------------------

1. WHAT IS DATA WRANGLING?
   - Definition: The process of cleaning, transforming, and mapping raw data into a usable format.
   - Also called: Data Munging or Data Preparation.
   - Analogy: Think of raw data like a messy room. Data Wrangling is the act of organizing, cleaning,
     and arranging everything before a guest (your ML model) arrives.
   - Core Activities:
     a) Discovery    - Understanding what data you have (shape, types, sample values).
     b) Structuring  - Reformatting data into a table with rows and columns.
     c) Cleaning     - Fixing missing values, typos, and inconsistencies.
     d) Enriching    - Adding new calculated columns (e.g., Age Groups from Age).
     e) Validating   - Confirming the data is logically consistent after all changes.
   - Importance: The famous rule "Garbage In, Garbage Out (GIGO)." Even the most powerful
     ML model will produce wrong results if the input data is dirty.

2. MISSING VALUES (NaN / Nulls):
   - Definition: A missing value is a data point with no recorded value. In Pandas, it shows as NaN
     (Not a Number).
   - Cause: Sensor failure, user skipping a form field, data entry errors, or merging tables.
   - Impact: Most scikit-learn ML models crash when they encounter NaN values.
   - How to detect: `df.isnull().sum()` — counts missing values per column.
   - Strategies to handle:
     a) Deletion (dropna): Delete the entire row. Safe only if less than 5% data is missing.
     b) Mean Imputation: Fill with the column average. Good for numeric, symmetric data.
     c) Median Imputation: Fill with the middle value. Best when data has outliers (skewed).
     d) Mode Imputation: Fill with the most frequent value. Best for categorical text columns.
     e) Forward Fill (ffill): Copy the previous row's value. Best for time-series data.
   - Example: Age column has NaN → fill with Mean Age (e.g., 29.7).

3. DATA TYPES IN PYTHON / PANDAS:
   - int64    : Whole numbers (e.g., Pclass=1, SibSp=3). No decimals.
   - float64  : Decimal numbers (e.g., Age=29.5, Fare=72.30).
   - object   : Text strings (e.g., Name='John', Sex='male').
   - category : A special type for low-cardinality text (e.g., Survived can only be 0 or 1).
   - bool     : True or False values.
   - Why it matters: Using the wrong type wastes memory and causes calculation errors.
     e.g., if 'Survived' is stored as float64, Python might compute the "average survival",
     which is meaningless. Converting it to 'category' prevents this.

4. DATA NORMALIZATION — MIN-MAX SCALING:
   - Definition: Rescaling all numeric values in a column to a fixed range of 0.0 to 1.0.
   - Formula: X_new = (X - X_min) / (X_max - X_min)
   - Example: Fare prices range from 0 to 500. After scaling, a fare of 250 becomes 0.5.
   - Why needed: ML algorithms like KNN and Neural Networks use mathematical "distance"
     between data points. If Fare is in the thousands and Age is 0-80, the algorithm
     thinks Fare is automatically more important. Normalization removes this bias.
   - Tool: `MinMaxScaler` from sklearn.preprocessing.
   - Other scaling methods:
     a) StandardScaler (Z-score): Centers data at Mean=0, Variance=1. Better when outliers exist.
     b) RobustScaler: Uses median and IQR. Excellent for datasets with many extreme outliers.

5. CATEGORICAL TO QUANTITATIVE CONVERSION:
   - Definition: Converting text labels into numeric codes that ML algorithms can process.
   - Why: Computers only understand binary numbers (0 and 1). They cannot do math on text.
   - Type 1 — Label Encoding:
     - Converts: Male=0, Female=1 (or any ordered number).
     - Best for: ORDINAL data where order matters (Low=0, Medium=1, High=2).
     - Risk: The model may incorrectly think 2 (High) is mathematically double 1 (Medium).
   - Type 2 — One-Hot Encoding (pd.get_dummies):
     - Creates a separate binary column for each category.
     - Example: 'Embarked' with values C, Q, S → creates 3 columns: Embarked_C, Embarked_Q, Embarked_S.
     - Best for: NOMINAL data where order does NOT matter (e.g., city names, port names).
     - drop_first=True: Drops one column to avoid the "Dummy Variable Trap" (Multicollinearity).

6. DATA DIMENSIONS (df.shape):
   - df.shape returns a tuple: (rows, columns).
   - Example: (891, 12) → 891 passengers, 12 attributes each.
   - Why it matters: Confirms no rows were accidentally deleted during cleaning.
   - Big Data Rule of Thumb: If rows < 100,000 → Small Data. If rows > 1 million → Big Data.

--- Detailed Viva Q&A ---

Q1: What is a Pandas DataFrame?
A1: It is a 2-dimensional, table-like data structure in Python, very similar to an Excel spreadsheet, with labeled rows and columns. Every column can hold a different data type.

Q2: What is the difference between `df.describe()` and `df.info()`?
A2: `info()` tells you the data types (int, float, object) and how many non-null values exist. `describe()` gives mathematical statistics (mean, min, max, standard deviation) for all numeric columns.

Q3: Why did we drop the 'Cabin' column?
A3: Because over 70% of its data was missing. Imputing (guessing) that much missing data would introduce massive errors into our dataset. It is safer to simply remove the column entirely.

Q4: Explain "One-Hot Encoding" vs "Label Encoding".
A4: Label Encoding converts categories into sequential numbers (0, 1, 2). One-Hot Encoding creates brand new binary columns for each category (1 if true, 0 if false). We prefer One-Hot for nominal categories to avoid the model thinking category '2' is mathematically greater than '0'.

Q5: Why did we use the 'Mean' to fill Age, but the 'Mode' to fill Embarked?
A5: Age is a continuous numeric variable — calculating an average (mean) makes logical sense. 'Embarked' is a categorical text variable (a port city name) — you cannot average city names. We use the most frequent city (mode) instead.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What does `drop_first=True` do in `pd.get_dummies`?
   A1: It prevents the "Dummy Variable Trap" (Multicollinearity). If a passenger is NOT from Port Q and NOT from Port S, the model automatically knows they MUST be from Port C. We don't need a third redundant column stating the obvious.

   Q2: What does `scaler.fit_transform()` actually do behind the scenes?
   A2: The `fit` part calculates the minimum and maximum values of the column. The `transform` part applies the formula to shrink all numbers between 0 and 1 using those min/max values.

   Q3: What is the difference between `fit_transform()` and `transform()`?
   A3: We call `fit_transform()` ONLY on the Training set (it learns the min/max from training data).
   We call only `transform()` on the Test set (applies the same min/max learned from training, to prevent data leakage).

------------------------- End of Viva Notes -------------------------
"""
