# Practical No. 2: Data Wrangling II
# Create an "Academic performance" dataset of students.
# 1. Scan and handle missing values/inconsistencies.
# 2. Scan and handle numeric outliers.
# 3. Apply data transformations to decrease skewness and convert to normal distribution.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Create the "Academic performance" dataset ---

# Option A: Create a synthetic dataset directly in the code
data = {
    'Student_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'F'],
    'Math_Score': [85, 92, np.nan, 78, 90, 88, 150, 82, 95, np.nan], # 150 is an outlier, NaNs exist
    'Reading_Score': [88, 90, 85, 80, 10, 85, 92, 88, 90, 89],     # 10 is an outlier
    'Writing_Score': [90, 95, 88, 85, 89, 90, 93, 85, 92, 96],
    'Placement_Score': [75, 80, 82, 99, 100, 70, 78, 85, 92, 60]
}
df = pd.DataFrame(data)

# Option B: Load from a local CSV (Uncomment if dataset is provided locally)
# df = pd.read_csv("academic_performance.csv")

print("--- Initial Dataset ---")
print(df)

# --- Step 2: Handle Missing Values ---

print("\n--- Missing Values Before ---")
print(df.isnull().sum())

# We fill missing Math_Scores with the median (because median is robust against the outlier 150)
math_median = df['Math_Score'].median()  # median(): Returns the middle value, unaffected by extreme outliers.
df['Math_Score'] = df['Math_Score'].fillna(math_median)

print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())

# --- Step 3: Scan and Handle Outliers (Using IQR Method) ---
# IQR (Interquartile Range) Method calculates the 25th and 75th percentiles.

def handle_outliers_iqr(dataframe, column_name):
    """
    Function to detect and cap outliers using the IQR method.
    Outliers are replaced with the upper or lower bound values.
    """
    Q1 = dataframe[column_name].quantile(0.25)  # quantile(0.25): Returns the 25th percentile boundary (bottom quarter of data).
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nOutlier Bounds for {column_name}: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}")
    
    # Cap the outliers (replace values outside bounds with the bound values)
    dataframe[column_name] = np.where(dataframe[column_name] > upper_bound, upper_bound, dataframe[column_name])  # np.where(): A vectorized IF-ELSE that caps extreme values to the boundary.
    dataframe[column_name] = np.where(dataframe[column_name] < lower_bound, lower_bound, dataframe[column_name])
    return dataframe

# Handle outliers in ALL numeric columns
numeric_columns = ['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']
for col in numeric_columns:
    df = handle_outliers_iqr(df, col)

print("\n--- Dataset After Outlier Handling ---")
print(df[['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']])

# --- Step 4: Data Transformation (Log Transformation) ---
# We apply a Log Transformation to 'Placement_Score' to decrease skewness.
# Log transformations pull in large extreme values and stretch out small values, 
# making the distribution closer to a Normal (Gaussian) distribution.

# np.log1p computes log(1 + x) which handles zeros safely.
df['Placement_Score_Log'] = np.log1p(df['Placement_Score'])  # log1p(): Computes log(1+x), avoiding log(0) crash when value is zero.

print("\n--- Placement Score Before vs After Log Transformation ---")
print(df[['Placement_Score', 'Placement_Score_Log']].head())
































# --- Step 5: Academic Explanation Output ---
# This section prints a descriptive text block for the examiner as requested.

"""
1. MISSING VALUES & INCONSISTENCIES: 
   We scanned the dataset using 'isnull().sum()'. The 'Math_Score' contained missing 
   values (NaN). We imputed these gaps using the Median instead of the Mean. 
   Reason: The dataset contained a massive outlier (150). The Mean is easily skewed 
   by outliers, making it an inaccurate representation. The Median is robust.

2. OUTLIER HANDLING: 
   We used the Interquartile Range (IQR) method to detect numeric outliers in 'Math_Score' 
   and 'Reading_Score'. Any value beyond 1.5 * IQR from the Q1/Q3 boundaries was identified.
   Reason: Instead of deleting these rows and losing valuable data, we 'Capped' them using 
   'np.where', replacing extreme values with the exact boundary limit.

3. DATA TRANSFORMATION: 
   We applied a Logarithmic Transformation (np.log1p) to the 'Placement_Score' variable.
   Reason: To decrease the skewness of the variable and convert its non-linear, skewed 
   distribution into a shape resembling a standard Normal Distribution (Bell Curve). This 
   is a strict prerequisite for many parametric machine learning models.
"""
"""
--- Code Explanation ---
1. pd.DataFrame: Creates a table-like dataset from a Python dictionary.
2. np.nan: Injects intentional "missing" values to simulate messy real-world data.
3. df.quantile(): Finds the 25th (Q1) and 75th (Q3) percentiles needed for the IQR outlier method.
4. np.where(condition, x, y): If condition is true, yield x, otherwise yield y. We use this to "cap" outliers.
5. np.log1p(): A mathematical Logarithmic transformation. It squashes skewed data to make it look like a Bell Curve (Normal Distribution).

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS AN OUTLIER?
   - Definition: A data point significantly different from the rest of the dataset.
   - Example (Error): A student's Math Score of 150 on a 100-point test is a data entry error.
   - Example (Natural): A billionaire's salary in a dataset of average workers is a natural outlier.
   - Types:
     a) Univariate Outlier: Extreme in a single column (e.g., Age = 900).
     b) Multivariate Outlier: Normal individually but extreme in combination (e.g., a 5-year-old with a PhD).
   - Impact: Outliers distort the Mean, Standard Deviation, and break linear regression models.

2. METHODS TO HANDLE OUTLIERS:
   a) IQR Method (Interquartile Range) — Used in this practical:
      - Q1 = 25th Percentile (value below which 25% of data falls).
      - Q3 = 75th Percentile (value below which 75% of data falls).
      - IQR = Q3 - Q1 (the width of the middle 50% of data).
      - Lower Bound = Q1 - 1.5 * IQR.
      - Upper Bound = Q3 + 1.5 * IQR.
      - Advantage: Does not assume a normal distribution. Works on any data shape.
   b) Z-Score Method: Calculates how many standard deviations a point is from the mean.
      - If |Z-Score| > 3, it is an outlier. Requires normally distributed data.
   c) Capping / Winsorization (used in this code):
      - Instead of deleting the outlier row, we "cap" the extreme value.
      - Values above Upper Bound → replaced WITH the Upper Bound.
      - Values below Lower Bound → replaced WITH the Lower Bound.
      - Advantage: Preserves all rows of data. No data is lost.
   d) Deletion: Remove the entire row. Safe only for large datasets.

3. MISSING VALUE IMPUTATION:
   - Mean Imputation: Replace NaN with the column average.
     - Risk: If data is skewed or has outliers, the mean is misleading. (Median is preferred.)
   - Median Imputation: Replace NaN with the middle value. Robust to outliers. (Used here.)
   - Mode Imputation: Replace NaN with the most frequent value. Best for categorical columns.

4. DATA TRANSFORMATIONS — TYPES AND PURPOSES:
   - Definition: Applying a mathematical function to a column to change its distribution shape.
   - Goal 1: Change Scale (e.g., convert meters to centimeters).
   - Goal 2: Linearize — Convert a curved/exponential relationship into a straight line so
     linear regression can model it accurately.
   - Goal 3: Reduce Skewness — Make a skewed distribution closer to a Normal (Bell) curve.
   - Common Transformations:
     a) Log Transformation — np.log1p(x): Best for right-skewed data (salaries, prices).
        Compresses large values, expands small values.
     b) Square Root — np.sqrt(x): Milder than log. Good for count data (number of events).
     c) Square — x**2: Amplifies large values. Useful for left-skewed data.

5. SKEWNESS AND NORMAL DISTRIBUTION:
   - Normal Distribution: A symmetric, bell-shaped curve. Mean = Median = Mode.
     Most parametric ML models work best when data is normally distributed.
   - Right-Skewed (Positive): Long tail to the right. Most values are small; a few are
     extremely large (e.g., income data — most earn little, a few earn millions).
   - Left-Skewed (Negative): Long tail to the left. Most values are large; a few are tiny.
   - How Log fixes Right-Skew: log() compresses massive right-tail values inward,
     pulling them back toward the center and making the shape symmetric.

--- Detailed Viva Q&A ---

Q1: Why did you use Median instead of Mean to fill missing Math scores?
A1: Because the Math column had an extreme outlier (150 on a 100-point scale). The Mean is pulled toward extreme values, making it an inaccurate replacement for gaps. The Median is "robust" — it ignores outliers completely and gives the true middle value.

Q2: What is an Outlier? Give a real-world example.
A2: A data point that stands out significantly from the rest. Example: In a student age dataset (ages 5-80), an age of 900 is clearly a data entry error. In company salary data, a CEO earning 100x more than employees is a natural but extreme outlier.

Q3: How does the IQR method detect outliers?
A3: It identifies the middle 50% of data (Q1 to Q3), calculates IQR = Q3-Q1, then creates a "safe fence" (Upper = Q3+1.5*IQR, Lower = Q1-1.5*IQR). Any point outside this fence is mathematically flagged as an outlier.

Q4: What is the purpose of Log Transformation in Step 4?
A4: Placement_Score was right-skewed — most scores were normal but a few were extremely high. Applying log1p() compresses those extreme high values, pulling the distribution toward a symmetric bell curve (Normal Distribution), which many ML algorithms require.

Q5: What is `np.where()` doing in the outlier-capping code?
A5: It is a vectorized IF-ELSE. For every value in the column: IF value > Upper Bound → replace WITH Upper Bound, ELSE keep original. This is capping/winsorization — it clips extreme values without deleting any rows.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What does `np.log1p` mean and why not use `np.log`?
   A1: `log1p(x)` computes `log(1 + x)`. If data contains a zero, `log(0)` is negative infinity (a crash). `log(1+0) = log(1) = 0`, which is perfectly safe. Always prefer log1p for data that may contain zeros.

   Q2: Can we handle outliers by deleting rows?
   A2: Yes, and it's the simplest method. But if the dataset is small (like ours with 10 students), deleting 2 outlier rows means losing 20% of all data. Capping preserves every row while still neutralizing extreme values.

   Q3: What is the difference between `quantile(0.25)` and `quantile(0.75)`?
   A3: `quantile(0.25)` returns Q1 — the value below which 25% of data falls (bottom quarter boundary). `quantile(0.75)` returns Q3 — the value below which 75% of data falls (top quarter boundary).

------------------------- End of Viva Notes -------------------------
"""
