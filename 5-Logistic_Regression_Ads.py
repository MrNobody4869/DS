# Practical No. 5: Data Analytics II
# 1. Implement logistic regression using Python to perform classification on Social_Network_Ads.csv dataset.
# 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

# Option A: Fetch from personal GitHub repo (use if local file is unavailable)
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/Social_Network_Ads.csv"
df = pd.read_csv(url)

# # Option B: Load from local CSV file (preferred in lab)
# df = pd.read_csv("Social_Network_Ads.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# -------------------------------------------------
# Step 2: Check Missing Values
# -------------------------------------------------

print("\nMissing values in dataset:")
print(df.isnull().sum())

# -------------------------------------------------
# Step 3: Convert Categorical to Numerical (if needed)
# -------------------------------------------------

# map(): Replaces each string value with a number. Male→0, Female→1.
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    print("\nAfter encoding Gender column:")
    print(df.head())

# -------------------------------------------------
# Step 4: Select Features and Target
# -------------------------------------------------

# Using Age and EstimatedSalary as input features
X = df[['Age', 'EstimatedSalary']]

# Purchased (0=No, 1=Yes) is what we want to predict
y = df['Purchased']

# -------------------------------------------------
# Step 5: Split Dataset (75% train, 25% test)
# -------------------------------------------------

# train_test_split(): Shuffles and divides data so the model is tested on unseen samples.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# -------------------------------------------------
# Step 6: Feature Scaling
# -------------------------------------------------

# StandardScaler(): Standardizes features to Mean=0, Variance=1.
# Essential because Salary (50,000) would mathematically overpower Age (25) without scaling.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # fit_transform(): Learns mean/std from train data, then scales it.
X_test = sc.transform(X_test)        # transform(): Applies the SAME scaling learned from train data. Never re-fit on test data.

# -------------------------------------------------
# Step 7: Create and Train the Logistic Regression Model
# -------------------------------------------------

model = LogisticRegression()
# fit(): Trains the model — finds the best S-curve (sigmoid) separating buyers from non-buyers.
model.fit(X_train, y_train)

# -------------------------------------------------
# Step 8: Make Predictions
# -------------------------------------------------

# predict(): Applies learned sigmoid function to test data. Outputs 0 or 1 for each sample.
y_pred = model.predict(X_test)

print("\nPredicted values (first 10):")
print(y_pred[:10])

# -------------------------------------------------
# Step 9: Confusion Matrix
# -------------------------------------------------

# confusion_matrix(): Produces a 2x2 grid of TP, TN, FP, FN counts.
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Extract the 4 values from the matrix
TN = cm[0][0]   # Predicted NO, Actual NO — Correct
FP = cm[0][1]   # Predicted YES, Actual NO — Wrong (Type 1 Error)
FN = cm[1][0]   # Predicted NO, Actual YES — Wrong (Type 2 Error)
TP = cm[1][1]   # Predicted YES, Actual YES — Correct

print(f"\nTrue Positives  (TP): {TP}  — Predicted Buy, Actually Bought")
print(f"True Negatives  (TN): {TN}  — Predicted Not Buy, Didn't Buy")
print(f"False Positives (FP): {FP}  — Predicted Buy, Didn't Actually Buy (Type 1 Error)")
print(f"False Negatives (FN): {FN}  — Predicted Not Buy, Actually Bought (Type 2 Error)")

# -------------------------------------------------
# Step 10: Performance Metrics (Calculated Manually)
# -------------------------------------------------

# Manual formula calculation proves formula knowledge during viva
accuracy   = (TP + TN) / (TP + TN + FP + FN)  # Overall correct predictions / total
error_rate = (FP + FN) / (TP + TN + FP + FN)  # Overall wrong predictions / total
precision  = TP / (TP + FP)                    # Of predicted buyers, how many actually bought?
recall     = TP / (TP + FN)                    # Of actual buyers, how many did we correctly catch?

print(f"\nAccuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Error Rate: {error_rate:.4f}  ({error_rate*100:.2f}%)")
print(f"Precision:  {precision:.4f}  ({precision*100:.2f}%)")
print(f"Recall:     {recall:.4f}  ({recall*100:.2f}%)")




























"""
--- Code Explanation ---
1. pd.read_csv(): Loads the Social Network Ads dataset (400 records, columns: Age, EstimatedSalary, Purchased).
2. df['Gender'].map(): Converts categorical text ('Male'/'Female') into numeric (0/1) so the model can do math on it.
3. train_test_split(): Splits 75% for training and 25% for testing so we can verify the model on unseen data.
4. StandardScaler(): Standardizes Age and Salary to the same scale (Mean=0, Std=1). Without this, Salary (50,000) drowns out Age (25) in the math.
5. LogisticRegression(): The classification model. It finds the best S-shaped sigmoid curve to separate buyers from non-buyers.
6. model.fit(): Trains the model on scaled training data.
7. model.predict(): Uses the trained model to classify each test sample as 0 (Not Bought) or 1 (Bought).
8. confusion_matrix(): Returns a 2x2 grid showing correct (TP/TN) and incorrect (FP/FN) prediction counts.
9. Manual Metric Formulas: Accuracy, Error Rate, Precision, and Recall are calculated directly from TP/TN/FP/FN to demonstrate formula knowledge.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS LOGISTIC REGRESSION?
   - Definition: A Supervised Machine Learning algorithm used for CLASSIFICATION tasks.
   - Despite its name containing "Regression," it predicts categories, NOT numbers.
   - Output: A probability between 0.0 and 1.0. If probability > 0.5 → Class 1 (Yes). Else → Class 0 (No).
   - Example uses: Email Spam detection, Disease diagnosis, Ad click prediction (used here).
   - Key difference from Linear Regression: Linear Regression draws a straight line (outputs any number).
     Logistic Regression draws an S-shaped curve (outputs a probability 0 to 1).

2. THE SIGMOID FUNCTION (The Core of Logistic Regression):
   - Formula: σ(z) = 1 / (1 + e^(-z))
   - What it does: Takes ANY number z (from -∞ to +∞) and squashes it into a value between 0 and 1.
   - z is the standard linear formula (z = m1x1 + m2x2 + c) from Linear Regression.
   - Example:
     z = +5 → σ(5) ≈ 0.99 → Model is 99% confident it's Class 1.
     z = -5 → σ(-5) ≈ 0.01 → Model is 99% confident it's Class 0.
     z = 0  → σ(0)  = 0.50 → Model is 50/50, completely uncertain.
   - Decision Boundary: The point where the S-curve crosses probability = 0.5.

3. THE CONFUSION MATRIX (Detailed):
   - Definition: A 2x2 table that shows exactly how many correct and incorrect predictions the model made for each class.
   - Structure (for Binary Classification):
     
     |               | Predicted: YES | Predicted: NO  |
     | Actual: YES   | True Positive  | False Negative |
     | Actual: NO    | False Positive | True Negative  |
     
   - True Positive (TP): Model said YES, it was actually YES. Correct! (e.g., predicted Buy, actually Bought.)
   - True Negative (TN): Model said NO, it was actually NO. Correct! (e.g., predicted Not Buy, didn't Buy.)
   - False Positive (FP): Model said YES, but it was actually NO. WRONG! (Type 1 Error / "False Alarm.")
     Example: System flags innocent email as spam.
   - False Negative (FN): Model said NO, but it was actually YES. WRONG! (Type 2 Error / "Miss.")
     Example: Cancer screening misses a real tumor.

4. EVALUATION METRICS — DEFINITIONS AND FORMULAS:
   a) Accuracy = (TP + TN) / (TP + TN + FP + FN)
      - Overall: "Out of all predictions, how many were correct?"
      - Problem: Misleading when classes are imbalanced (e.g., 999 normal, 1 fraud).
   b) Error Rate = 1 - Accuracy = (FP + FN) / Total
      - Overall: "Out of all predictions, how many were WRONG?"
   c) Precision = TP / (TP + FP)
      - "Of all the people I claimed WOULD BUY, how many actually did?"
      - Precision is important when False Positives are costly (e.g., sending spam ad to non-buyer wastes money).
   d) Recall (Sensitivity) = TP / (TP + FN)
      - "Of all the people who ACTUALLY BOUGHT, how many did I correctly identify?"
      - Recall is critical when False Negatives are dangerous (e.g., missing a cancer diagnosis).
   e) F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
      - The harmonic mean of Precision and Recall. Used when you need a balanced single metric.

5. STANDARDSCALER — FEATURE SCALING:
   - Definition: Transforms each feature column to have Mean=0 and Standard Deviation=1.
   - Formula: X_scaled = (X - Mean) / Standard Deviation.
   - Why essential for Logistic Regression: The algorithm uses mathematical distances and gradients.
     If "Salary" ranges in the tens of thousands and "Age" ranges 18-60, without scaling,
     the algorithm treats Salary as thousands of times more important. Scaling balances them.
   - fit_transform on TRAIN only: We learn the mean/std from training data only.
     We then apply those same statistics to the test data (transform only).
     This prevents "data leakage" (test data influencing training statistics).

--- Detailed Viva Q&A ---

Q1: Why is it called Logistic Regression if it's used for Classification?
A1: Because the core math is identical to Linear Regression. It calculates z = mx+c (a continuous linear value), but then wraps it in the "Logistic" (Sigmoid) mathematical function to convert that value into a 0-to-1 probability for classification.

Q2: Why do we need to "Scale" data using StandardScaler before training?
A2: Logistic Regression relies on gradient descent (an optimization algorithm that uses mathematical distances). Without scaling, "Salary" (in the thousands) dominates "Age" (0-60) purely due to magnitude — not actual importance. Scaling forces all features to compete on equal mathematical footing.

Q3: What is the difference between Type 1 and Type 2 Error?
A3: Type 1 Error = False Positive (False Alarm): Model screams "YES!" when the real answer is "NO." Example: Telling a healthy person they have cancer. Type 2 Error = False Negative (Miss): Model says "NO" when the real answer is "YES." Example: Missing a real tumor in a cancer patient.

Q4: If a model has 99% accuracy, is it always good?
A4: No — this is the "Accuracy Paradox." If your dataset has 99 normal emails and 1 spam, a model that blindly labels EVERYTHING as "Not Spam" achieves 99% accuracy while completely failing its actual job. This is why we need Precision, Recall, and the Confusion Matrix.

Q5: When should you prioritize "Recall" over "Precision"?
A5: Prioritize Recall when False Negatives are more dangerous. Medical diagnostics: better to have a False Positive (healthy person told to do more tests) than a False Negative (sick patient told they're healthy). Prioritize Precision when False Positives are costly (e.g., spam filtering — you don't want real emails deleted).

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What does `cm.ravel()` do in your code?
   A1: `confusion_matrix()` returns a 2D NumPy array (a grid). `ravel()` flattens that 2D grid into a 1D list of 4 numbers, allowing us to directly assign: `TN, FP, FN, TP = cm.ravel()` in a single clean line.

   Q2: What is the Decision Boundary in Logistic Regression?
   A2: It is the threshold (default = 0.5) that separates the two classes. If the predicted probability > 0.5, the model predicts Class 1 (Purchased). If ≤ 0.5, it predicts Class 0 (Not Purchased). This threshold can be adjusted based on whether we want to prioritize Precision or Recall.

   Q3: What does `solver='lbfgs'` mean in LogisticRegression()?
   A3: It specifies the mathematical optimization algorithm used to find the best weights. 'lbfgs' stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno — it is efficient for small to medium datasets and is the default in modern sklearn.

------------------------- End of Viva Notes -------------------------
"""
