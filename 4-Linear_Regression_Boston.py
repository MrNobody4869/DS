# Practical No. 4: Data Analytics I
# Create a Linear Regression Model using Python to predict home prices using the Boston Housing Dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Load the Dataset ---

# Option A: Fetch the Boston Housing Dataset online (target column is named 'medv')
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/Housing.csv"
df = pd.read_csv(url)
TARGET_COL = 'price'  # GitHub/Local Housing.csv uses 'price' as the house price column

# Option B: Load from local Housing.csv if provided in the lab
# Uncomment the 3 lines below and comment out Option A above to switch:
# df = pd.read_csv("Housing.csv")
# TARGET_COL = 'price'  # Local Housing.csv uses 'price' as the house price column
# print("[INFO] Loaded local Housing.csv — target column is 'price'")

print("--- Initial Dataset View ---")
print(df.head())

# --- Step 1.5: Categorical Encoding (Required for local Housing.csv) ---
# Linear Regression only understands numbers. Local Housing.csv often has 
# 'yes'/'no' text. We must convert these to 0s and 1s.
text_columns = df.select_dtypes(include=['object']).columns
if len(text_columns) > 0:
    print(f"\n[INFO] Converting text columns to numeric: {text_columns.tolist()}")
    # get_dummies(): Creates new binary columns (1 for yes, 0 for no).
    # drop_first=True: Prevents redundancy (if it's not 'yes', it must be 'no').
    df = pd.get_dummies(df, columns=text_columns, drop_first=True)

# Confirm dimensions
print(f"\n--- Dataset Dimensions ---")
print(f"Rows (Samples): {df.shape[0]}, Columns (Features + Target): {df.shape[1]}")
print("Features in use:", df.columns.tolist())
print(f"Target Column in use: '{TARGET_COL}'")

# Separate features (X) and target (y) using the active TARGET_COL variable
X = df.drop(columns=[TARGET_COL])  # drop(): Removes the target column to isolate input features.
y = df[TARGET_COL]                 # This is what we want to predict

# --- Step 2: Split the Data ---
# We split the data: 75% for training the model, 25% for testing the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # random_state: Fixes the random seed so the data split is identical on every run.

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- Step 3: Build and Train the Linear Regression Model ---

# Initialize the model
model = LinearRegression()

# Train (fit) the model using the training data
print("\n--- Training the Linear Regression Model ---")
model.fit(X_train, y_train)  # fit(): Trains the model by studying the training data to learn patterns and optimal coefficients.

# --- Step 4: Make Predictions and Evaluate ---

# Make predictions on the unseen testing data
y_pred = model.predict(X_test)

# Evaluate how well the model did
mse = mean_squared_error(y_test, y_pred)  # mean_squared_error(): Squares each prediction error to eliminate negatives and heavily penalize large mistakes.
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R2): {r2:.2f}")

# Optional: Print the coefficients (the "weights" the model learned for each feature)
# This tells us which features are most important for predicting the house price.
print("\n--- Feature Coefficients ---")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# --- Step 5: Visualize the Results (The "Visual Proof") ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True)

# Backup: Save the plot as an image file in case the window doesn't pop up
plt.savefig("house_price_prediction.png")
print("\n[INFO] Plot saved as 'house_price_prediction.png' in your project folder.")

print("[INFO] Displaying Plot window... (Check your taskbar for a new icon)")
plt.show()
































"""
--- Code Explanation ---
1. pd.read_csv: Loads the raw Boston Housing data from a direct URL into a DataFrame.
2. df.drop(columns=['medv']): Removes the target column to isolate our 13 input features.
3. train_test_split: Shuffles and cuts the dataset into Training (80%) and Testing (20%) subsets so we can verify if the model memorized or actually learned.
4. LinearRegression(): Creates an empty mathematical model.
5. model.fit(): The algorithm studies the training data and finds the best straight line (formula) to predict the prices.
6. model.predict(): We ask the trained model to guess the prices for the hidden 20% test set.
7. mean_squared_error & r2_score: Mathematical formulas used to grade the model's performance.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS LINEAR REGRESSION?
   - Definition: A Supervised Machine Learning algorithm that finds the best straight-line
     relationship between input features (X) and a continuous numeric output (y).
   - Type: Regression (predicts a number, NOT a category).
   - Goal: Draw the "Line of Best Fit" through the scatter plot of data points.
   - Examples of use: Predicting house prices, stock prices, exam scores, temperature.
   - Contrast with Classification: Classification predicts categories (Dog/Cat, Spam/Not).
     Regression predicts a continuous number (Price = $245,000).

2. TYPES OF LINEAR REGRESSION:
   a) Simple Linear Regression:
      - One input feature (X), one output (y).
      - Formula: y = mx + c
        (y = Prediction, m = Slope/Weight, x = Input feature, c = Y-Intercept).
      - Example: Predict salary based only on years of experience.
   b) Multiple Linear Regression (used in this practical):
      - Multiple input features (X1, X2, X3...), one output (y).
      - Formula: y = m1x1 + m2x2 + m3x3 + ... + c
      - Example: Predict house price based on 13 features (rooms, crime rate, tax, etc.).
      - The model finds the best "weight" (coefficient) for each feature.

3. KEY CONCEPTS:
   a) Coefficient (Weight / Slope):
      - Each input feature gets assigned a coefficient (m).
      - Positive coefficient: As the feature increases, price increases (e.g., more rooms = higher price).
      - Negative coefficient: As the feature increases, price decreases (e.g., higher crime rate = lower price).
      - The magnitude of the coefficient shows HOW MUCH influence that feature has.
   b) Intercept (c):
      - The predicted price when ALL feature values are zero. Usually not meaningful alone.
   c) Residual (Error):
      - The difference between the actual real price and the model's predicted price.
      - Goal of Linear Regression: Find the coefficients that MINIMIZE the total squared residuals.
      - This minimization process is called "Ordinary Least Squares (OLS)."

4. EVALUATION METRICS DEFINED:
   a) Mean Squared Error (MSE):
      - Formula: Average of (Actual_Price - Predicted_Price)^2 for all test samples.
      - Why square? To eliminate negative errors (a $10,000 underestimate is just as bad as a $10,000 overestimate).
      - Why squaring is powerful: It PUNISHES large errors heavily. A $20,000 error contributes
        4x more to MSE than a $10,000 error (because 20^2 = 400 vs 10^2 = 100).
      - Lower MSE = Better model. Units: Dollars squared ($$^2).
   b) Root Mean Squared Error (RMSE):
      - RMSE = sqrt(MSE). Same unit as the original data (dollars).
      - Easier to interpret: "On average, my house price prediction is off by $X."
   c) R-Squared Score (R2 / Coefficient of Determination):
      - Range: 0.0 to 1.0 (sometimes negative for terrible models).
      - Meaning: "What percentage of the price variation is explained by our features?"
      - R2 = 0.85 → 85% of house price differences can be explained by our 13 features.
      - R2 = 1.0 → Perfect model (usually means overfitting — memorized the training data).
      - R2 = 0.0 → Model is no better than guessing the average price every time.

5. TRAIN / TEST SPLIT:
   - Why split? If we train AND test on the same data, the model just memorizes answers (cheating).
     It would score 100% on training data but fail on new, real-world data.
   - Standard split: 80% Training, 20% Testing.
   - Analogy: Studying from a textbook (training) and then taking an exam with new questions (testing).
   - random_state=42: Sets a fixed "seed" for the random shuffle so the split is identical every run.
     Without it, you get different splits each time, making results impossible to reproduce.

6. BOSTON HOUSING DATASET:
   - Contains 506 house records from Boston, USA (1970s census data).
   - Target: 'medv' = Median value of homes in $1000s.
   - 13 Input Features: CRIM (crime rate), ZN (residential land), INDUS (industry),
     CHAS (river adjacency), NOX (air pollution), RM (rooms), AGE (old buildings),
     DIS (distance to employment), RAD (highway access), TAX (tax rate),
     PTRATIO (pupil-teacher ratio), B (proportion of Black residents), LSTAT (% lower income).

--- Detailed Viva Q&A ---

Q1: What is the main difference between Linear Regression and Logistic Regression?
A1: Linear Regression predicts a continuous numeric output (e.g., house price = $245,000). Logistic Regression predicts a discrete class label (e.g., 1 or 0, Spam or Not Spam). The math starts similarly but Logistic Regression wraps the result in a sigmoid function.

Q2: Why do we perform a `train_test_split`?
A2: To prevent "Overfitting" — if we train on 100% of data, the model memorizes exact answers. By hiding 20% (Test Set), we force the model to prove it learned the PATTERN, not just memorized the training data.

Q3: What does the "Line of Best Fit" actually mean?
A3: It is the single straight line through the data points that minimizes the total squared distance between the line itself and every actual data point. Mathematically, it minimizes the sum of squared residuals (Ordinary Least Squares method).

Q4: If the R-squared score is 0.85, what does that mean?
A4: 85% of the variation in house prices can be mathematically explained by our 13 input features. The remaining 15% is unexplained "noise" (random variation the model cannot capture).

Q5: In the code, what do `model.fit()` and `model.predict()` do?
A5: `fit()` is the training phase — the algorithm studies training data to learn the optimal coefficients (weights) for each of the 13 features. `predict()` is the exam phase — the trained model applies those learned weights to new, unseen test data to generate price predictions.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What is a "Coefficient" in Linear Regression?
   A1: The weight or "importance score" assigned to each input feature. If 'RM' (rooms) coefficient is +5.0, it means adding one extra room increases the predicted price by $5,000. A negative coefficient means the feature DECREASES the price.

   Q2: Why might the MSE be a very large number?
   A2: Because MSE squares the errors. If a single house is predicted $10,000 off the real price, that one error contributes 10,000^2 = 100,000,000 to the total MSE. Large individual errors are amplified dramatically by squaring.

   Q3: Can the R2 score ever be negative?
   A3: Yes! If the model is so bad that it performs worse than simply predicting the mean value for every house, R2 becomes negative. This means the model has literally learned nothing useful from the features.

------------------------- End of Viva Notes -------------------------
"""
