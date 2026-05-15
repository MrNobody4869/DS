# Practical No. 6: Data Analytics III
# 1. Implement Simple Naïve Bayes classification algorithm using Python on iris dataset.
# 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall.

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# --- Step 1: Load the Dataset ---

# Option A: Using the built-in sklearn dataset (no file needed)
iris = load_iris()
X = iris.data
y = iris.target

# Option B: Load from local Iris.csv if provided in the lab
# IMPORTANT: The provided Iris.csv has no proper header row (first row contains metadata).
# Use header=None and assign column names manually.
# Uncomment the 5 lines below and comment out Option A above to switch:
# df = pd.read_csv("Iris.csv", header=None,
#                  names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
# X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
# from sklearn.preprocessing import LabelEncoder
# y = LabelEncoder().fit_transform(df['Species'].values)
# print("[INFO] Loaded local Iris.csv — columns: SepalLength, SepalWidth, PetalLength, PetalWidth, Species")

# Option C: Fetch from personal GitHub repo (use if no local file and sklearn fails)
# Uncomment the 5 lines below and comment out Option A above to switch:
# df = pd.read_csv("https://raw.githubusercontent.com/MrNobody4869/datasets/main/Iris.csv")
# X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
# from sklearn.preprocessing import LabelEncoder
# y = LabelEncoder().fit_transform(df['Species'].values)
# print("[INFO] Fetched Iris.csv from GitHub — columns: SepalLength, SepalWidth, PetalLength, PetalWidth, Species")

# --- Step 2: Split the Data ---
# 75% for training, 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Step 3: Train the Naïve Bayes Model ---
# We use Gaussian Naive Bayes because the Iris features are continuous (length/width in cm)
# and we assume they follow a Gaussian (Normal) distribution.
model = GaussianNB()  # GaussianNB(): Naïve Bayes variant that assumes continuous features follow a Gaussian (Bell Curve) distribution.
model.fit(X_train, y_train)

# --- Step 4: Make Predictions ---
y_pred = model.predict(X_test)

# --- Step 5: Confusion Matrix (Multi-class 3x3) ---
print("--- Model Evaluation ---")

# Iris has 3 classes → confusion_matrix() produces a 3x3 grid, not a 2x2.
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Multi-class 3x3):\n", cm)

# --- Step 6: One-vs-Rest Binary Conversion ---
# TP/FP/TN/FN are binary concepts. For multi-class, we use One-vs-Rest:
# We treat Class 0 (Iris-Setosa) as POSITIVE, all others (Versicolor, Virginica) as NEGATIVE.
# If the sample is Class 0 → binary label = 1 (Positive)
# If the sample is any other class → binary label = 0 (Negative)
y_test_binary = [1 if i == 0 else 0 for i in y_test]
y_pred_binary = [1 if i == 0 else 0 for i in y_pred]

cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
print("\nBinary Confusion Matrix (Class 0 = Setosa as Positive):")
print(cm_binary)

# Extract the 4 values from the binary matrix
TN = cm_binary[0][0]   # Predicted NOT Setosa, Actually NOT Setosa — Correct
FP = cm_binary[0][1]   # Predicted Setosa, Actually NOT Setosa — Wrong (Type 1 Error)
FN = cm_binary[1][0]   # Predicted NOT Setosa, Actually Setosa — Wrong (Type 2 Error)
TP = cm_binary[1][1]   # Predicted Setosa, Actually Setosa — Correct

print(f"\nTrue Positives  (TP): {TP}  — Predicted Setosa, Actually Setosa")
print(f"True Negatives  (TN): {TN}  — Predicted Not Setosa, Actually Not Setosa")
print(f"False Positives (FP): {FP}  — Predicted Setosa, Actually Not Setosa (Type 1 Error)")
print(f"False Negatives (FN): {FN}  — Predicted Not Setosa, Actually Setosa (Type 2 Error)")

# --- Step 7: Performance Metrics (Calculated Manually) ---
# Manual formula calculation proves formula knowledge during viva
accuracy   = (TP + TN) / (TP + TN + FP + FN)  # Overall correct predictions / total
error_rate = (FP + FN) / (TP + TN + FP + FN)  # Overall wrong predictions / total
precision  = TP / (TP + FP)                    # Of predicted Setosa, how many were actually Setosa?
recall     = TP / (TP + FN)                    # Of all actual Setosa, how many did we correctly catch?

print(f"\nAccuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Error Rate: {error_rate:.4f}  ({error_rate*100:.2f}%)")
print(f"Precision:  {precision:.4f}  ({precision*100:.2f}%)")
print(f"Recall:     {recall:.4f}  ({recall*100:.2f}%)")


















"""
--- Code Explanation ---
1. load_iris(): Built-in sklearn function that loads the Iris dataset (150 samples, 4 features, 3 classes) directly without needing a CSV file.
2. GaussianNB(): Creates a Naïve Bayes classifier that assumes continuous features follow a bell-curve (Gaussian) distribution — correct for Iris since all features are physical cm measurements.
3. model.fit(): Calculates the mean and variance of each feature per class — this IS the training of a Naïve Bayes model.
4. model.predict(): Applies Bayes' Theorem to each test sample and returns the most probable class (0, 1, or 2).
5. confusion_matrix() (3x3): Since Iris has 3 classes, the matrix is 3×3 — not a simple 2×2.
6. One-vs-Rest Binary Conversion: TP/FP/TN/FN are binary concepts. We treat Class 0 (Setosa) as Positive and all others as Negative, then build a 2×2 binary confusion matrix from which the 4 values are extracted.
7. Manual Metric Formulas: Accuracy, Error Rate, Precision, and Recall are calculated directly from TP/TN/FP/FN — proving formula knowledge during the viva.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. BAYES' THEOREM — THE FOUNDATION:
   - Definition: A mathematical formula for calculating the probability of an event based on
     prior knowledge of related conditions.
   - Formula: P(A|B) = [P(B|A) × P(A)] / P(B)
     - P(A|B): Posterior — probability of A given that B has occurred.
     - P(B|A): Likelihood — probability of seeing B if A were true.
     - P(A): Prior — initial probability of A before seeing any evidence.
     - P(B): Evidence — total probability of B occurring.
   - Example in our context:
     "What is the probability this flower is Setosa, given petal length = 1.4 cm?"
     P(Setosa | petal_length=1.4) = P(petal_length=1.4 | Setosa) × P(Setosa) / P(petal_length=1.4)

2. NAÏVE BAYES ALGORITHM:
   - Definition: A classification algorithm based on Bayes' Theorem that assumes all features
     are completely independent of each other.
   - Why "Naïve"? Because the assumption of total independence is rarely true in real life.
     (Petal length and petal width of a flower ARE correlated — longer petals are also wider.)
     But despite this "naïve" assumption, the math still works remarkably well in practice.
   - Advantages:
     a) Extremely fast to train and predict (even on millions of records).
     b) Works well with small training datasets.
     c) Handles multi-class problems naturally (unlike binary-only logistic regression).
     d) Needs very little memory.
   - Disadvantages:
     a) The independence assumption is almost never perfectly true.
     b) Fails when features are highly correlated.

3. TYPES OF NAÏVE BAYES — WHEN TO USE EACH:
   a) Gaussian NB (used in this practical):
      - Use when: Features are continuous decimal numbers.
      - Assumption: Each feature follows a Normal (Bell Curve / Gaussian) distribution.
      - Example: Iris flower measurements in centimeters (5.1, 3.5, etc.).
   b) Multinomial NB:
      - Use when: Features are discrete counts or frequencies.
      - Assumption: Features represent counts of events.
      - Example: Word count in a document for text classification (spam detection).
   c) Bernoulli NB:
      - Use when: Features are binary (Yes/No, True/False, 0/1).
      - Example: Whether a specific word appears in an email (present=1, absent=0).

4. MULTI-CLASS CONFUSION MATRIX:
   - When you have 3 classes (Setosa, Versicolor, Virginica), the matrix becomes 3x3.
   - Structure of a 3x3 matrix:
     
     |              | Pred: Setosa | Pred: Versi | Pred: Virgi |
     | Act: Setosa  |  TP_setosa   |     FN      |     FN      |
     | Act: Versi   |     FP       | TP_versi    |     FN      |
     | Act: Virgi   |     FP       |     FP      | TP_virgi    |
     
   - The DIAGONAL (top-left to bottom-right): True Positives for each class.
   - OFF-DIAGONAL: All types of errors (wrong class predictions).
   - A perfect model has all values on the diagonal and zeros everywhere else.

5. EVALUATION FOR MULTI-CLASS — MACRO AVERAGING:
   - Simple Accuracy: (Total correct diagonal) / (Total all cells).
   - Macro Precision: Calculate Precision for Class 0, Class 1, Class 2 separately → average them.
   - Macro Recall: Same — calculate per class, then average.
   - Why macro? It treats all classes equally, regardless of how many samples each class has.
   - Alternative — Weighted Average: Weights each class by its sample count (better for imbalanced datasets).

6. THE IRIS DATASET:
   - 150 samples, 3 species, 50 samples per species.
   - 4 Features: SepalLength, SepalWidth, PetalLength, PetalWidth (all in cm).
   - Easiest to separate: Setosa (very small petals, easy to distinguish).
   - Hardest to separate: Versicolor and Virginica (overlapping measurements in sepal dimensions).

--- Detailed Viva Q&A ---

Q1: What is the core principle behind Naïve Bayes?
A1: It uses Bayes' Theorem to calculate the probability of a flower belonging to each species based on its measurements. It then predicts the species with the highest calculated probability. For example, it asks: "Given petal length=1.4cm and width=0.2cm, is this most likely Setosa (98%), Versicolor (1%), or Virginica (1%)?"

Q2: Why is the algorithm called "Naïve"?
A2: Because it "naively" assumes every single input feature is completely independent of the others. In reality, Iris petal length and petal width are heavily correlated (longer petals are also wider). The algorithm ignores this correlation entirely to simplify the math — and surprisingly, it still works well.

Q3: Why did we use "Gaussian" Naïve Bayes specifically?
A3: The Iris dataset uses continuous decimal measurements (e.g., 5.1cm). Gaussian NB assumes these continuous values follow a Normal (Bell Curve) distribution within each class. This is a reasonable assumption for biological measurements, which tend to cluster around a species average.

Q4: How do you find True Positives in a 3x3 multi-class confusion matrix?
A4: The diagonal of the matrix contains the True Positives. The number at position [0,0] is TP for Setosa. Position [1,1] is TP for Versicolor. Position [2,2] is TP for Virginica. All off-diagonal numbers represent misclassifications.

Q5: What does `average='macro'` mean when calculating Precision?
A5: In multi-class problems, we cannot compute a single global Precision directly. 'macro' tells sklearn to: (1) compute Precision for Setosa independently, (2) compute for Versicolor, (3) compute for Virginica, then (4) return the simple unweighted average of those 3 results — giving equal weight to all classes regardless of sample size.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: Does Naïve Bayes require Feature Scaling (like StandardScaler)?
   A1: No. Naïve Bayes is fundamentally based on calculating probabilities from data distributions, not on measuring physical distances between points (like KNN or Logistic Regression). Scaling doesn't change the underlying probability distributions, so it is unnecessary.

   Q2: Can Naïve Bayes handle multi-class classification natively?
   A2: Yes, this is one of its biggest strengths. Unlike Logistic Regression (which is inherently binary), Naïve Bayes naturally calculates a probability for EVERY class and simply picks the highest one. No special "one vs rest" trick is needed.

   Q3: What is a Classification Report in sklearn?
   A3: `classification_report()` prints a detailed table showing Precision, Recall, and F1-Score for EACH individual class, plus overall accuracy. It gives a much more complete picture of model performance than a single accuracy number.

------------------------- End of Viva Notes -------------------------
"""
