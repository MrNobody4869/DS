# Practical No. 10: Data Visualization III
# --- HOW TO RUN ON UBUNTU ---
# 1. Open Terminal and ensure libraries are installed: pip3 install pandas seaborn matplotlib scikit-learn
# 2. Run the script: python3 10-Data_Visualization_III.py
# ----------------------------
# 1. List features and types.
# 2. Create a histogram for each feature.
# 3. Create a boxplot for each feature.
# 4. Compare distributions and identify outliers.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# --- Step 1: Load the Dataset ---

print("--- Loading Iris Dataset ---")
# Option A: Using Scikit-Learn's built in dataset (no file needed)
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Adding the target species column and mapping it to actual names
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)  # from_codes(): Converts raw numeric targets (0,1,2) into readable string category names.

# Option B: Load from local Iris.csv if provided in the lab
# IMPORTANT: The provided Iris.csv has no proper header (first row is metadata, not column names).
# Use header=None and assign column names manually.
# Uncomment the 4 lines below and comment out Option A above to switch:
# df = pd.read_csv("Iris.csv", header=None,
#                  names=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','species'])
# df = df.iloc[1:].reset_index(drop=True)  # Drop the metadata row
# print("[INFO] Loaded local Iris.csv — columns assigned manually")

# Option C: Fetch from personal GitHub repo (use if no local file and sklearn fails)
# Uncomment the 4 lines below and comment out Option A above to switch:
# df = pd.read_csv("https://raw.githubusercontent.com/MrNobody4869/datasets/main/Iris.csv")
# df.columns = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','species']
# df = df.iloc[1:].reset_index(drop=True)  # Drop metadata row if present
# print("[INFO] Fetched Iris.csv from GitHub — columns assigned manually")

# --- Task 1: List Features and Their Types ---

print("\n--- Task 1: Features and their Types ---")
# df.dtypes will list the data type of every column
print(df.dtypes)

print("\nInference for Task 1:")
print("- 'sepal length (cm)': Numeric (Float)")
print("- 'sepal width (cm)': Numeric (Float)")
print("- 'petal length (cm)': Numeric (Float)")
print("- 'petal width (cm)': Numeric (Float)")
print("- 'species': Nominal / Categorical (String)")


# --- Task 2: Create a Histogram for Each Feature ---

print("\n--- Task 2: Generating Histograms ---")

# We loop through the 4 numeric columns and plot them
features = iris.feature_names

# Setting up a 2x2 grid for subplots so all 4 histograms show in one window
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # subplots(2,2): Creates a 2x2 grid of 4 empty plot slots in a single window.
fig.suptitle('Histograms of Iris Features (Distributions)', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()  # flatten(): Converts the 2D grid array into a 1D list for easy iteration in a for loop.

for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')  # kde=True: Overlays a smooth Kernel Density Estimate curve on top of the histogram bars.
    axes[i].set_title(f'Histogram of {col}')

plt.tight_layout()
plt.show()


# --- Task 3: Create a Boxplot for Each Feature ---

print("\n--- Task 3: Generating Boxplots ---")

# Setting up a 2x2 grid for subplots for boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Boxplots of Iris Features (Outlier Detection)', fontsize=16)

axes = axes.flatten()

for i, col in enumerate(features):
    # We plot the boxplot. We also split by 'species' on the x-axis to get deeper insights
    sns.boxplot(x=df['species'], y=df[col], ax=axes[i], palette='Set2')
    axes[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# --- Task 4: Compare Distributions and Identify Outliers ---

print("\n--- Task 4: Inferences on Distributions and Outliers ---")

print("1. Distribution Comparison (From Histograms):")
print("   - Sepal Length & Width: These appear to have a roughly 'Normal Distribution' (Bell curve), with values centered around their means.")
print("   - Petal Length & Width: These have a 'Bimodal Distribution' (two peaks). This strongly suggests that there are distinct groups (species) within the data, where one species has very small petals and the other two have larger petals.")

print("\n2. Outlier Identification (From Boxplots):")
print("   - Looking at the dots outside the whiskers in the Boxplots:")
print("   - 'Sepal Width' has outliers, specifically for the 'Iris-virginica' species (some flowers are unusually wide or narrow).")
print("   - 'Petal Length' has a few lower outliers for 'Iris-setosa' and 'Iris-versicolor'.")
print("   - 'Petal Width' generally does not show extreme outliers.")



























"""
--- Code Explanation ---
1. pd.Categorical.from_codes: Converts numeric targets (0, 1, 2) back into their string names (setosa, versicolor, virginica) automatically.
2. plt.subplots(2, 2): Creates a 2x2 grid (4 empty slots) so we can plot all 4 features in a single window instead of popping up 4 separate windows.
3. axes.flatten(): Converts the 2D grid of plots into a 1D list, allowing us to easily loop over them using `for i, col in enumerate(features)`.
4. ax=axes[i]: Tells Seaborn specifically which of the 4 grid slots to draw the current plot into.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. TYPES OF DATA:
   a) Numeric (Quantitative) Data:
      - Definition: Data that represents measurable quantities.
      - Operations: Can add, subtract, multiply, divide, and compute averages.
      - Two sub-types:
        - Continuous: Any value within a range (e.g., height = 5.4 cm, weight = 62.3 kg).
        - Discrete: Only whole number counts (e.g., number of petals = 3, number of rooms = 4).
      - Visualized with: Histograms, Box Plots, Scatter Plots.
   
   b) Categorical (Nominal/Ordinal) Data:
      - Definition: Data that represents labels, names, or groups.
      - Nominal: No natural order (e.g., Species = "setosa", Color = "blue"). Cannot be sorted.
      - Ordinal: Has a natural order (e.g., Low < Medium < High, Rating 1-5).
      - Operations: Cannot add or average. Can count (frequency) and find mode.
      - Visualized with: Bar Charts, Count Plots, Pie Charts.

2. SHAPES OF DISTRIBUTIONS (From Histograms):
   The shape of a histogram tells us about the underlying nature of the data.
   
   a) Normal Distribution (Bell Curve):
      - A perfectly symmetric, single-peaked curve.
      - Mean = Median = Mode (all three are identical and at the center).
      - Examples: Human heights, IQ scores, measurement errors.
      - Ideal for most ML models.
   
   b) Bimodal Distribution (Two peaks):
      - A histogram with TWO distinct humps/peaks.
      - Almost always indicates two separate sub-populations are mixed in the dataset.
      - Example in Iris: Petal Length histogram has two peaks because Setosa has tiny
        petals (first peak at ~1.5cm) and the other two species have larger petals
        (second peak at ~4-5cm).
      - Implication: If you see bimodal data, consider splitting by category first.
   
   c) Right-Skewed Distribution (Positive Skew):
      - Long tail extends to the RIGHT. Most values are low, a few are very high.
      - Mean > Median > Mode.
      - Examples: Income data, House prices, Website traffic.
   
   d) Left-Skewed Distribution (Negative Skew):
      - Long tail extends to the LEFT. Most values are high, a few are very low.
      - Mean < Median < Mode.
      - Examples: Age of retirement, Test scores where most students scored high.
   
   e) Uniform Distribution:
      - All bins have roughly equal height. No dominant value.
      - Example: Rolling a fair dice (each number equally likely).

3. OUTLIER DETECTION (From Boxplots):
   - Boxplots provide an automatic, mathematically-defined outlier detection system.
   - The IQR Rule:
     - Lower fence = Q1 - 1.5 × IQR.
     - Upper fence = Q3 + 1.5 × IQR.
   - Any point outside these fences is automatically plotted as an isolated dot (outlier).
   - Outlier in Iris Sepal Width: Some Virginica flowers have unusually wide or narrow sepals.
   - Decision: Natural biological variation (mutation) → do NOT delete these outliers.
   - Rule: Only delete outliers if you have EVIDENCE they are measurement/recording errors.

4. SUBPLOTS AND MATPLOTLIB GRID:
   - `plt.subplots(2, 2)` creates a 2×2 grid of 4 empty plot slots.
   - Returns: `fig` (the whole figure) and `axes` (a 2D array of 4 individual plot slots).
   - `axes.flatten()`: Converts the 2D [[ax1, ax2], [ax3, ax4]] array into a 1D
     [ax1, ax2, ax3, ax4] list — making it easy to loop over with enumerate().
   - `ax=axes[i]`: Tells Seaborn exactly which slot to draw the current plot into.
   - `plt.tight_layout()`: Automatically adjusts spacing between subplots so
     titles, labels, and tick marks don't overlap each other.

5. THE IRIS DATASET — KEY VISUAL FINDINGS:
   - Sepal measurements: Overlap significantly between species. Hard to separate visually.
   - Petal measurements: Much cleaner separation. Setosa petals are dramatically smaller.
   - Bimodal histograms (Petal Length & Width): Immediately hint at species clusters.
   - Boxplot outliers: Virginica has Sepal Width outliers (some unusually wide/narrow flowers).
   - Conclusion: Petal measurements are MUCH better features for classification than sepal measurements.

--- Detailed Viva Q&A ---

Q1: What exactly did the Histograms tell us about the Petal Lengths?
A1: The Petal Length histogram was Bimodal (two distinct peaks separated by a gap). One peak around 1-2cm represents Setosa (tiny petals). The second peak around 4-7cm represents Versicolor and Virginica. This immediately reveals that the dataset contains at least two distinct flower types with very different petal sizes.

Q2: Why did we pass `x=df['species']` into the Boxplot function?
A2: Without species, it would draw one massive single box blending all 150 flowers — useless for comparison. Passing species to the X-axis draws THREE separate boxes (one per species) side-by-side. This instantly reveals that Setosa has much smaller petals than the other two, and that the Virginica outliers in Sepal Width are specific to that species.

Q3: How do you identify an outlier visually in a Boxplot?
A3: Look for individual dots (or diamonds) floating BEYOND the whisker tips. Whisker tips are the upper and lower boundaries of the "normal" range (Q1-1.5×IQR to Q3+1.5×IQR). Any point plotted beyond those boundaries is automatically flagged as an outlier by Seaborn's boxplot.

Q4: What is `plt.tight_layout()`?
A4: When creating multiple subplots in a grid, Python often lets titles overlap with neighboring plot axes, creating an unreadable mess. `tight_layout()` automatically calculates the optimal padding between subplots, subplot labels, and the figure border, making everything clean and readable with one line.

Q5: Are Outliers always bad? Should we delete the Iris dataset ones?
A5: No. An outlier is just a rare but real data point. A measurement error outlier (e.g., petal length of 900cm) should definitely be deleted. But the Iris outliers are natural biological variations — some flowers genuinely have unusually wide sepals. Deleting real biological data would make our model LESS accurate, not more.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: What does `enumerate(features)` do in the `for` loop?
   A1: `enumerate` returns TWO values simultaneously: the INDEX `i` (0, 1, 2, 3) and the ITEM `col` ('sepal length', etc.). We need the index `i` to tell Seaborn which grid slot (`axes[i]`) to draw the current plot in. Without enumerate, we'd need a separate counter variable.

   Q2: What does `pd.Categorical.from_codes()` do?
   A2: The sklearn iris dataset stores species as numeric codes: 0=setosa, 1=versicolor, 2=virginica. `from_codes()` converts those integers BACK into human-readable string names using the `target_names` list. Without this, our boxplot x-axis would just show "0", "1", "2" instead of species names.

   Q3: What is the difference between `sns.histplot()` and `sns.boxplot()`?
   A3: `histplot()` shows the FREQUENCY distribution — how many data points fall in each value range. It reveals the shape (normal, bimodal, skewed). `boxplot()` shows the STATISTICAL SUMMARY — min, Q1, median, Q3, max, and outliers. Use histplot for shape, boxplot for comparison and outlier detection.

------------------------- End of Viva Notes -------------------------
"""
