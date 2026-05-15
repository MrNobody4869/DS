# Practical No. 8: Data Visualization I
# 1. Use the inbuilt dataset 'titanic'. Use Seaborn to find patterns in the data.
# 2. Check how the price of the ticket ('fare') is distributed by plotting a histogram.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load the Dataset ---

print("--- Loading Titanic Dataset ---")
# Option A: Seaborn built-in dataset (uses lowercase column names: 'sex', 'age', 'survived', 'fare')
df = sns.load_dataset('titanic')  # load_dataset(): Built-in Seaborn function that downloads standard datasets directly into a Pandas DataFrame.

# Option B: Load from local train.csv if provided in the lab
# IMPORTANT: train.csv uses CAPITALIZED column names (Sex, Age, Survived, Fare).
# After loading, rename them to lowercase to match this code.
# Uncomment the 3 lines below and comment out Option A above to switch:
# df = pd.read_csv("train.csv")
# df.columns = [col.lower() for col in df.columns]  # Rename: 'Sex'->'sex', 'Age'->'age', etc.
# print("[INFO] Loaded local train.csv — columns renamed to lowercase")

# Option C: Fetch from personal GitHub repo (use if no local file and seaborn fails)
# Uncomment the 3 lines below and comment out Option A above to switch:
# df = pd.read_csv("https://raw.githubusercontent.com/MrNobody4869/datasets/main/train.csv")
# df.columns = [col.lower() for col in df.columns]
# print("[INFO] Fetched train.csv from GitHub — columns renamed to lowercase")

print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# --- Step 2: Basic Dataset Info ---
print("\nDataset Info:")
df.info()

# --- Step 3: Age Distribution Histogram ---

print("\n--- Age Distribution Histogram ---")
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=10, kde=False)  # histplot(): Creates a histogram. bins=10 divides the age range into 10 buckets.
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# --- Step 4: Joint Plot (Age vs Fare) ---

# jointplot(): Shows the relationship between two numeric variables with scatter + marginal histograms.
sns.jointplot(x='age', y='fare', data=df, kind='scatter')
plt.show()

# --- Step 5: Bar Plot (Average Age by Gender) ---

print("\n--- Average Age by Gender ---")
plt.figure(figsize=(7, 5))
sns.barplot(x='sex', y='age', data=df)  # barplot(): Shows the mean value of a numeric variable for each category.
plt.title("Average Age by Gender")
plt.show()

# --- Step 6: Count Plot (Gender Count) ---

print("\n--- Count of Passengers by Gender ---")
plt.figure(figsize=(7, 5))
sns.countplot(x='sex', data=df)  # countplot(): Counts and plots the number of samples in each category.
plt.title("Count of Male and Female Passengers")
plt.show()

# --- Step 7: Box Plot (Age by Gender) ---

print("\n--- Box Plot of Age by Gender ---")
plt.figure(figsize=(7, 5))
sns.boxplot(x='sex', y='age', data=df)  # boxplot(): Displays median, IQR (box), whiskers, and outlier dots.
plt.title("Box Plot of Age by Gender")
plt.show()

# --- Step 8: Violin Plot (Age by Gender) ---

print("\n--- Violin Plot of Age by Gender ---")
plt.figure(figsize=(7, 5))
# violinplot(): Combines a box plot with a kernel density estimate — shows both summary stats and distribution shape.
sns.violinplot(x='sex', y='age', data=df)
plt.title("Violin Plot of Age by Gender")
plt.show()

# --- Step 9: Strip Plot (Age by Gender) ---

print("\n--- Strip Plot ---")
plt.figure(figsize=(7, 5))
# stripplot(): Plots each individual data point. jitter=True spreads overlapping points horizontally.
sns.stripplot(x='sex', y='age', data=df, jitter=True)
plt.title("Strip Plot of Age by Gender")
plt.show()

# --- Step 10: Finding Patterns (Survival) ---

print("\n--- Generating Pattern Visualizations ---")

# Survival rate by Gender
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='survived', hue='sex', palette='Set1')  # hue='sex': Splits each bar into two color-coded sub-bars representing Male vs Female.
plt.title("Survival Count based on Gender (0 = Died, 1 = Survived)")
plt.show()

# Survival rate by Passenger Class
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='survived', hue='class', palette='Set2')
plt.title("Survival Count based on Passenger Class")
plt.show()

# --- Step 11: Correlation Heatmap ---

plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # select_dtypes(): Filters to only numeric columns since text cannot be correlated mathematically.
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")  # annot=True: Prints the exact correlation number inside each colored cell of the heatmap.
plt.title("Correlation Heatmap of Numeric Titanic Features")
plt.show()

# --- Step 12: Fare Distribution Histogram (MAIN SYLLABUS REQUIREMENT) ---

print("\n--- Generating Fare Distribution Histogram ---")
plt.figure(figsize=(10, 6))
# bins=40: Divides the fare range into 40 buckets. kde=True: Adds a smooth density curve over the bars.
sns.histplot(df['fare'], bins=40, kde=True, color='purple')
plt.title("Distribution of Ticket Prices (Fare)")
plt.xlabel("Fare Price ($)")
plt.ylabel("Number of Passengers")
plt.show()



























"""
--- Code Explanation ---
1. sns.load_dataset('titanic'): Downloads the Titanic dataset (891 rows) directly into a Pandas DataFrame.
2. sns.histplot(): Creates a histogram — bins divide the data range into equal buckets and count samples in each.
3. sns.jointplot(): Shows the relationship between two numeric columns (age vs fare) with a scatter plot and marginal histograms on the sides.
4. sns.barplot(): Shows the MEAN value of a numeric column (age) for each category (male/female).
5. sns.countplot(): Counts and plots the number of samples in each category — useful to see imbalance (more males than females, etc.).
6. sns.boxplot(): Displays median, IQR (the box), whiskers, and individual outlier dots — shows data spread.
7. sns.violinplot(): Combines boxplot with a kernel density estimate — shows both summary statistics and the full distribution shape.
8. sns.stripplot(): Plots every individual data point. jitter=True spreads overlapping points so they're visible.
9. sns.heatmap(): Colors a grid based on correlation values. annot=True prints exact numbers. coolwarm palette: Red = strong positive, Blue = strong negative correlation.
10. kde=True in histplot: Adds a smooth Kernel Density Estimate curve over the bars to show the general distribution shape.


------------------------- Detailed Theory Notes (Short Points) -------------------------

1. DATA VISUALIZATION:
   - Definition: The graphical/visual representation of data and information to help humans
     understand patterns, trends, and outliers quickly.
   - Why essential: The human brain processes visuals 60,000× faster than raw numbers.
     A dataset of 891 rows of numbers tells you nothing at a glance. A bar chart immediately
     shows "Females survived at 3× the rate of males."
   - Goal: Convert raw numbers into actionable insights.
   - Key libraries in Python: Matplotlib (low-level control), Seaborn (high-level statistical),
     Plotly (interactive web charts).

2. SEABORN VS MATPLOTLIB:
   - Matplotlib:
     - The original Python plotting library (released 2003).
     - Extremely powerful and customizable (every pixel can be controlled).
     - Verbose: requires many lines of code for a simple styled chart.
     - Example: plt.plot(), plt.bar(), plt.hist().
   - Seaborn:
     - Built ON TOP of Matplotlib (uses it internally).
     - Designed specifically for STATISTICAL visualizations.
     - Much less code required. Automatically handles Pandas DataFrames.
     - Automatically adds beautiful color palettes, legends, and grid lines.
     - Example: sns.countplot(), sns.boxplot(), sns.heatmap().
   - Rule of thumb: Use Seaborn for statistical plots. Use Matplotlib for customization.

3. TYPES OF PLOTS — WHEN TO USE EACH:
   a) Count Plot (Bar Chart):
      - Use for: Categorical / text data.
      - What it shows: The frequency (count) of each category.
      - Example: How many passengers survived (0) vs didn't survive (1)?
      - Seaborn function: `sns.countplot()`
   
   b) Histogram:
      - Use for: Continuous / numeric data.
      - What it shows: Distribution of values by dividing the range into "bins."
      - Difference from Bar Chart: Bars are adjacent (no gaps) because data is continuous.
      - Example: How are ticket prices (Fare) distributed? Are most cheap or expensive?
      - Bins parameter: More bins = more detailed (but noisier). Fewer bins = smoother.
      - KDE curve: A smooth line drawn over the histogram showing the probability density.
      - Seaborn function: `sns.histplot(bins=40, kde=True)`
   
   c) Heatmap (Correlation Matrix):
      - Use for: Showing relationships between multiple numeric variables.
      - What it shows: A color-coded grid where each cell's color represents
        the correlation strength between two variables.
      - +1 (bright red/warm): Perfect positive correlation (both go up together).
      - -1 (bright blue/cool): Perfect negative correlation (one goes up, other goes down).
      - 0 (neutral): No relationship.
      - Seaborn function: `sns.heatmap(df.corr(), annot=True)`

4. CORRELATION (Pearson's Correlation Coefficient):
   - Definition: A statistical measure of how strongly two numeric variables move together.
   - Formula: r = Σ[(Xi - X̄)(Yi - Ȳ)] / [n × std_x × std_y]
   - Range: Always between -1.0 and +1.0.
   - Examples:
     - Titanic: Fare and Pclass are negatively correlated (r ≈ -0.55). Higher class (1st) = higher fare.
     - Age and Survived show weak negative correlation — slight tendency for younger to survive.
   - Important: Correlation ≠ Causation. Ice cream sales correlate with drowning rates
     (both are high in summer), but eating ice cream does NOT cause drowning.

5. THE TITANIC DATASET (Pattern Analysis):
   - 891 passengers, 12 columns.
   - Key survival patterns revealed by visualization:
     a) Gender: Female survival rate ≈ 74%. Male survival rate ≈ 19%.
        → "Women and Children First" policy was clearly followed.
     b) Class: 1st class survival ≈ 63%. 3rd class survival ≈ 24%.
        → Class privilege was a life-or-death factor.
     c) Fare: Right-skewed distribution — most passengers paid low fares (3rd class),
        a few paid very high fares (1st class suites).
     d) Age: Children under 10 had higher survival rates; middle-aged males had lowest.

--- Detailed Viva Q&A ---

Q1: What exactly did you conclude from the Fare Histogram?
A1: The distribution is heavily "Right-Skewed." The vast majority of passengers paid very low fares (3rd class tickets, $7-$15), while a small minority paid extreme amounts ($200-$500 for 1st class cabins). This reflects the strong economic inequality among Titanic passengers.

Q2: What did the Count Plots tell us about survival patterns?
A2: Two clear patterns emerged: (1) "Women and Children First" — females had a 3× higher survival rate than males. (2) Class privilege — 1st class passengers survived at nearly 3× the rate of 3rd class passengers, likely because they had cabins closer to the lifeboats and received priority evacuation.

Q3: What is the main difference between a Bar Chart and a Histogram?
A3: A Bar Chart plots CATEGORICAL data with gaps between bars (e.g., count of Males vs Females). A Histogram plots CONTINUOUS numerical data — bars are adjacent (no gaps) and each bar represents a range (bin) of values rather than a single category.

Q4: In `sns.histplot`, what does `kde=True` mean?
A4: KDE = Kernel Density Estimate. It draws a smooth, continuous bell-curve-like line overlaid on top of the rigid histogram bars. This curve shows the underlying probability density function of the data — making it much easier to see whether the distribution is normal, skewed, or bimodal.

Q5: Why do we use Heatmaps for Correlation analysis?
A5: A 10-variable dataset creates a 10×10 = 100-number correlation table. Reading 100 decimal numbers is impossible at a glance. A Heatmap replaces those numbers with colors (bright = strong link, dark = no link), allowing our eyes to instantly spot which features are strongly related.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: Why did we run `select_dtypes(include=['float64', 'int64'])` before the heatmap?
   A1: The `.corr()` function can only compute mathematical correlations between numeric columns. Text columns like "Name" or "Sex" have no numeric correlation. We filter to numeric-only to prevent a crash and to get a meaningful correlation matrix.

   Q2: What does the `hue` parameter do in Seaborn plots?
   A2: `hue` is a "color-splitter." Adding `hue='sex'` to a survival count plot automatically splits each survival bar into TWO sub-bars colored differently (one for Male, one for Female), revealing gender-based patterns without needing two separate charts.

   Q3: What is the difference between `sns.barplot()` and `sns.countplot()`?
   A3: `countplot()` counts the NUMBER OF ROWS in each category (frequency). `barplot()` shows the AVERAGE VALUE of a numeric variable for each category. Example: `countplot(x='survived')` shows how many survived. `barplot(x='sex', y='fare')` shows average fare paid by each gender.

------------------------- End of Viva Notes -------------------------
"""
