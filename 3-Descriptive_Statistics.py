# Practical No. 3: Descriptive Statistics - Measures of Central Tendency and Variability
# 1. Provide summary statistics for a numeric variable grouped by a categorical variable.
# 2. Display basic statistical details for different species in the Iris dataset.

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# ==========================================
# PART 1: Grouped Summary Statistics
# ==========================================
print("=== PART 1: Grouped Summary Statistics ===")

# Creating a simple synthetic dataset: Income grouped by Age Groups
data_part1 = {
    'Age_Group': ['Young', 'Adult', 'Senior', 'Young', 'Adult', 'Senior', 'Adult', 'Young'],
    'Income': [25000, 50000, 45000, 28000, 60000, 42000, 55000, 22000]
}
df_income = pd.DataFrame(data_part1)

print("\n--- Original Income Dataset ---")
print(df_income)

# 1. Summary statistics of income grouped by age groups
print("\n--- Summary Statistics of Income Grouped by Age Group ---")
grouped_stats = df_income.groupby('Age_Group')['Income'].agg(['mean', 'median', 'min', 'max', 'std'])  # agg(): Applies multiple statistical functions to grouped data simultaneously.
print(grouped_stats)

# 2. Create a list that contains a numeric value for each response to the categorical variable
# We can map 'Young': 1, 'Adult': 2, 'Senior': 3
category_mapping = {'Young': 1, 'Adult': 2, 'Senior': 3}
df_income['Age_Group_Numeric'] = df_income['Age_Group'].map(category_mapping)  # map(): Replaces each category label with its corresponding numeric value from the dictionary.

print("\n--- Categorical Variable Mapped to Numeric List ---")
print("Numeric Values List:", df_income['Age_Group_Numeric'].tolist())
print(df_income)


# ==========================================
# PART 2: Iris Dataset Statistics
# ==========================================
print("\n=== PART 2: Iris Dataset Statistics ===")

# Option A: Load from sklearn's built-in datasets
iris = load_iris()
# Convert it to a pandas DataFrame for easier manipulation
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the species column (Target) and map 0,1,2 to actual string names
df_iris['Species'] = iris.target
df_iris['Species'] = df_iris['Species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})

# Option B: Load from a local CSV (Uncomment if dataset is provided locally)
# df_iris = pd.read_csv("iris.csv")

# Display statistical details (percentile, mean, std, etc.) grouped by Species
print("\n--- Statistical Details of Iris Species ---")

# We loop through each unique species to print its individual statistics
species_list = df_iris['Species'].unique()

for species in species_list:
    print(f"\n--- Statistics for {species} ---")
    # Filter the dataframe for the current species
    species_data = df_iris[df_iris['Species'] == species]  # Boolean Filtering: Extracts only the rows where the Species column matches the current loop value.
    # .describe() automatically calculates percentiles, mean, std, min, and max
    print(species_data.describe())

























"""
--- Code Explanation ---
1. df.groupby('Column'): A powerful Pandas function that groups the rows of a dataset based on unique categories in a specific column (like 'Age_Group').
2. .agg(['mean', 'median', ...]): Short for 'aggregate'. It applies multiple statistical math functions to the grouped data all at once.
3. .map(): Used to replace string labels ('Young') with specific numeric values (1) based on a dictionary mapping.
4. load_iris(): Imports the classic Iris Flower dataset directly from Scikit-Learn.
5. df_iris[df_iris['Species'] == species]: This is called "Boolean Filtering." It creates a sub-table containing ONLY the rows that match the current species in the loop.
6. .describe(): A built-in Pandas function that automatically generates the mean, standard deviation, min, max, and percentiles (25%, 50%, 75%).

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS DESCRIPTIVE STATISTICS?
   - Definition: The process of summarizing, organizing, and describing the basic features of a
     dataset using numbers and charts. It describes WHAT the data looks like right now.
   - Contrast: Descriptive Statistics describes the current data. Inferential Statistics uses
     a sample to make PREDICTIONS about a larger unknown population.
   - Example: Calculating the average exam score of 30 students in a classroom is descriptive.
     Using that average to estimate the scores of ALL students in the country is inferential.
   - Tools used: Mean, Median, Mode, Standard Deviation, Percentiles, GroupBy.

2. MEASURES OF CENTRAL TENDENCY:
   These three measures tell us where the "center" or "typical value" of the data sits.
   a) MEAN (Average):
      - Formula: Sum of all values / Number of values.
      - Example: Scores = [70, 80, 90, 100]. Mean = 340/4 = 85.
      - Problem: Heavily pulled by outliers. If one student scores 150, the mean jumps.
      - Best used when: Data is symmetric and has no extreme outliers.
   b) MEDIAN (Middle Value):
      - Method: Sort all values, then pick the exact middle one.
      - Example: Sorted = [70, 80, 90, 100]. Median = (80+90)/2 = 85.
      - Advantage: Completely unaffected by outliers. If one score is 1000, median stays 85.
      - Best used when: Data is skewed or has extreme values (e.g., salary data, house prices).
   c) MODE (Most Frequent Value):
      - Example: Scores = [70, 80, 80, 80, 90]. Mode = 80 (appears 3 times).
      - Best used when: Dealing with categorical data (e.g., most popular color, most chosen city).
      - A dataset can have: No mode, one mode (unimodal), or two modes (bimodal).

3. MEASURES OF VARIABILITY (Dispersion):
   These tell us how SPREAD OUT the data is from the center.
   a) STANDARD DEVIATION (std):
      - Definition: The average distance of each data point from the Mean.
      - Low std: Data is tightly clustered around the mean (e.g., factory parts same size).
      - High std: Data is widely scattered (e.g., scores of students with very different abilities).
      - Example: Scores [85,86,87,88] → low std. Scores [10,50,90,150] → high std.
   b) VARIANCE:
      - Variance = std squared. It amplifies differences. std is preferred because it's in the
        same unit as the original data (e.g., years, dollars). Variance is in squared units.
   c) MINIMUM and MAXIMUM:
      - Simply the smallest and largest values in the dataset.
   d) PERCENTILES:
      - Definition: A value below which a given percentage of observations fall.
      - 25th Percentile (Q1): 25% of data is below this. (Lower quartile).
      - 50th Percentile (Q2): 50% of data is below this. Same as the Median.
      - 75th Percentile (Q3): 75% of data is below this. (Upper quartile).
      - Example: Scoring at the 90th percentile in an exam means you scored better than 90%
        of all students who took the exam.

4. GROUPING DATA (GroupBy):
   - Definition: Splitting a dataset into sub-groups based on a categorical column, then
     computing statistics for each group separately.
   - Why: Raw statistics on the whole dataset often hide important sub-group patterns.
   - Example: Average income for all ages = $45,000. But grouped by age:
     Young (<30): $28,000 | Adult (30-50): $55,000 | Senior (50+): $42,000.
     The grouped view reveals the hidden age-income relationship.
   - Real-world use: A/B testing (comparing two groups), Market segmentation, Medical trials.

5. THE IRIS DATASET:
   - Contains 150 flower measurements from 3 species of Iris flowers.
   - Features: SepalLength, SepalWidth, PetalLength, PetalWidth (all in cm).
   - Target: Species (Iris-setosa, Iris-versicolor, Iris-virginica).
   - Why famous: One of the most famous datasets in ML history. R.A. Fisher used it in 1936
     to demonstrate linear discriminant analysis. Still used today as a standard benchmark.

--- Detailed Viva Q&A ---

Q1: What is the main difference between Descriptive and Inferential Statistics?
A1: Descriptive statistics describe and summarize the current dataset only (e.g., the average age in this specific classroom). Inferential statistics use that sample to make predictions about a larger population (e.g., estimating the average age of all students in the country).

Q2: When should you use Median instead of Mean?
A2: Use Median when the dataset has extreme outliers. Classic example: The average wealth of 10 people in a room where one is a billionaire — the Mean is destroyed by the billionaire's wealth, but the Median still shows the true middle-class center.

Q3: What does the `groupby()` function do in Pandas?
A3: It performs a "Split-Apply-Combine" operation. It splits the dataset into groups based on a category (like Age_Group), applies a function independently to each group (like calculating the mean income), and combines the results into a clean summary table.

Q4: Explain what the 50th Percentile is.
A4: The 50th Percentile is exactly the same as the Median. It is the value below which exactly 50% of the data falls and above which 50% of the data falls. It is the exact center point of any sorted dataset.

Q5: What does a high Standard Deviation indicate about the dataset?
A5: It indicates the data points are spread far apart from the average. Example: Exam scores of [20, 50, 85, 95] have a high std — students have very different abilities. Scores of [78, 79, 80, 81] have a very low std — everyone performed similarly.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: Why did we use the `.map()` function in Part 1?
   A1: ML algorithms cannot process text labels like 'Young' or 'Senior'. We used `.map()` with a dictionary to convert those text categories into numeric codes (Young=1, Adult=2, Senior=3) that algorithms can compute with.

   Q2: What exactly does `.describe()` output in Pandas?
   A2: It produces an 8-row statistical summary for each numeric column: Count, Mean, Standard Deviation (std), Minimum, 25th Percentile (Q1), 50th Percentile (Median), 75th Percentile (Q3), and Maximum.

   Q3: What is Boolean Filtering in Pandas?
   A3: It is the technique of selecting only the rows from a DataFrame where a condition is True.
   Example: `df[df['Species'] == 'setosa']` returns only the rows where Species is setosa.
   It works like a filter — True rows pass through, False rows are blocked.

------------------------- End of Viva Notes -------------------------
"""
