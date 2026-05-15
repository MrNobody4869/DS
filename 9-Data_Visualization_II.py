# Practical No. 9: Data Visualization II
# 1. Use the inbuilt 'titanic' dataset. Plot a box plot for distribution of age with respect 
#    to each gender along with the information about whether they survived or not.
# 2. Write observations on the inference from the above statistics.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load the Dataset ---

print("--- Loading Titanic Dataset ---")
# Option A: Seaborn built-in dataset (uses lowercase column names: 'sex', 'age', 'survived')
df = sns.load_dataset('titanic')

# Option B: Load from local train.csv if provided in the lab
# IMPORTANT: train.csv uses CAPITALIZED column names (Sex, Age, Survived).
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

# Display a preview to verify columns ('sex', 'age', 'survived')
print(df[['sex', 'age', 'survived']].head())

# --- Step 2: Plot the Box Plot ---

print("\n--- Generating Box Plot ---")

plt.figure(figsize=(10, 6))

# sns.boxplot breaks down the numeric distribution (age) across categories (sex).
# We add 'hue="survived"' to further split each gender box into two boxes (died vs survived).
sns.boxplot(data=df, x='sex', y='age', hue='survived', palette='Set3')  # boxplot(): Displays median, IQR, whiskers, and outlier dots for comparing distributions across categories.

plt.title("Distribution of Age by Gender and Survival Status")
plt.xlabel("Gender (Sex)")
plt.ylabel("Age (Years)")

# Show the plot
plt.show()

# --- Step 3: Printing the Observations/Inferences directly in the console ---
# (Also provided in the notes below)

print("\n--- Observations / Inferences from the Box Plot ---")
print("1. For Males, the median age of those who survived is slightly lower than those who died. This suggests younger males (children/boys) had a higher chance of survival.")
print("2. For Females, the median age is relatively similar regardless of survival, though the interquartile range for female survivors is slightly wider.")
print("3. There are several 'Outliers' (the dots above the boxes) in the male categories, indicating some very elderly men were on board, most of whom did not survive.")
print("4. The 'Whiskers' (lines extending from the boxes) show that the age range of survivors and non-survivors spans across almost all age groups, but the bulk of the population (the box itself) is concentrated between ages 20 and 40.")



























"""
--- Code Explanation ---
1. sns.boxplot(): The main function for creating a Box and Whisker plot.
2. x='sex': We put the categorical variable on the X-axis (Male vs Female).
3. y='age': We put the continuous numeric variable on the Y-axis (Age).
4. hue='survived': This is the magic parameter. It takes the Male box, splits it in two (Survived vs Died), and colors them differently. It does the same for the Female box.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS A BOX PLOT?
   - Also known as: "Box and Whisker Plot."
   - Definition: A standardized graphical way to display the statistical distribution of a
     dataset using its Five-Number Summary. It shows in one compact diagram where the data
     is concentrated, how spread it is, and if there are any outliers.
   - Advantage over Histogram: A histogram shows the "shape" of one variable's distribution.
     A box plot excels at COMPARING distributions of multiple groups side-by-side (e.g.,
     Male Age vs Female Age vs Male who survived vs Female who survived — all in one chart).

2. THE FIVE-NUMBER SUMMARY (Anatomy of a Box Plot):
   Understanding every part of a box plot:
   
   a) MINIMUM (End of the bottom whisker):
      - The smallest value in the dataset, EXCLUDING outliers.
      - Mathematically: The lowest value that is WITHIN 1.5×IQR of Q1.
   
   b) Q1 / First Quartile (Bottom edge of the box):
      - The value below which 25% of the data falls.
      - Also called the 25th percentile or lower quartile.
      - Example: Q1 Age = 21. Means 25% of passengers were younger than 21.
   
   c) Q2 / MEDIAN (The line inside the box):
      - The exact middle value (50th percentile). 50% of data above, 50% below.
      - IMPORTANT: This is NOT the Mean (Average). It is the Median.
      - Why Median not Mean? The Median is completely unaffected by extreme outliers.
      - Example: Median Age = 28. The "typical" Titanic passenger was 28 years old.
   
   d) Q3 / Third Quartile (Top edge of the box):
      - The value below which 75% of the data falls.
      - Also called the 75th percentile or upper quartile.
      - Example: Q3 Age = 38. 75% of passengers were younger than 38.
   
   e) MAXIMUM (End of the top whisker):
      - The largest value in the dataset, EXCLUDING outliers.
      - Mathematically: The highest value that is WITHIN 1.5×IQR of Q3.
   
   f) OUTLIER DOTS (Points beyond the whiskers):
      - Individual data points plotted as dots if they fall outside the whisker range.
      - Mathematically: Any point below Q1 - 1.5×IQR or above Q3 + 1.5×IQR.
      - Example: An 80-year-old passenger would be an outlier in a mostly 20-40 aged group.

3. IQR AND WHAT THE BOX TELLS US:
   - IQR (Interquartile Range) = Q3 - Q1.
   - IQR is literally the HEIGHT of the box.
   - It represents the middle 50% of the entire population.
   - Large IQR (tall box): Data is widely spread — high variability.
   - Small IQR (short box): Data is tightly packed — low variability.
   - If the Median line is exactly centered in the box: Distribution is symmetric (normal).
   - If the Median line is close to Q1 (bottom): Distribution is right-skewed.
   - If the Median line is close to Q3 (top): Distribution is left-skewed.

4. READING THE TITANIC AGE BOXPLOT WITH hue='survived':
   - Without hue: One box for Males, one for Females (comparing genders).
   - With hue='survived': FOUR boxes — Male/Survived, Male/Died, Female/Survived, Female/Died.
   - Key insight: Surviving males had a LOWER median age than males who died.
     → Evidence that "children first" evacuation protocols were actually followed.
   - Female boxes: Females' median ages for survived vs died were closer together,
     confirming survival was more about GENDER than age for women.

--- Detailed Viva Q&A ---

Q1: Why use a Box Plot instead of a Histogram?
A1: A Histogram is excellent for seeing the shape and density of ONE variable's distribution. A Box Plot is superior when COMPARING multiple groups (e.g., Male vs Female, Survived vs Died) because it summarizes each group's entire statistical range in a compact visual, making side-by-side comparison very intuitive.

Q2: What does the line inside the box represent? Is it the Mean?
A2: No. It represents the MEDIAN (the exact middle value when sorted, 50th percentile). We use Median instead of Mean because the Median is completely unaffected by extreme outliers. If one passenger was 100 years old, the Median stays stable while the Mean would be dragged upward.

Q3: What do the dots above the top "whisker" represent?
A3: Those are outliers — statistically extreme values. Defined mathematically as points beyond Q3 + 1.5×IQR (above) or below Q1 - 1.5×IQR (below). In the Titanic Age plot, dots above the whisker represent unusually elderly passengers (e.g., 70-80 year olds) who are statistically rare in this dataset.

Q4: What specific insight did the `hue='survived'` parameter give us?
A4: Adding hue='survived' split each gender box into two boxes (survived vs died). Key revelation: surviving males had a notably LOWER median age than males who died. This statistically confirms that younger males (boys) were more likely to be evacuated — "children first" wasn't just a policy, the data proves it was actually practiced.

Q5: What does a very tall box (large IQR) indicate?
A5: A tall box indicates high variability — the data is widely spread across a large range. For example, if the Female age box is tall (IQR = 30 years), it means female passengers ranged widely in age. A short box (small IQR) means most women were in a tight age range with little variation.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: If the upper whisker is much longer than the lower whisker, what does that mean?
   A1: The data is "Right-Skewed" (positively skewed). The top 25% of data is spread over a much wider range than the bottom 25%. In Titanic ages, this would mean most passengers are in a certain age range but a tail of older passengers stretches the upper whisker far right.

   Q2: Can a Box Plot tell you if your data is normally distributed?
   A2: Yes, approximately. If (1) the median line is exactly centered in the box, AND (2) both whiskers are equal length, AND (3) there are very few outlier dots, the data is likely normally (symmetrically) distributed. Any asymmetry in these three criteria indicates skewness.

   Q3: Why does each gender show TWO boxes when we add `hue='survived'`?
   A3: Because `hue='survived'` splits the x-axis category (sex) by an additional dimension (survived = 0 or 1). Each gender now has TWO sub-groups: those who survived and those who didn't. So the chart has 4 total boxes: Male+Survived, Male+Died, Female+Survived, Female+Died.

------------------------- End of Viva Notes -------------------------
"""
