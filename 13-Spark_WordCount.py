# Practical No. 13: Write a simple program using Apache Spark framework (Word Count - PySpark).
# --- HOW TO RUN ---
# Option 1 (Google Colab / Jupyter Notebook):
#   - Run cell by cell. The !pip install pyspark cell must run first.
# Option 2 (Ubuntu Terminal):
#   - Install Java: sudo apt install openjdk-11-jdk -y
#   - Install PySpark: pip3 install pyspark
#   - Remove the !pip line below, then run: python3 13-Spark_WordCount.py
# ----------------------------
# This Python (PySpark) version is equivalent to the Scala version in 13-Spark_WordCount.scala.
# Use this file if you cannot run spark-shell directly.

# -------------------------------------------------
# Step 1: Install PySpark (Colab/Jupyter only)
# -------------------------------------------------

# Uncomment the line below if running in Colab or Jupyter:
# !pip install pyspark

# -------------------------------------------------
# Step 2: Import SparkSession
# -------------------------------------------------

# SparkSession: The single entry point to all Spark functionality in PySpark.
from pyspark.sql import SparkSession

# -------------------------------------------------
# Step 3: Create Spark Session
# -------------------------------------------------

# builder: Configures the Spark application before starting.
# appName(): Sets a human-readable name for this Spark job.
# getOrCreate(): Creates a new session, or reuses an existing one if already running.
spark = SparkSession.builder \
    .appName("WordCount") \
    .master("local[*]") \
    .getOrCreate()

# Suppress excessive INFO logs — keeps output clean
spark.sparkContext.setLogLevel("ERROR")

print("--- Spark Session Started ---")

# -------------------------------------------------
# Step 4: Input Data → RDD
# -------------------------------------------------

# Sample input data (list of strings simulating lines of a text file)
data = [
    "Hello Spark Hello Hadoop",
    "Spark is fast",
    "Hadoop and Spark"
]

# parallelize(): Converts a Python list into a distributed RDD (Resilient Distributed Dataset).
# RDD: Spark's core data structure. Data is split across nodes for parallel processing.
rdd = spark.sparkContext.parallelize(data)

print("Input Data:")
for line in data:
    print(" ", line)

# -------------------------------------------------
# Step 5: Word Count Logic (MapReduce Transformations)
# -------------------------------------------------

# Spark uses LAZY evaluation — no actual computation happens until an Action is called.

# Transformation 1: flatMap()
# flatMap(): Splits each line into individual words. Unlike map(), it FLATTENS the result
#            so all words go into a single flat list (not a list of lists).
words = rdd.flatMap(lambda line: line.split(" "))

# Transformation 2: map()
# map(): Assigns a value of 1 to each word → creates Key-Value pairs: (word, 1)
word_pairs = words.map(lambda word: (word, 1))

# Transformation 3: reduceByKey()
# reduceByKey(): Groups all pairs with the same key (word) and applies the function.
# For (word, 1) pairs: sums all the 1s to get the final count.
word_counts = word_pairs.reduceByKey(lambda a, b: a + b)

# -------------------------------------------------
# Step 6: Collect and Display Output
# -------------------------------------------------

# collect(): An ACTION that forces Spark to execute the lazy plan and returns all results
#            to the driver (your local machine). Use cautiously on large datasets.
print("\nWord Count Result:")
for word, count in word_counts.collect():
    print(f"  {word}: {count}")

# -------------------------------------------------
# Step 7: Stop Spark Session
# -------------------------------------------------

# stop(): Gracefully shuts down the Spark context and frees resources.
spark.stop()
print("\n--- Spark Session Stopped ---")





















"""
--- Code Explanation ---
1. SparkSession.builder: Configures and initializes the Spark engine. 'local[*]' means run locally using all CPU cores.
2. parallelize(): Converts a Python list into a distributed RDD — Spark's core data structure spread across the cluster.
3. flatMap(): A transformation that splits each sentence into individual words and flattens the result into a single list.
4. map(): A transformation that attaches a count of 1 to each word, creating (word, 1) Key-Value pairs.
5. reduceByKey(): An aggregation transformation that groups identical keys (words) and sums their values (counts).
6. collect(): An action that forces Spark to execute the lazy plan and return all results to the local driver.
7. spark.stop(): Terminates the Spark session and releases memory/threads.

Note on Lazy Evaluation: Spark does NOT execute transformations (flatMap, map, reduceByKey) immediately.
It only executes when an Action (collect, count, saveAsTextFile) is called. This allows Spark to
optimize the entire computation plan before running — making it much faster.

Expected Output:
  Hello: 2
  Spark: 3
  Hadoop: 2
  is: 1
  fast: 1
  and: 1

--- Ubuntu Terminal Setup (if not using Colab) ---
1. Install Java:   sudo apt install openjdk-11-jdk -y
2. Check Java:     java -version
3. Install PySpark: pip3 install pyspark
4. Run:            python3 13-Spark_WordCount.py

--- Key Concepts for Viva ---
RDD (Resilient Distributed Dataset):
   - Spark's fundamental data structure.
   - Distributed: Data is split across multiple nodes.
   - Resilient: If a node fails, Spark recreates the lost partition from the original data.
   - Immutable: Cannot be changed — transformations create new RDDs.

Transformations vs Actions:
   - Transformations (flatMap, map, reduceByKey): Define WHAT to do. Lazy — not executed immediately.
   - Actions (collect, count, saveAsTextFile): Trigger actual execution. Forces Spark to compute everything.

MapReduce in Spark:
   - Map phase: map() → each word becomes (word, 1)
   - Reduce phase: reduceByKey() → (word, [1,1,1]) → (word, 3)

PySpark vs Scala Spark:
   - PySpark: Python API for Spark. Easier to use, Colab-compatible.
   - Scala Spark: Native Spark language. Faster, but requires spark-shell or sbt.
   - Both implement the same RDD operations and produce identical results.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS APACHE SPARK?
   - Open-source, distributed computing framework for processing Big Data.
   - Up to 100x faster than Hadoop MapReduce because it uses in-memory processing.
   - Supports batch, real-time (streaming), ML, and graph processing.

2. RDD (RESILIENT DISTRIBUTED DATASET):
   - Core data structure in Spark.
   - Created by: parallelize() (from local data) or textFile() (from files/HDFS).
   - Properties: Immutable, Distributed, Fault-tolerant (Resilient).

3. SPARK vs HADOOP MAPREDUCE:
   - Hadoop MapReduce: Writes intermediate results to disk after each step → very slow.
   - Spark: Keeps intermediate results IN MEMORY → 100x faster for iterative algorithms.

4. WORD COUNT — THE BIG DATA "HELLO WORLD":
   - Input: List of sentences.
   - Step 1 (Map): Split into words → assign (word, 1).
   - Step 2 (Reduce): Group by word → sum counts.
   - Output: (word, total_count) for every unique word.

------------------------- End of Theory Notes -------------------------
"""
