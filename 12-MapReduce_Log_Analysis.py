# Practical No. 12: Design a distributed application using MapReduce to process a log file.
# --- HOW TO RUN ON UBUNTU ---
# 1. Install mrjob: pip3 install mrjob
# 2. Create a dummy log file: echo '" 200 ' > server_logs.txt
# 3. Run the script: python3 12-MapReduce_Log_Analysis.py server_logs.txt
# ----------------------------
# We will use the 'mrjob' library to write a MapReduce job in a single Python file.
# To run this:
# 1. pip install mrjob
# 2. Save some sample log text into a file called 'server_logs.txt'
# 3. Run in terminal: python 12-MapReduce_Log_Analysis.py server_logs.txt

from mrjob.job import MRJob
from mrjob.step import MRStep
import re

# Regex pattern to find the HTTP Status Code in a standard Apache/System Log
# Example log line: "192.168.1.1 - - [20/May/2026] "GET /index.html HTTP/1.1" 200 453"
# The status code is the "200" (or 404, 500, etc.)
STATUS_CODE_PATTERN = re.compile(r'"\s(\d{3})\s')

class LogStatusCount(MRJob):  # MRJob: Base class that simulates Hadoop MapReduce locally and enables direct deployment to AWS EMR.
    """
    A MapReduce job that counts the frequency of HTTP Status Codes (e.g., 200 OK, 404 Not Found)
    in a massive system log file.
    """

    def steps(self):
        # We define the sequence of Map and Reduce phases
        return [
            MRStep(mapper=self.mapper_extract_status,
                   reducer=self.reducer_count_status)
        ]

    # --- THE MAPPER PHASE ---
    def mapper_extract_status(self, _, line):
        """
        The Mapper reads the log file line by line.
        It extracts the Status Code and "emits" a Key-Value pair.
        Example: If it finds a 404 error, it emits -> (404, 1)
        """
        match = STATUS_CODE_PATTERN.search(line)
        if match:
            status_code = match.group(1)
            # Emit Key: Status Code (e.g., '404'), Value: 1
            yield (status_code, 1)  # yield: Generates a Key-Value pair stream without terminating the function (like a generator).

    # --- THE REDUCER PHASE ---
    def reducer_count_status(self, status_code, counts):
        """
        Hadoop automatically gathers all identical keys from the mappers.
        The Reducer receives the key and a LIST of all its values.
        Example: It receives -> (404, [1, 1, 1, 1, 1])
        It sums them up and emits the final count -> (404, 5)
        """
        total_count = sum(counts)
        yield (status_code, total_count)


if __name__ == '__main__':
    LogStatusCount.run()




























"""
--- Code Explanation ---
1. MRJob: The base class from the `mrjob` library that wraps Hadoop MapReduce. It allows this code to run locally on your laptop, or scaled up to 1,000 servers on a real Hadoop cluster without changing the code.
2. MRStep: Defines the pipeline (Data goes to Mapper -> then goes to Reducer).
3. mapper_extract_status(self, _, line): The mapper function. It ignores the key (`_`) and reads the raw text `line`.
4. yield (status_code, 1): The Python `yield` keyword is used instead of `return` because it generates a continuous stream of Key-Value pairs without stopping the function.
5. reducer_count_status(self, key, counts): Receives grouped data from the Shuffle & Sort phase. `sum(counts)` physically adds all the 1s together to find the grand total.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS MAPREDUCE?
   - Definition: A programming model and processing framework created by Google (2004) for
     processing and generating large datasets in parallel across a distributed cluster of computers.
   - Core idea: Instead of moving massive data to a single powerful computer, you move small
     computation tasks TO the data (distributed across many machines).
   - Framework: Apache Hadoop implements the MapReduce model for open-source Big Data.
   - Analogy: Counting words in 1,000 books. Instead of one person counting all books (slow),
     you hire 1,000 people to each count one book (fast), then tally the totals.

2. THE 3 PHASES OF MAPREDUCE (Detailed):
   
   a) MAP PHASE:
      - Input: Raw data is split into fixed-size "chunks" (typically 128MB each).
      - Each chunk is sent to a separate "Mapper" worker (a different server).
      - Mapper function: Reads each line of its chunk, extracts useful information,
        and emits intermediate Key-Value pairs.
      - Example (Word Count): Line = "hello world hello". Mapper emits:
        ("hello", 1), ("world", 1), ("hello", 1).
      - Example (Log Analysis): Line = '192.168.1.1 "GET /" 200 453'. Mapper extracts
        the status code and emits: ("200", 1).
      - Each Mapper runs INDEPENDENTLY on different data chunks SIMULTANEOUSLY.
      - This is the "parallel" part of parallel processing.
   
   b) SHUFFLE AND SORT PHASE (Automatic — done by Hadoop):
      - After all Mappers finish, Hadoop automatically collects all Key-Value pairs.
      - It SORTS them by key and GROUPS all values with the same key together.
      - Example: All ("404", 1) pairs from 1,000 different Mappers are collected
        and sent to the SAME Reducer.
      - Result: Each Reducer receives: Key = "404", Values = [1, 1, 1, 1, 1, ...].
      - This phase ensures: ALL counts for "404" from ALL machines end up in ONE place.
      - It is like a massive mail-sorting room — routes data to the correct destination.
   
   c) REDUCE PHASE:
      - Input: A single Key with a list of all its associated values.
      - Reducer function: Aggregates/summarizes those values.
      - Example: ("404", [1,1,1,1,1]) → sum([1,1,1,1,1]) = 5 → emit ("404", 5).
      - Output: The final, compact result (e.g., a list of all status codes and counts).
      - Multiple Reducers can run in parallel on different keys.

3. WHY MAPREDUCE FOR LOG ANALYSIS?
   - Problem: A global website (e.g., Google, Amazon) may generate 50 Terabytes of logs per day.
   - Single machine approach: A Python script reading 50TB would take weeks to complete.
   - MapReduce approach:
     - Split 50TB into 50,000 chunks of 128MB each.
     - Run 50,000 Mapper instances simultaneously across a 1,000-server cluster.
     - Each server processes its 50MB chunk in minutes.
     - The total job completes in hours instead of weeks.
   - This is the power of HORIZONTAL SCALING (more servers) vs VERTICAL SCALING (bigger server).

4. KEY CONCEPTS:
   a) Key-Value Pair:
      - The universal "language" of MapReduce.
      - Every piece of data is represented as: Key → Value.
      - Example: ("404", 1) means "Status code 404 occurred 1 time in this line."
   
   b) yield keyword (Python):
      - Unlike `return` (which stops the function permanently), `yield` pauses the function,
        sends a value out, and then RESUMES right where it left off.
      - In the Mapper: `yield (status_code, 1)` continuously streams Key-Value pairs
        without stopping the Mapper function for each line.
      - This makes the Mapper very memory-efficient for processing millions of lines.
   
   c) Fault Tolerance in Hadoop:
      - Each data chunk is replicated 3 times across different servers (by default).
      - If Server #42 crashes during a Map job, the Master Node (NameNode) detects
        the failure and automatically re-assigns Server #42's chunk to another healthy server.
      - The job continues without any human intervention — it never fails.
   
   d) mrjob Library:
      - A Python library that lets you write MapReduce jobs in pure Python.
      - It simulates the entire Hadoop MapReduce pipeline locally on your laptop.
      - The same script can be deployed to a real Hadoop cluster or AWS EMR without changing code.

5. APACHE HADOOP ECOSYSTEM:
   - HDFS (Hadoop Distributed File System): Storage layer. Splits files into 128MB chunks
     and replicates them across servers.
   - MapReduce: Processing layer. Runs Map and Reduce tasks on the HDFS data.
   - YARN: Resource manager. Schedules and allocates CPU/RAM to different jobs.
   - Apache Hive: SQL-like query language on top of MapReduce.
   - Apache Spark: An evolution of MapReduce that runs 100× faster using in-memory computing.

--- Detailed Viva Q&A ---

Q1: Why is it called "MapReduce"?
A1: It is named after its two core operations: "Map" (which applies a function to all input data and transforms it into Key-Value pairs — like mapping a territory) and "Reduce" (which aggregates/summarizes all those Key-Value pairs into a compact final answer — reducing many to few).

Q2: What is a Key-Value pair? Give an example from this code.
A2: The fundamental data structure of MapReduce. A Key is the grouping identifier and Value is the associated data. In our log analysis: Key = "404" (the HTTP status code), Value = 1 (one occurrence of that code in this log line). After reduction: Key = "404", Value = 523 (total occurrences across all log files).

Q3: What happens between the Mapper and the Reducer?
A3: The "Shuffle and Sort" phase. Hadoop automatically: (1) collects all Key-Value pairs emitted by ALL Mappers running across ALL servers, (2) sorts them alphabetically by key, (3) groups all pairs with the same key together, and (4) sends each group to a single Reducer. This guarantees every occurrence of "404" from every server ends up at one Reducer.

Q4: What does the `yield` keyword do in Python?
A4: `yield` is a "lazy return" — it sends a value out of the function but PAUSES instead of terminating the function. The function resumes when called again. This creates a "generator" — it processes millions of log lines one-by-one without loading all of them into RAM simultaneously, making it extremely memory-efficient.

Q5: Why is MapReduce highly fault-tolerant?
A5: Because Hadoop replicates every data chunk 3× across different physical machines. The Master Node (NameNode) continuously monitors all worker nodes. If a worker crashes mid-job, the NameNode automatically detects the failure and re-assigns that failed node's data chunk to another healthy node. The job simply continues — no data is lost, no human intervention required.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: In the reducer: `def reducer(self, status_code, counts)`, what data type is `counts`?
   A1: `counts` is a Python Generator (an iterator) — not a regular list. It produces values one by one from memory without storing them all at once. `sum(counts)` iterates through it, adding each 1 together to produce the final total. This handles billions of log entries without running out of RAM.

   Q2: Can we run this code without Hadoop installed?
   A2: Yes! `mrjob` simulates the entire Hadoop pipeline locally. It runs the Mapper and Reducer as local Python processes, uses your local file system instead of HDFS, and performs the Shuffle/Sort in memory. The exact same script works on a real Hadoop cluster or AWS EMR without modifying a single line.

   Q3: What is the HTTP Status Code "200" vs "404" vs "500"?
   A3: HTTP Status Codes are 3-digit response codes from web servers.
   200 = OK (request successful). 
   404 = Not Found (the URL doesn't exist on the server).
   500 = Internal Server Error (the server crashed while processing the request).
   Log analysis counts these codes to quickly identify broken links (404s) and server crashes (500s).

------------------------- End of Viva Notes -------------------------
"""
