# Practical No. 11: Create databases and tables, insert data, and run simple queries using Impala.
# Run on Jupyter Notebook, cell by cell. or else, Run on Ubuntu.
# --- HOW TO RUN ON UBUNTU ---
# 1. Open Terminal and ensure sqlite3 is available (usually pre-installed).
# 2. Run the script: python3 11-Impala_Queries.py
# 3. To run in actual impala-shell: Copy the SQL strings from this file and paste them into the shell.
# ----------------------------
# Since an actual Impala cluster requires massive Hadoop infrastructure, this script provides:
# Option A: Connecting to an actual Impala cluster (using the impyla library).
# Option B: Connecting to a local SQLite database (for immediate practice and execution).

# import impala.dbapi # Required for Option A (pip install impyla)
import sqlite3      # Required for Option B

# --- Step 1: Establish Connection ---

# Option A: Connecting to a real Apache Impala cluster
"""
conn = impala.dbapi.connect(host='localhost', port=21050)
cursor = conn.cursor()
"""

# Option B: Connecting to a local SQLite database for practice (Uncommented for execution)
# This creates a file named 'practice.db' in your folder.
conn = sqlite3.connect('practice.db')
cursor = conn.cursor()

print("--- Connected to Database Successfully ---")

# --- Step 2: Create Database and Tables ---

# Note: Impala supports 'CREATE DATABASE'. SQLite does not (it just uses the .db file).
# If using Impala, you would run: cursor.execute("CREATE DATABASE IF NOT EXISTS EmployeeDB")
# If using Impala, you would run: cursor.execute("USE EmployeeDB")

# Create a Table
create_table_query = """
CREATE TABLE IF NOT EXISTS Employees (
    EmpID INTEGER PRIMARY KEY,
    Name VARCHAR(100),
    Department VARCHAR(50),
    Salary DECIMAL(10,2)
)
"""
cursor.execute(create_table_query)
print("\n--- Table 'Employees' Created ---")

# --- Step 3: Insert Small Amounts of Data ---

# Clear existing data so we can rerun the script cleanly
cursor.execute("DELETE FROM Employees")

insert_query = """
INSERT INTO Employees (EmpID, Name, Department, Salary) 
VALUES (?, ?, ?, ?)
"""
# Note: Impala often uses %s for placeholders, while SQLite uses ?
data_to_insert = [
    (101, 'Pratik', 'Data Science', 85000.00),
    (102, 'Aditi', 'Engineering', 92000.00),
    (103, 'Rohan', 'Data Science', 78000.00),
    (104, 'Priya', 'HR', 60000.00)
]

# ExecuteMany inserts all rows at once
cursor.executemany(insert_query, data_to_insert)  # executemany(): Inserts all rows in a single batched database call instead of individual slow loops.

# COMMIT is required to save the changes to the database
conn.commit()  # commit(): Permanently saves all pending INSERT/UPDATE changes to disk. Without it, changes are lost on close.
print(f"--- Inserted {len(data_to_insert)} Rows Successfully ---")


# --- Step 4: Run Simple Queries ---

print("\n--- Query 1: View All Employees ---")
cursor.execute("SELECT * FROM Employees")
for row in cursor.fetchall():
    print(row)

print("\n--- Query 2: Filter by Department (Data Science) ---")
cursor.execute("SELECT Name, Salary FROM Employees WHERE Department = 'Data Science'")
for row in cursor.fetchall():
    print(row)

print("\n--- Query 3: Aggregate (Average Salary per Department) ---")
cursor.execute("SELECT Department, AVG(Salary) FROM Employees GROUP BY Department")
for row in cursor.fetchall():
    print(f"Department: {row[0]}, Avg Salary: {row[1]}")

# Close the connection when done
cursor.close()
conn.close()


























"""
--- Code Explanation ---
1. impala.dbapi / sqlite3: Libraries used to connect Python to a database.
2. cursor(): A cursor is an object that acts like a "pointer" or "messenger." It takes your SQL string, carries it to the database, executes it, and brings the results back to Python.
3. executemany(): A highly efficient function that inserts a massive list of records in a single batch, rather than running a loop and executing one at a time.
4. conn.commit(): Databases require a manual "Save" command after modifying data (Insert, Update, Delete) to prevent accidental destruction.
5. fetchall(): Retrieves all the rows that the database sent back in response to a SELECT query.

------------------------- Detailed Theory Notes (Short Points) -------------------------

1. WHAT IS APACHE IMPALA?
   - It is an open-source, massively parallel processing (MPP) SQL query engine.
   - It is designed specifically for Apache Hadoop.
   - Purpose: It allows you to run standard SQL queries on massive amounts of Big Data stored in HDFS (Hadoop Distributed File System) extremely fast (in real-time).

2. IMPALA VS HIVE:
   - Hive translates SQL into MapReduce jobs behind the scenes. This is very slow and meant for massive, long-running batch jobs (like generating a monthly report).
   - Impala bypasses MapReduce completely. It uses its own distributed daemons to fetch data directly. It is meant for high-speed, interactive, real-time querying.

3. DDL vs DML:
   - DDL (Data Definition Language): Commands that define the structure (e.g., CREATE TABLE, DROP TABLE).
   - DML (Data Manipulation Language): Commands that manipulate the actual data inside (e.g., INSERT, UPDATE, DELETE).

4. WHY USE IMPALA FOR DATA SCIENCE?
   - Data scientists often need to extract features from terabytes of raw logs stored in a data lake. Impala allows them to use standard, familiar SQL to query that data instantly without writing complex Java MapReduce code.

5. RUNNING ACTUAL IMPALA ON UBUNTU (Docker Approach):
   Step 1 — Install Docker:
      sudo apt update && sudo apt install docker.io -y
   Step 2 — Start Docker:
      sudo systemctl start docker
   Step 3 — Download Cloudera image:
      sudo docker pull cloudera/quickstart:latest
   Step 4 — Run the container:
      sudo docker run -it --hostname=quickstart.cloudera --privileged=true -p 8888:8888 -p 7180:7180 cloudera/quickstart:latest /usr/bin/docker-quickstart
   Step 5 — Access the container:
      sudo docker exec -it <container_id> /bin/bash
   Step 6 — Start Impala shell:
      impala-shell
   Prompt appears: [localhost:21000] >

6. IMPALA SQL COMMANDS (Inside impala-shell):
   CREATE DATABASE my_database;
   USE my_database;
   CREATE TABLE my_table (id INT, name STRING, age INT);
   INSERT INTO my_table VALUES (1, 'John', 25), (2, 'Jane', 30), (3, 'Bob', 40);
   SELECT * FROM my_table;
   SELECT * FROM my_table WHERE age > 25;
   SELECT COUNT(*) FROM my_table;
   SELECT AVG(age) FROM my_table;
   SELECT * FROM my_table ORDER BY age;
   quit;

7. KEY DIFFERENCE — SQLite vs Impala:
   - SQLite: Local file-based database. No server needed. Used here for simulation.
   - Impala: Distributed SQL engine on Hadoop. Needs a cluster. Uses port 21050.
   - Both use standard SQL syntax. Main differences: data types (STRING vs TEXT), placeholder (%s vs ?).

--- Detailed Viva Q&A ---

Q1: What is the main advantage of using Apache Impala?
A1: Real-time speed. It allows users to write standard SQL queries and get immediate results from Big Data stored in Hadoop, without suffering the massive delays of MapReduce.

Q2: What is the difference between a Connection and a Cursor in Python?
A2: The `Connection` represents the physical pipeline between your Python script and the Database server. The `Cursor` is the workspace or messenger that travels through that pipeline to execute specific SQL commands and hold the results.

Q3: Why must we run `conn.commit()` after inserting data?
A3: Databases use "Transactions." When you insert data, it is only saved temporarily in memory. This allows you to "Rollback" (undo) if a massive error occurs midway. `commit()` makes the changes permanent on the hard drive.

Q4: What does the `GROUP BY` clause do in Query 3?
A4: It groups identical data into a single summary row. Instead of showing every single employee, it collapsed all "Data Science" employees into one row so we could calculate the aggregate Average (`AVG`) salary for that specific group.

Q5: Does Impala store data?
A5: No. Impala is just a *Query Engine*. It does not store data itself; it sits on top of storage systems like HDFS or Amazon S3 and simply acts as a fast calculator to read their files.

6. PRACTICAL & CODING VIVA QUESTIONS:

   Q1: Why did we use `IF NOT EXISTS` in the CREATE TABLE statement?
   A1: If you run the script twice, the database will crash and throw an error saying "Table already exists." Adding `IF NOT EXISTS` tells the database to safely ignore the command if the table is already there.

   Q2: What is the benefit of `executemany()` over a standard `execute()` in a for-loop?
   A2: Network traffic. `executemany()` bundles all 4 rows into a single network packet and sends it to the database once. A for-loop would open and close a network request 4 separate times, causing massive lag in a real-world scenario.

------------------------- End of Viva Notes -------------------------
"""
