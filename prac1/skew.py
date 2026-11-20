import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Configure JVM flags required for Java 17 compatibility

spark = (
    SparkSession.builder
    .appName("SparkSkewPractical")
    .master("local[4]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "false")           # Disable AQE to observe manual fix
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")    # Force Shuffle SortMergeJoin
   # .config("spark.driver.extraJavaOptions", java_flags)
   # .config("spark.executor.extraJavaOptions", java_flags)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setJobDescription("Setup: Generate Skewed Dataset")

number_of_records = 5_000_000
SALT_BUCKETS = 8

# 1. Create Skewed Fact DataFrame (90% assigned to key=0) and cache it
skewed_df = (
    spark.range(number_of_records)
    .withColumn(
        "key",
        when(col("id") < number_of_records * 0.90, lit(0))
        .otherwise((col("id") % 999) + 1)
    )
    .cache()
)

# Dimension table (1,000 keys)
dim_df = spark.range(1000).withColumnRenamed("id", "key").withColumn("category", lit("ACTIVE"))

print("Key distribution (Top 5):")
skewed_df.groupBy("key").count().orderBy(col("count").desc()).show(5)

# ==============================================================================
# JOB 1: UNOPTIMIZED SKEWED JOIN (Triggers Shuffle Bottleneck & Straggler Task)
# ==============================================================================
spark.sparkContext.setJobDescription("Job 1: Unoptimized Skewed Join")
t0 = time.time()

unoptimized_res = (
    skewed_df.join(dim_df, on="key", how="inner")
    .groupBy("key", "category")
    .count()
)
unoptimized_res.collect()
t_unopt = time.time() - t0
print(f"[Job 1] Unoptimized Skewed Join Duration: {t_unopt:.2f}s")

# ==============================================================================
# JOB 2: SALTED JOIN + TWO-STAGE AGGREGATION (Data Skew Fix)
# ==============================================================================
spark.sparkContext.setJobDescription("Job 2: Salted Join & Aggregation (Fixed)")
t1 = time.time()

# Step 1: Salt the skewed fact table (0 to 7)
salted_fact = skewed_df.withColumn(
    "salted_key",
    concat_ws("_", col("key"), (rand() * SALT_BUCKETS).cast("int"))
)

# Step 2: Replicate dimension table keys across all 8 salts
salted_dim = (
    dim_df.withColumn("salt", explode(array([lit(i) for i in range(SALT_BUCKETS)])))
    .withColumn("salted_key", concat_ws("_", col("key"), col("salt")))
    .drop("salt")
)

# Step 3: Evenly distributed join + two-stage aggregation
salted_res = (
    salted_fact.join(salted_dim, on="salted_key", how="inner")
    .groupBy(salted_fact["key"], "category")
    .count()
)
salted_res.collect()
t_opt = time.time() - t1
print(f"[Job 2] Salted Join Duration:{t_opt:.2f}s")

print("\n====================================")
print("Open Spark UI: http://localhost:4040")
print("Compare 'Job 1' vs 'Job 2' in the Jobs & Stages tabs.")
print("====================================")

time.sleep(120)
spark.stop()