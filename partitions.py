from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ListHivePartitions") \
    .enableHiveSupport() \
    .getOrCreate()

table_name = "crdmaster.csm_securities"
partitions_df = spark.sql(f"SHOW PARTITIONS {table_name}")
partitions_df.show(truncate=False)
