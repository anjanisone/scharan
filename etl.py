from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, expr, to_date, date_format, year, month, dayofmonth
from pyspark.sql.window import Window

from Attributes import Attributes
from taslibrary import sns_helper as sns_service
from taslibrary import glue_service, s3_service, logService
from utils import Utils


def extract_csv_to_df(spark: SparkSession, file_name: str, args: dict, load_type: str, schema=None) -> DataFrame:
    """
    Reads a CSV file from the S3 path configured via Attributes.CSV_FILES_PATH.

    :param spark: SparkSession object
    :param file_name: CSV file name
    :param args: Runtime arguments dictionary
    :param load_type: Descriptive name for logging
    :param schema: Optional schema to enforce
    :return: Spark DataFrame
    """
    s3_path = args["s3GlueScriptBucket"] + Attributes.CSV_FILES_PATH.value + file_name
    logger.info(f"Reading CSV file: {file_name} for {args['processType']} in {args['sysLevel']}")

    try:
        reader = spark.read.option("delimiter", ",").option("escape", "\"").option("quote", "\"")
        df = reader.csv(s3_path, header=True, schema=schema) if schema else reader.csv(s3_path, header=True)
    except Exception as e:
        logger.error(f"Error reading CSV from {s3_path}: {e}")
        raise e

    logger.info(f"CSV read complete: {df.count()} rows from {s3_path}")
    return df


def load_allocations(spark: SparkSession, glue_helper, s3_helper, args: dict) -> DataFrame:
    """
    Loads and filters allocation data from SQL source. Drops duplicates based on sec_id and trade_date_est only.

    :return: Cleaned allocation DataFrame
    """
    df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOCATIONS_QUERY_PATH.value,
    )

    df = df.filter(
        (col("sec_typ_cd").isin("ADR", "COM", "GDR", "MFC", "MFO", "PFD")) &
        (col("status") == "ACCT")
    )

    df = df.select(
        "sec_id",
        to_date(col("crdtradedate").cast("string").substr(1, 10)).alias("crdtradedate"),
        to_date(col("trade_date_local").cast("string").substr(1, 10)).alias("trade_date_local"),
        to_date(col("trade_date_est").cast("string").substr(1, 10)).alias("trade_date_est"),
        to_date(col("trade_date_utc").cast("string").substr(1, 10)).alias("trade_date_utc")
    )

    # Drop duplicates on (sec_id, trade_date_est)
    df = df.dropDuplicates(["sec_id", "trade_date_est"])

    logger.info(f"Allocations cleaned and deduplicated: {df.count()} records")
    s3_helper.upload_process_logs_spdf(df, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "allocations_cleaned")
    return df


def load_csm_security(spark: SparkSession, s3_helper, args: dict) -> DataFrame:
    """
    Loads CSM security data from SQL source and prepares it for joining.
    Fixes null datadate issue using correct format parsing.

    :return: Cleaned CSM DataFrame
    """
    logger.info("Loading CSM security data")

    df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.CSM_SECURITY_UNION_QUERY_PATH.value,
    ).withColumn("source", expr("'O'"))

    # Correctly parse datadate from int string like 20240103 to date
    df_clean = df.select(
        to_date(col("datadate").cast("string"), "yyyyMMdd").alias("datadate"),
        "sec_id", "ext_sec_id", "sedol", "cusip", "isin_no", "ticker"
    )

    logger.info(f"CSM security cleaned: {df_clean.count()} records")
    s3_helper.upload_process_logs_spdf(df_clean, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "csm_cleaned")
    return df_clean


def join_allocations_with_security(alloc_df: DataFrame, sec_df: DataFrame, s3_helper, args: dict) -> DataFrame:
    logger.info("Joining allocations with cleaned CSM security")
    alloc_df_casted = alloc_df.withColumn("sec_id_str", col("sec_id").cast("string")).alias("a")
    sec_df = sec_df.alias("c")

    full_joined_df = alloc_df_casted.join(
        sec_df,
        (col("a.sec_id_str") == col("c.sec_id")) &
        (col("a.trade_date_est") == col("c.datadate")),
        how="left"
    )

    logger.info(f"Total records after join: {full_joined_df.count()}")

    selected_df = full_joined_df.select(
        *[col("a." + c).alias("alloc_" + c) for c in alloc_df.columns],
        *[col("c." + c).alias("csm_" + c) for c in sec_df.columns]
    )

    null_df = selected_df.filter(col("csm_ext_sec_id").isNull())
    logger.info(f"ext_sec_id IS NULL records: {null_df.count()}")
    s3_helper.upload_process_logs_spdf(
        null_df, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "allocations_security_extsecid_null"
    )

    non_null_df = selected_df.filter(col("csm_ext_sec_id").isNotNull())
    logger.info(f"ext_sec_id IS NOT NULL records: {non_null_df.count()}")
    s3_helper.upload_process_logs_spdf(
        non_null_df, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "allocations_security_extsecid_notnull"
    )

    return non_null_df


def load_and_join_srm(spark: SparkSession, glue_helper, s3_helper, joined_df: DataFrame, args: dict) -> DataFrame:
    """
    Joins allocations+CSM with SRM using (file_eff_dt - 1).
    Extracts date from file_eff_dt, adjusts it, and joins with alloc_trade_date_est.
    Includes both original and adjusted dates in the output.
    """
    logger.info("Joining with SRM data using file_eff_dt - 1")

    # Load SRM
    srm_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRM_QUERY_PATH.value
    ).alias("s")

    # Adjust: file_eff_dt - 1 day
    srm_df = srm_df.withColumn("file_eff_dt_minus1", date_add(col("file_eff_dt"), -1))

    # Join with allocations+CSM using alloc_trade_date_est
    joined_df = joined_df.alias("a")

    final_df = joined_df.join(
        srm_df,
        col("a.alloc_trade_date_est") == col("s.file_eff_dt_minus1"),
        how="left"
    ).select(
        col("s.file_eff_dt"),
        col("s.file_eff_dt_minus1"),
        col("s.vanguard_id_undr").alias("neoxam_underlying_vanguard_id"),
        col("s.sedol_id").alias("neoxam_underlying_trade_date_sedol"),
        col("s.ticker").alias("neoxam_underlying_trade_date_ticker"),
        col("s.isin_id").alias("neoxam_underlying_trade_date_isin")
    )

    logger.info(f"Final SRM records after join using file_eff_dt - 1: {final_df.count()}")

    s3_helper.upload_process_logs_spdf(
        final_df,
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "final_output_neoxam_srm_minus1_day"
    )

    return final_df


# -------------------- MAIN ENTRY POINT --------------------

if __name__ == "__main__":
    logger = logService.get_logger()

    try:
        logger.info("Starting security_trade_date_identifiers ETL job")
        util = Utils(logger)
        args = util.get_args_dict(sys_args_list=Attributes.SVS_ARGS_LIST.value)

        spark = util.get_spark_session()
        glue_helper = glue_service.GlueService(logger)
        s3_helper = s3_service.S3Service(logger, args["sysLevel"])

        # Optional CSV test
        # csv_df = extract_csv_to_df(spark, "sample.csv", args, "test")
        # csv_df.show()

        # Step 1: Load and clean allocations
        allocations_df = load_allocations(spark, glue_helper, s3_helper, args)

        # Step 2: Load and clean CSM security
        csm_df = load_csm_security(spark, s3_helper, args)

        # Step 3: Join allocations with security
        joined_df = join_allocations_with_security(allocations_df, csm_df, s3_helper, args)

        # Step 4: Enrich with SRM
        final_df = load_and_join_srm(spark, glue_helper, s3_helper, joined_df, args)

        final_df.show(truncate=False)
        logger.info("ETL job completed successfully.")

    except Exception as e:
        logger.error("ETL job failed with error: " + str(e))
        raise

    finally:
        logger.info("Stopping Spark session.")
        spark.stop()
