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


def join_allocations_with_security(
    alloc_df: DataFrame,
    sec_df: DataFrame,
    s3_helper,
    args: dict
) -> DataFrame:
    """
    Joins allocations with CSM security on sec_id and trade_date_est.
    Saves both null-ext_sec_id filtered records and full join result to S3.

    :return: Full joined DataFrame (with ext_sec_id both null and not null)
    """
    logger.info("Joining allocations with cleaned CSM security")

    # Cast sec_id for join
    alloc_df_casted = alloc_df.withColumn("sec_id_str", col("sec_id").cast("string")).alias("a")
    sec_df = sec_df.alias("c")

    # Perform the left join
    full_joined_df = alloc_df_casted.join(
        sec_df,
        (col("a.sec_id_str") == col("c.sec_id")) &
        (col("a.trade_date_est") == col("c.datadate")),
        how="left"
    )

    logger.info(f"Total records after join: {full_joined_df.count()}")

    # Define selected columns from both sources
    selected_cols = [
        col("a.sec_id").alias("sec_id"),
        col("c.ext_sec_id"),
        col("c.sedol"),
        col("c.cusip"),
        col("c.isin_no"),
        col("c.ticker")
    ]

    # Save full joined result to S3
    s3_helper.upload_process_logs_spdf(
        full_joined_df.select(*selected_cols),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "allocations_security_join_all"
    )

    # Filter and save ext_sec_id IS NULL subset
    null_filtered_df = full_joined_df.filter(col("c.ext_sec_id").isNull()).select(*selected_cols)

    logger.info(f"Filtered records where ext_sec_id IS NULL: {null_filtered_df.count()}")

    s3_helper.upload_process_logs_spdf(
        null_filtered_df,
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "allocations_security_join_null_only"
    )

    return full_joined_df


def load_and_join_srm(spark: SparkSession, glue_helper, s3_helper, joined_df: DataFrame, args: dict) -> DataFrame:
    """
    Joins the intermediate result with SRM data on trade_date_est (by year, month, day)
    and selects key identifiers.

    :return: Final DataFrame with Neoxam identifier columns
    """
    logger.info("Joining with SRM data")

    srm_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRM_QUERY_PATH.value,
    )

    srm_df = srm_df.withColumn("srm_year", year(col("dte_eff_ct"))) \
                   .withColumn("srm_month", month(col("dte_eff_ct"))) \
                   .withColumn("srm_day", dayofmonth(col("dte_eff_ct")))

    joined_df = joined_df.withColumn("alloc_year", year(col("trade_date_est"))) \
                         .withColumn("alloc_month", month(col("trade_date_est"))) \
                         .withColumn("alloc_day", dayofmonth(col("trade_date_est")))

    final_df = joined_df.join(
        srm_df,
        (joined_df["alloc_year"] == srm_df["srm_year"]) &
        (joined_df["alloc_month"] == srm_df["srm_month"]) &
        (joined_df["alloc_day"] == srm_df["srm_day"]),
        how="left"
    ).select(
        col("srm.vanguard_id_undr").alias("neoxam_underlying_vanguard_id"),
        col("srm_dom.sedol_id").alias("neoxam_underlying_trade_date_sedol"),
        col("srm_dom.ticker").alias("neoxam_underlying_trade_date_ticker"),
        col("srm_dom.isin_id").alias("neoxam_underlying_trade_date_isin")
    )

    logger.info(f"Final SRM-enriched records: {final_df.count()}")
    s3_helper.upload_process_logs_spdf(final_df, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "final_output_neoxam_srm_only")
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
