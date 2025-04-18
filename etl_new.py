from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, expr, to_date, date_format, year, month, dayofmonth, date_add, coalesce, lag
from pyspark.sql.window import Window

from Attributes import Attributes
from taslibrary import sns_helper as sns_service
from taslibrary import glue_service, s3_service, logService
from utils import Utils
import sys


def load_allocations(spark: SparkSession, glue_helper, s3_helper, args: dict) -> DataFrame:
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

    df = df.dropDuplicates(["sec_id", "trade_date_est"])

    logger.info(f"Allocations cleaned and deduplicated: {df.count()} records")
    s3_helper.upload_process_logs_spdf(df.limit(1000), args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "allocations_cleaned")
    return df


def load_csm_security(spark: SparkSession, s3_helper, args: dict) -> DataFrame:
    logger.info("Loading CSM security data")

    df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.CSM_SECURITY_UNION_QUERY_PATH.value,
    ).withColumn("source", expr("'O'"))

    df_clean = df.select(
        to_date(col("datadate").cast("string"), "yyyyMMdd").alias("datadate"),
        "sec_id", "ext_sec_id", "sedol", "cusip", "isin_no", "ticker"
    )

    logger.info(f"CSM security cleaned: {df_clean.count()} records")
    s3_helper.upload_process_logs_spdf(df_clean.limit(1000), args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "csm_cleaned")
    return df_clean


def join_allocations_with_security(spark, alloc_df, sec_df, s3_helper, args):
    logger.info("Joining allocations with cleaned CSM security using Spark SQL")

    alloc_df_casted = alloc_df.withColumn("sec_id_str", col("sec_id").cast("string"))
    alloc_df_casted.createOrReplaceTempView("alloc_df")
    sec_df.createOrReplaceTempView("csm_df")

    sql_path = Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_CSM_JOIN_SQL.value
    sql_query = Utils(logger).read_sql_file(sql_path)

    joined_df = spark.sql(sql_query)

    logger.info(f"Total records after allocations + CSM join: {joined_df.count()}")

    null_df = joined_df.filter(
        col("ext_sec_id").isNull() &
        col("sedol").isNull() &
        col("cusip").isNull() &
        col("isin_no").isNull() &
        col("ticker").isNull()
    )

    s3_helper.upload_process_logs_spdf(
        null_df.limit(1000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "allocations_security_extsecid_null"
    )

    return joined_df


def load_and_join_srm(spark, glue_helper, s3_helper, joined_df: DataFrame, args: dict) -> DataFrame:
    logger.info("Joining with SRM and SRMMF using string-based joins on ext_sec_id and previous_trade_date_est")

    joined_df = joined_df.withColumn("previous_trade_date_est", date_add(col("trade_date_est"), -1))
    joined_df = joined_df.withColumn("previous_trade_date_est_str", date_format(col("previous_trade_date_est"), "yyyyMMdd"))
    joined_df.createOrReplaceTempView("alloc_csm_view")

    logger.info("NOTE: T-1 logic assumes simple 1-day back without holiday/weekend checks. To be enhanced.")

    srm_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRM_QUERY_PATH.value
    ).filter(col("file_eff_dt") >= "2021-01-01")
    srm_df.createOrReplaceTempView("srm")
    logger.info(f"SRM filtered to post-2021: {srm_df.count()} records")

    srmmf_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRMMF_QUERY_PATH.value
    ).filter(col("file_eff_dt") >= "2021-01-01")
    srmmf_df.createOrReplaceTempView("srmmf")
    logger.info(f"SRMMF filtered to post-2021: {srmmf_df.count()} records")

    sql_path = Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SRM_SRMMF_JOIN_SQL.value
    sql_query = Utils(logger).read_sql_file(sql_path)

    final_df = spark.sql(sql_query)

    logger.info(f"Final SRM+SRMMF join result count: {final_df.count()}")

    # Corporate action detection logic
    w = Window.partitionBy("sec_id").orderBy(col("previous_trade_date_est_str"))
    final_df = final_df.withColumn("prev_ext_sec_id", lag("ext_sec_id").over(w))
    final_df = final_df.withColumn("prev_isin_no", lag("isin_no").over(w))
    final_df = final_df.withColumn("prev_ticker", lag("ticker").over(w))

    final_df = final_df.withColumn(
        "corp_action_flag",
        (col("prev_ext_sec_id").isNotNull() & (col("prev_ext_sec_id") != col("ext_sec_id"))) |
        (col("prev_isin_no").isNotNull() & (col("prev_isin_no") != col("isin_no"))) |
        (col("prev_ticker").isNotNull() & (col("prev_ticker") != col("ticker")))
    )

    logger.info("Corporate action flagging complete. True = identifier change over time.")

    # Assertion check
    count_total = final_df.count()
    count_flags = final_df.filter(col("corp_action_flag") == True).count()

    assert count_total > 0, "Final joined DataFrame is empty"
    assert count_flags < count_total, "All rows are flagged as corporate actions — unexpected behavior"

    logger.info(f"Total corporate action flagged rows: {count_flags}")

    s3_helper.upload_process_logs_spdf(
        final_df.limit(10000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "final_output_neoxam_srm_srmmf"
    )

    return final_df


# -------------------- MAIN ENTRY POINT --------------------

if __name__ == "__main__":
    logger = logService.get_logger()
    logger.info("Starting security_trade_date_identifiers ETL job")
    util = Utils(logger)
    args = util.get_args_dict(sys_args_list=Attributes.SVS_ARGS_LIST.value)

    spark = util.get_spark_session()
    glue_helper = glue_service.GlueService(logger)
    s3_helper = s3_service.S3Service(logger, args["sysLevel"])

    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Spark Version: {spark.version}")
    logger.info(f"Job Arguments: {args}")

    logger.info("Glue Job Initialized")
    logger.info(f"Process type: {args['processType']}")

    if args['processType'] == "daily":
        pass

    elif args['processType'] == "historic":
        allocations_df = load_allocations(spark, glue_helper, s3_helper, args)
        logger.info(f"Allocations loaded: {allocations_df.count()} records")

        csm_df = load_csm_security(spark, s3_helper, args)
        logger.info(f"CSM security loaded: {csm_df.count()} records")

        joined_df = join_allocations_with_security(spark, allocations_df, csm_df, s3_helper, args)
        logger.info(f"Joined allocations with CSM security: {joined_df.count()} records")

        final_df = load_and_join_srm(spark, glue_helper, s3_helper, joined_df, args)
        final_df.show(truncate=False)

        logger.info("ETL job completed successfully.")

    spark.stop()
    logger.info("Spark session stopped.")
