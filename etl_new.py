from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, expr, to_date, date_format, date_add
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

    # ------------------- CSM SECURITY ASSERTIONS -------------------

    df1 = df_clean.filter((col("sec_id") == "80931510") & (col("datadate").cast("string").isin(
        "20241114", "20241115", "20241116", "20241117", "20241118", "20241119", "20241120"))).select("datadate", "ticker")
    ticker_by_date_1 = {row["datadate"].strftime("%Y%m%d"): row["ticker"] for row in df1.collect()}
    logger.info(f"[CSM Assertion 1] sec_id=80931510 ticker by date: {ticker_by_date_1}")
    assert all(ticker_by_date_1[dt] == "SVW" for dt in ["20241114", "20241115", "20241116", "20241117"])
    assert ticker_by_date_1["20241118"] == "SGH"
    assert all(ticker_by_date_1[dt] == "SGH" for dt in ["20241119", "20241120"])

    df2 = df_clean.filter((col("sec_id") == "3817154325") & (col("datadate").cast("string").isin(
        "20250220", "20250221", "20250222", "20250223", "20250224", "20250225"))).select("datadate", "ticker")
    ticker_by_date_2 = {row["datadate"].strftime("%Y%m%d"): row["ticker"] for row in df2.collect()}
    logger.info(f"[CSM Assertion 2] sec_id=3817154325 ticker by date: {ticker_by_date_2}")
    assert all(ticker_by_date_2[dt] == "QRTEB" for dt in ["20250220", "20250221", "20250222", "20250223"])
    assert ticker_by_date_2["20250224"] == "QVCGB"
    assert ticker_by_date_2["20250225"] == "QVCGB"

    df3 = df_clean.filter((col("sec_id") == "3259235802") & (col("datadate").cast("string").isin(
        "20250226", "20250227", "20250228"))).select("datadate", "ticker")
    ticker_by_date_3 = {row["datadate"].strftime("%Y%m%d"): row["ticker"] for row in df3.collect()}
    logger.info(f"[CSM Assertion 3] sec_id=3259235802 ticker by date: {ticker_by_date_3}")
    assert all(ticker_by_date_3[dt] == "MPLN" for dt in ["20250226", "20250227"])
    assert ticker_by_date_3["20250228"] == "CTEV"

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

    srm_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRM_QUERY_PATH.value
    )
    srm_df.createOrReplaceTempView("srm")

    srmmf_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRMMF_QUERY_PATH.value
    )
    srmmf_df.createOrReplaceTempView("srmmf")

    sql_path = Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SRM_SRMMF_JOIN_SQL.value
    sql_query = Utils(logger).read_sql_file(sql_path)

    final_df = spark.sql(sql_query)

    logger.info(f"Final SRM+SRMMF join result count: {final_df.count()}")

    # ------------------- SRM ASSERTIONS -------------------

    srm_1 = srm_df.filter((col("vanguard_id") == "V22427") & (col("year") == "2024") &
                          (col("month") == "11") & (col("day").isin("14", "15", "18", "19", "20")).select("day", "ticker"))
    tickers_srm_1 = {row["day"]: row["ticker"] for row in srm_1.collect()}
    logger.info(f"[SRM Assertion 1] V22427 ticker by day: {tickers_srm_1}")
    assert tickers_srm_1["14"] == "SVW"
    for d in ["15", "18", "19", "20"]:
        assert tickers_srm_1[d] == "SGH"

    srm_2 = srm_df.filter((col("vanguard_id") == "V1031397501") & (col("year") == "2825") &
                          (col("month") == "02") & (col("day").isin("20", "21", "24", "25")).select("day", "ticker"))
    tickers_srm_2 = {row["day"]: row["ticker"] for row in srm_2.collect()}
    logger.info(f"[SRM Assertion 2] V1031397501 ticker by day: {tickers_srm_2}")
    assert tickers_srm_2["20"] == "QRTEB"
    for d in ["21", "24", "25"]:
        assert tickers_srm_2[d] == "QVCGB"

    srmmf_3 = srmmf_df.filter((col("vanguard_id") == "V1028403501") & (col("year") == "2025") &
                              (col("month") == "02") & (col("day").isin("26", "27", "28")).select("day", "ticker"))
    tickers_srmmf_3 = {row["day"]: row["ticker"] for row in srmmf_3.collect()}
    logger.info(f"[SRM Assertion 3] V1028403501 ticker by day: {tickers_srmmf_3}")
    assert tickers_srmmf_3["26"] == "MPLN"
    for d in ["27", "28"]:
        assert tickers_srmmf_3[d] == "CTEV"

    s3_helper.upload_process_logs_spdf(
        final_df.limit(1000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "final_output_neoxam_srm_srmmf"
    )

    return final_df


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
