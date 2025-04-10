from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, expr, to_date, date_format, year, month, dayofmonth, date_add, coalesce
from pyspark.sql.window import Window

from Attributes import Attributes
from taslibrary import sns_helper as sns_service
from taslibrary import glue_service, s3_service, logService
from utils import Utils


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
    s3_helper.upload_process_logs_spdf(df, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "allocations_cleaned")
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
    s3_helper.upload_process_logs_spdf(df_clean, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "csm_cleaned")
    return df_clean


def join_allocations_with_security(spark, alloc_df, sec_df, s3_helper, args):
    logger.info("Joining allocations with cleaned CSM security using Spark SQL")

    alloc_df_casted = alloc_df.withColumn("sec_id_str", col("sec_id").cast("string"))
    alloc_df_casted.createOrReplaceTempView("alloc_df")
    sec_df.createOrReplaceTempView("csm_df")

    joined_df = spark.sql("""
        SELECT 
            a.sec_id AS sec_id,
            a.crdtradedate AS crdtradedate,
            a.trade_date_local AS trade_date_local,
            a.trade_date_est AS trade_date_est,
            a.trade_date_utc AS trade_date_utc,
            a.sec_id_str AS sec_id_str,

            c.ext_sec_id AS ext_sec_id,
            c.sedol AS sedol,
            c.cusip AS cusip,
            c.isin_no AS isin_no,
            c.ticker AS ticker,
            c.datadate AS datadate
        FROM alloc_df a
        LEFT JOIN csm_df c
            ON a.sec_id_str = c.sec_id
            AND a.trade_date_est = c.datadate
    """)

    logger.info(f"Total records after allocations + CSM join: {joined_df.count()}")

    null_df = joined_df.filter(
        col("ext_sec_id").isNull() &
        col("sedol").isNull() &
        col("cusip").isNull() &
        col("isin_no").isNull() &
        col("ticker").isNull()
    )

    s3_helper.upload_process_logs_spdf(
        null_df,
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "allocations_security_extsecid_null"
    )

    return joined_df


def load_and_join_srm(spark, glue_helper, s3_helper, joined_df: DataFrame, args: dict) -> DataFrame:
    logger.info("Joining with SRM and SRMMF using previous_trade_date_est and Spark SQL")

    joined_df = joined_df.withColumn("previous_trade_date_est", date_add(col("trade_date_est"), -1))
    joined_df.createOrReplaceTempView("alloc_csm_view")

    srm_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRM_QUERY_PATH.value
    ).withColumn(
        "file_eff_dt_parsed", to_date(col("file_eff_dt").cast("string"), "yyyyMMdd")
    )
    srm_df.createOrReplaceTempView("srm")

    srmmf_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.SRMMF_QUERY_PATH.value
    ).withColumn(
        "file_eff_dt_parsed", to_date(col("file_eff_dt").cast("string"), "yyyyMMdd")
    )
    srmmf_df.createOrReplaceTempView("srmmf")

    final_df = spark.sql("""
        SELECT 
            a.*, 
            COALESCE(srm.vanguard_id_undr, srmmf.vanguard_id_undr) AS neoxam_underlying_vanguard_id,
            COALESCE(srm.sedol_id, srmmf.sedol_id) AS neoxam_underlying_trade_date_sedol,
            COALESCE(srm.ticker, srmmf.ticker) AS neoxam_underlying_trade_date_ticker,
            COALESCE(srm.isin_id, srmmf.isin_id) AS neoxam_underlying_trade_date_isin,
            srm.file_eff_dt AS srm_file_eff_dt,
            srm.file_eff_dt_parsed AS srm_file_eff_dt_parsed,
            srmmf.file_eff_dt AS srmmf_file_eff_dt,
            srmmf.file_eff_dt_parsed AS srmmf_file_eff_dt_parsed
        FROM alloc_csm_view a
        LEFT JOIN srm
            ON a.previous_trade_date_est = srm.file_eff_dt_parsed
        LEFT JOIN srmmf
            ON a.previous_trade_date_est = srmmf.file_eff_dt_parsed
    """)

    logger.info(f"Final SRM+SRMMF join result count: {final_df.count()}")

    s3_helper.upload_process_logs_spdf(
        final_df,
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "final_output_neoxam_srm_srmmf"
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

        # Step 1: Load and clean allocations
        allocations_df = load_allocations(spark, glue_helper, s3_helper, args)

        # Step 2: Load and clean CSM security
        csm_df = load_csm_security(spark, s3_helper, args)

        # Step 3: Join allocations with CSM security
        joined_df = join_allocations_with_security(spark, allocations_df, csm_df, s3_helper, args)

        # Step 4: Join with SRM/SRMMF using previous_trade_date_est
        final_df = load_and_join_srm(spark, glue_helper, s3_helper, joined_df, args)

        final_df.show(truncate=False)
        logger.info("ETL job completed successfully.")

    except Exception as e:
        logger.error("ETL job failed with error: " + str(e))
        raise

    finally:
        logger.info("Stopping Spark session.")
        spark.stop()
