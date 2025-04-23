import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import *

from attributes import Attributes
from taslibrary import sns_helper as sns_service
from taslibrary import glue_service, s3_service, log_service
from utils import Utils
from taslibrary.ingestion_exceptions.manager import InvalidProcessTypeException
from taslibrary.tas_onetick_query_helper import TasOneTickQueryHelper
import taslibrary.tas_onetick_utility as tas_onetick_util



otq_helper = TasOneTickQueryHelper(sys_level = "eng", sys_type="glue")

def extract_allocations(spark: SparkSession, glue_helper, s3_helper, args: dict):
    logger.info("Loading and deduplicating allocation data")
    alloc_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOCATIONS_QUERY_PATH.value
    )

    logger.info(f"Fetched allocation records: {alloc_df.count()}")
    s3_helper.upload_process_logs_spdf(
        alloc_df,
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "raw_allocations"
    )

    window = Window.partitionBy("sec_id", "trade_date_est").orderBy("trade_date_est")
    alloc_dedup_df = (
        alloc_df.withColumn("row_number", row_number().over(window))
        .filter(col("row_number") == 1)
        .drop("row_number")
    )

    s3_helper.upload_process_logs_spdf(
        alloc_dedup_df,
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "dedup_allocations"
    )

    return alloc_dedup_df


def extract_securities(spark: SparkSession, s3_helper, args: dict) -> DataFrame:
    logger.info("Loading CRD security data")

    securities_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.CSM_SECURITY_UNION_QUERY_PATH.value
    )

    logger.info(f"CRD security historic: {securities_df.count()} records")
    s3_helper.upload_process_logs_spdf(
        securities_df.limit(10000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_crd_securities"
    )

    return securities_df


def join_orders_with_security(allocations_df: DataFrame, securities_df: DataFrame, s3_helper, args: dict) -> DataFrame:
    logger.info("Joining allocations with CRD securities using Spark SQL")

    allocations_df_casted = allocations_df.withColumn("sec_id_str", col("sec_id").cast("string"))
    allocations_df_casted.createOrReplaceTempView("allocations_tmp")
    securities_df.createOrReplaceTempView("crd_securities_tmp")

    allocations_securities_join_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SECURITY_JOIN_SQL.value
    )

    logger.info(f"Total records after allocations + CRD join: {allocations_securities_join_df.count()}")
    s3_helper.upload_process_logs_spdf(
        allocations_securities_join_df.limit(10000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_allocations_securities"
    )

    null_df = allocations_securities_join_df.filter(
        col("ext_sec_id").isNull() |
        col("sedol").isNull() |
        col("ticker").isNull()
    )

    s3_helper.upload_process_logs_spdf(
        null_df.limit(1000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_allocations_securities_null"
    )

    return allocations_securities_join_df


def extract_and_join_orders_srm(spark, glue_helper, s3_helper, allocations_securities_df: DataFrame, args: dict) -> DataFrame:
    logger.info("Joining with SRM_EQUITY and SRM_MF using Spark SQL from SQL file")

    allocations_securities_df = allocations_securities_df.withColumn(
        "previous_trade_date_est", date_add(col("trade_date_est"), -1)
    )

    allocations_securities_df = allocations_securities_df.withColumn(
        "previous_trade_date_est_str", date_format(col("previous_trade_date_est"), "yyyyMMdd")
    )

    allocations_securities_df.createOrReplaceTempView("alloc_securities_tmp")

    alloc_securities_srm_eq_mf_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SECURITY_SRM_JOIN_SQL.value
    )

    logger.info(f"Final SRM_EQUITY+SRM_MF join result count: {alloc_securities_srm_eq_mf_df.count()}")

    s3_helper.upload_process_logs_spdf(
        alloc_securities_srm_eq_mf_df.limit(10000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_allocations_srm_join"
    )

        # ------------------- Write to Hive/Glue Table -------------------
    output_table = Attributes.SECURITY_IDENTIFIER_OUTPUT_TABLE.value  # Define this in Attributes.py
    logger.info(f"Writing final output to table: {output_table}")

    alloc_securities_srm_eq_mf_df.write.mode("overwrite").saveAsTable(output_table)

    return alloc_securities_srm_eq_mf_df


def do_otq(otq_name, graph_name, query_params):
    logger.info(f"Running OTQ query: {otq_name} with params: {query_params}")
    otq_test_response = otq_helper.run(f'{otq_name}::{graph_name}', query_params, output_structure = "OutputStructure.symbol_result_list")
    logger.info("OTQ query completed successfully.")

    return tas_onetick_util.otq_from_list_to_df(otq_test_response)


if __name__ == "__main__":
    logger = log_service.LogService(
        app_prefix=Attributes.APP_PREFIX.value,
        job_name=Attributes.GLUE_JOB_NAME.value
    ).get_logger()

    util = Utils(logger)
    args = util.get_args_dict(sys_args_list=Attributes.SYS_ARGS_LIST.value)

    spark = util.get_spark_session()
    glue_helper = glue_service.GlueService(logger)
    s3_helper = s3_service.S3Service(logger, args["sysLevel"])
    sns_helper = sns_service.SnsHelper(args["sysLevel"])

    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PySpark Version: {spark.version}")
    logger.info(f"Job Arguments: {args}")

    logger.info("Glue Job Initialized")
    logger.info(f"Starting security trade date identifiers ETL job. Process type: {args['processType']}")

    if args['processType'] == "daily":
        pass

    elif args['processType'] == "historic":
        allocations_df = extract_allocations(spark, glue_helper, s3_helper, args)
        securities_df = extract_securities(spark, s3_helper, args)
        allocations_securities_df = join_orders_with_security(allocations_df, securities_df, s3_helper, args)
        alloc_securities_srm_srmuf = extract_and_join_orders_srm(
            spark, glue_helper, s3_helper, allocations_securities_df, args
        )
        logger.info("Security trade date identifiers job completed successfully.")
        otq_file_name = Attributes.OTQ_FILE_NAME.value
        graph_name = Attributes.OTQ_GRAPH_NAME.value
        query_params = {
        "giveFile": args["giveFile"],
        "prior_days": 5,
        "after_days": 10
        }
        otq_df = do_otq(otq_file_name, graph_name, query_params)
        logger.info(f"OTQ DataFrame: {otq_df.show(5)}")
        # ------------------- Write to S3 -------------------
        s3_helper.upload_process_logs_spdf(
            otq_df,
            args["s3TCASecIdBucket"].replace("s3://", ""),
            args["JOB_NAME"],
            "otq_results"
        )

    else:
        logger.error("Process Type not in (historic, daily) - Invalid Process Type")
        raise InvalidProcessTypeException(args['processType'])
    
    spark.stop()
    logger.info("Spark session stopped.")
