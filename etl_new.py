import sys

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import *

from attributes import Attributes
from taslibrary import glue_service, s3_service, log_service, sns_helper as sns_service
from taslibrary.ingestion_exceptions.manager import InvalidProcessTypeException
from taslibrary.tas_onetick_query_helper import TasOneTickQueryHelper
import taslibrary.tas_onetick_utility as tas_onetick_util
from utils import Utils

otq_helper = TasOneTickQueryHelper(sys_level="eng", sys_type="glue")


def extract_allocations(spark: SparkSession, glue_helper, s3_helper, args: dict) -> DataFrame:
    logger.info("Loading and deduplicating allocation data")
    alloc_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOCATIONS_QUERY_PATH.value,
    )
    logger.info(f"Fetched allocation records: {alloc_df.count()}")
    s3_helper.upload_process_logs_spdf(
        alloc_df, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "raw_allocations"
    )
    window = Window.partitionBy("sec_id", "trade_date_est").orderBy("trade_date_est")
    alloc_dedup_df = (
        alloc_df.withColumn("row_number", row_number().over(window))
        .filter(col("row_number") == 1)
        .drop("row_number")
    )
    s3_helper.upload_process_logs_spdf(
        alloc_dedup_df, args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "dedup_allocations"
    )
    return alloc_dedup_df


def extract_securities(spark: SparkSession, s3_helper, args: dict) -> DataFrame:
    logger.info("Loading CRD security data")
    securities_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.CSM_SECURITY_UNION_QUERY_PATH.value,
    )
    logger.info(f"CRD security historic: {securities_df.count()} records")
    s3_helper.upload_process_logs_spdf(
        securities_df.limit(10000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_crd_securities",
    )
    return securities_df


def join_orders_with_security(alloc_df: DataFrame, sec_df: DataFrame, s3_helper, args: dict) -> DataFrame:
    logger.info("Joining allocations with CRD securities using Spark SQL")
    alloc_df = alloc_df.withColumn("sec_id_str", col("sec_id").cast("string"))
    alloc_df.createOrReplaceTempView("allocations_tmp")
    sec_df.createOrReplaceTempView("crd_securities_tmp")

    joined_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SECURITY_JOIN_SQL.value,
    )
    logger.info(f"Total records after allocations + CRD join: {joined_df.count()}")
    s3_helper.upload_process_logs_spdf(
        joined_df.limit(10000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_allocations_securities",
    )
    null_df = joined_df.filter(
        col("ext_sec_id").isNull() | col("sedol").isNull() | col("ticker").isNull()
    )
    s3_helper.upload_process_logs_spdf(
        null_df.limit(1000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_allocations_securities_null",
    )
    return joined_df


def extract_and_join_orders_srm(spark, glue_helper, s3_helper, joined_df: DataFrame, args: dict) -> DataFrame:
    logger.info("Joining with SRM_EQUITY and SRM_MF using Spark SQL")
    joined_df = joined_df.withColumn("previous_trade_date_est", date_add(col("trade_date_est"), -1))
    joined_df = joined_df.withColumn("previous_trade_date_est_str", date_format(col("previous_trade_date_est"), "yyyyMMdd"))
    joined_df.createOrReplaceTempView("alloc_securities_tmp")

    final_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SECURITY_SRM_JOIN_SQL.value,
    )
    logger.info(f"Final SRM_EQUITY+SRM_MF join result count: {final_df.count()}")

    selected_df = final_df.select("trade_date_local", "trade_date_sedol", "sec_id")
    s3_helper.upload_process_logs_spdf(
        selected_df.limit(10000), args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "srm_selected_fields"
    )

    final_df.write.mode("overwrite").saveAsTable(Attributes.SECURITY_IDENTIFIER_OUTPUT_TABLE.value)

    enriched_df = final_df.withColumn(
        "symbol_name", when(~col("currency").isin("AUD", "THB"), col("sedol")).otherwise(col("underlying_sedol"))
    ).withColumn(
        "order_start_date", col("trade_date_local")
    ).select(
        "sec_id", "currency", "ticker", "sedol", "trade_date_local", "order_start_date", "symbol_name"
    )

    s3_helper.upload_process_logs_spdf(
        enriched_df.limit(10000), args["s3TCASecIdBucket"].replace("s3://", ""), args["JOB_NAME"], "srm_enriched_export"
    )
    return enriched_df


def do_otq(otq_name, graph_name, query_params):
    logger.info(f"Running OTQ query: {otq_name} with params: {query_params}")
    response = otq_helper.run(
        f"{otq_name}::{graph_name}",
        query_params,
        output_structure="OutputStructure.symbol_result_list"
    )
    logger.info("OTQ query completed successfully.")
    return tas_onetick_util.otq_from_list_to_df(response)


def split_and_run_otq_by_year(spark, s3_helper, args, base_df: DataFrame, otq_file: str, graph: str, base_s3_path: str):
    year_list = [row["year"] for row in base_df.select(year("trade_date_local").alias("year")).distinct().collect()]
    logger.info(f"Processing OTQ per year: {year_list}")
    for yr in year_list:
        logger.info(f"Filtering for year: {yr}")
        year_df = base_df.filter(year("trade_date_local") == yr)
        file_tag = f"srm_symbol_feed_{yr}"
        year_df.write.mode("overwrite").option("header", True).csv(f"{base_s3_path}/{file_tag}")
        csv_path = f"{base_s3_path}/{file_tag}/"
        query_params = {"giveFile": csv_path, "prior_days": 5, "after_days": 10}
        otq_df = do_otq(otq_file, graph, query_params)
        logger.info(f"OTQ results for year {yr}:")
        otq_df.show(5)


def main():
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

    logger.info("Glue Job Initialized")
    logger.info(f"Process type: {args['processType']}")

    if args["processType"] == "historic":
        allocations_df = extract_allocations(spark, glue_helper, s3_helper, args)
        securities_df = extract_securities(spark, s3_helper, args)
        joined_df = join_orders_with_security(allocations_df, securities_df, s3_helper, args)
        enriched_df = extract_and_join_orders_srm(spark, glue_helper, s3_helper, joined_df, args)

        # initial run
        query_params = {
            "giveFile": args["giveFile"],
            "prior_days": 5,
            "after_days": 10
        }
        otq_df = do_otq(Attributes.OTQ_FILE_NAME.value, Attributes.OTQ_GRAPH_NAME.value, query_params)
        logger.info("Initial OTQ Output:")
        otq_df.show(5)

        # yearly run
        base_path = f"s3://{args['s3TCASecIdBucket'].replace('s3://', '')}/{args['JOB_NAME']}"
        split_and_run_otq_by_year(
            spark, s3_helper, args,
            enriched_df,
            Attributes.OTQ_FILE_NAME.value,
            Attributes.OTQ_GRAPH_NAME.value,
            base_path
        )

    spark.stop()
    logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()
