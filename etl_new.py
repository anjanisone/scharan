import os
import sys, pytz, numpy as np
from io import StringIO
from datetime import datetime
import pandas as pd
import pyspark
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import *
from attributes import Attributes
from taslibrary import glue_service, s3_service, log_service, sns_helper as sns_service
from taslibrary.ingestion_exceptions.manager import InvalidProcessTypeException
from pyspark.sql.types import IntegerType
import taslibrary.tas_onetick_utility as tas_onetick_util
from utils import Utils


def extract_allocations(spark: SparkSession, glue_helper, args: dict) -> DataFrame:
    logger.info("Loading and deduplicating allocation data")
    alloc_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOCATIONS_QUERY_PATH.value,
    )
    logger.info(f"Fetched allocation records: {alloc_df.count()}")
    
    window = Window.partitionBy("sec_id", "trade_date_est", "trade_date_local").orderBy("trade_date_est")
    alloc_dedup_df = (
        alloc_df.withColumn("row_number", row_number().over(window))
        .filter(col("row_number") == 1)
        .drop("row_number")
    )
    alloc_dedup_df = (
        alloc_dedup_df
        .withColumn("sec_id_str", col("sec_id").cast("string"))
        .withColumn("DATA_ID", row_number().over(Window.orderBy(monotonically_increasing_id())))
        .withColumn("processType", lit(args["processType"]))
    )

    logger.info(f"Deduplicated allocation records: {alloc_dedup_df.count()}")
    load_reference_data_to_etl(
        alloc_dedup_df, 
        Attributes.SECURITY_IDENTIFIERS_ETL_DB_NAME.value,
        Attributes.ALLOCATIONS_DEDUP_TABLE.value,
        Attributes.SECURITY_IDENTIFIERS_ETL_LOCATION.value,
    )
    return alloc_dedup_df


def extract_securities_partitions_df(spark: SparkSession, glue_helper, args: dict) -> DataFrame:
    logger.info("Loading CRD security data")
    raw_securities_partitions_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.CSM_SECURITY_UNION_QUERY_PATH.value,
    )
    securities_partitions_df = (
        raw_securities_partitions_df
        .withColumn("partition_date", date_format(col("partition_date"), "yyyyMMdd"))
        .drop("datadate")
    )
    return securities_partitions_df


def alloc_with_securities_partitions(allocations_df: DataFrame, securities_partitions_df: DataFrame) -> DataFrame:
    logger.info("Joining allocations with CRD securities partitions")
    alloc_dedup_with_partitions = allocations_df.join(
        broadcast(securities_partitions_df),
        securities_partitions_df["partition_date"] < allocations_df["trade_date_est"],
    )

    prev_sec_partition_window = Window.partitionBy("sec_id", "trade_date_est").orderBy(col("partition_date").desc())
    alloc_dedup_sec_prev_partition = (
        alloc_dedup_with_partitions
        .withColumn("security_prev_partition_datadate", max("partition_date").over(prev_sec_partition_window))
        .drop("partition_date")
    )
    logger.info(f"Joined allocations with CRD securities partitions: {alloc_dedup_sec_prev_partition.count()}")
    alloc_dedup_sec_prev_partition.createOrReplaceTempView("allocations_securities_partitions_tmp")
    return alloc_dedup_sec_prev_partition


def extract_securities(spark: SparkSession, args: dict, data_date:str = '2025-01-01') -> DataFrame:
    logger.info("Loading CRD securities data")
    securities_df = Utils.read_sql_to_df(spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.CSM_SECURITY_UNION_QUERY_PATH.value,
        data_date=data_date
    )
    securities_df.createOrReplaceTempView("crd_securities_tmp")
    logger.info(f"Fetched CRD securities records: {securities_df.count()}")
    return securities_df


def join_allocations_with_security(spark: SparkSession, glue_helper, args:dict) -> DataFrame:
    logger.info("Joining allocations with CRD securities")
    allocations_securities_join_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SECURITY_JOIN_SQL.value,
    )
    logger.info(f"Total records after allocations + CRD join: {allocations_securities_join_df.count()}")
    allocations_securities_join_df.createOrReplaceTempView("alloc_securities_tmp")
    load_reference_data_to_etl(
        allocations_securities_join_df,
        Attributes.SECURITY_IDENTIFIERS_ETL_DB_NAME.value,
        Attributes.ALLOCATIONS_SECURITIES_TABLE.value,
        Attributes.SECURITY_IDENTIFIERS_ETL_LOCATION.value,
    )
    return allocations_securities_join_df



def extract_and_join_orders_sec_srm(spark: SparkSession, glue_helper, s3_helper, allocations_securities_df: DataFrame, args: dict) -> DataFrame:
    logger.info("Joining with SRM_EQUITY and SRM_MF using Spark SQL from SQL file")

    alloc_securities_srm_eq_mf_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SECURITY_SRM_JOIN_SQL.value
    )

    logger.info(f"Final SRM_EQUITY + SRM_MF join result count: {alloc_securities_srm_eq_mf_df.count()}")

    load_reference_data_to_etl(
        alloc_securities_srm_eq_mf_df,
        Attributes.SECURITY_IDENTIFIERS_ETL_DB_NAME.value,
        Attributes.ALLOCATIONS_SECURITIES_SRM_TABLE.value,
        Attributes.SECURITY_IDENTIFIERS_ETL_LOCATION.value,
    )

    return alloc_securities_srm_eq_mf_df



def onetick_input_df_with_sedol(alloc_securities_srm_df: DataFrame) -> DataFrame:
    onetick_sedol_input_df = alloc_securities_srm_df.withColumn(
        "SYMBOL_NAME", col("sedol")
    ).withColumn(
        "order_start_date",
        to_timestamp(concat_ws(" ", col("alloc_trade_date_local").cast("string"), lit("00:00")), "yyyy-MM-dd HH:mm")
    ).withColumn(
        "order_end_date",
        to_timestamp(concat_ws(" ", col("alloc_trade_date_local").cast("string"), lit("23:59")), "yyyy-MM-dd HH:mm")
    ).withColumn(
        "sym_date", to_date(col("alloc_trade_date_local"))
    ).select(
        "DATA_ID",
        col("alloc_sec_id").alias("sec_id"),
        col("sec_currency").alias("currency"),
        col("crd_ticker").alias("ticker"),
        col("crd_sec_sedol").alias("sedol"),
        col("alloc_trade_date_local").alias("order_start_date"),
        "order_end_date",
        "sym_date",
        "SYMBOL_NAME",
        "process_type"
    )

    load_reference_data_to_etl(
        onetick_sedol_input_df,
        Attributes.SECURITY_IDENTIFIERS_ETL_DB_NAME.value,
        Attributes.ONETICK_SEDOL_DATA_INPUT_TABLE.value,
        Attributes.SECURITY_IDENTIFIERS_ETL_LOCATION.value,
    )
    onetick_sedol_input_df = onetick_sedol_input_df.filter(col("SYMBOL_NAME").isNotNull() & (col("SYMBOL_NAME") != ""))
    onetick_sedol_input_df.printSchema()
    logger.info(f"OneTick input DataFrame count: {onetick_sedol_input_df.count()}")
    return onetick_sedol_input_df


def onetick_input_df_with_ticker(alloc_securities_srm_df: DataFrame) -> DataFrame:
    onetick_ticker_df = alloc_securities_srm_df.filter(col("crd_sec_currency").isin("THB", "AUD"))

    onetick_ticker_df = onetick_ticker_df.withColumn(
        "SYMBOL_NAME_BASE",
        when(
            (col("crd_sec_currency") == "THB") & col("crd_ticker").endsWith("/F$"),
            regexp_replace(col("crd_ticker"), "/F$", " ")
        ).when(
            (col("crd_sec_currency") == "AUD") & col("crd_ticker").rlike("(ZZ|BB|YY|XX\\*?|0A|0B|O|DA|-W)$"),
            regexp_replace(col("crd_ticker"), "(ZZ|BB|YY|XX\\*?|0A|0B|O|DA|-W)$", " ")
        ).otherwise(col("crd_ticker")))

    onetick_ticker_input_df = onetick_ticker_df.withColumn(
        "SYMBOL_NAME",
        when(
            col("crd_sec_currency") == "THB",
            concat(col("SYMBOL_NAME_BASE"), lit(" TB"))
        ).when(
            col("crd_sec_currency") == "AUD",
            concat(col("SYMBOL_NAME_BASE"), lit(" AU"))
        ).otherwise(col("SYMBOL_NAME_BASE"))
    ).withColumn(
        "order_start_date",
        to_timestamp(concat_ws(" ", col("alloc_trade_date_local").cast("string"), lit("00:00")), "yyyy-MM-dd HH:mm")
    ).withColumn(
        "order_end_date",
        to_timestamp(concat_ws(" ", col("alloc_trade_date_local").cast("string"), lit("23:59")), "yyyy-MM-dd HH:mm")
    ).withColumn(
        "sym_date", to_date(col("alloc_trade_date_local"))
    ).select(
        "DATA_ID",
        col("alloc_sec_id").alias("sec_id"),
        col("crd_sec_currency").alias("currency"),
        col("crd_ticker").alias("ticker"),
        col("crd_sec_sedol").alias("sedol"),
        col("alloc_trade_date_local").alias("order_start_date"),
        "order_end_date",
        "sym_date",
        "SYMBOL_NAME",
        "SYMBOL_NAME_BASE",
        "process_type"
    )
    load_reference_data_to_etl(
        onetick_ticker_input_df,
        Attributes.SECURITY_IDENTIFIERS_ETL_DB_NAME.value,
        Attributes.ONETICK_TICKER_DATA_INPUT_TABLE.value,
        Attributes.SECURITY_IDENTIFIERS_ETL_LOCATION.value,
    )
    onetick_ticker_input_df = onetick_ticker_input_df.filter(col("SYMBOL_NAME").isNotNull() & (col("SYMBOL_NAME") != ""))
    onetick_ticker_input_df.printSchema()
    logger.info(f"OneTick input DataFrame count: {onetick_ticker_input_df.count()}")
    return onetick_ticker_input_df


def write_otq_request_to_tmp(df:DataFrame, s3_helper, file_name_suffix:str, file_name:str, args: dict):
    s3_helper.upload_process_logs_spdf(
        df,
        args['s3TCASecIdBucket'],
        Attributes.GLUE_JOB_NAME.value,
        f"otq_request_{file_name_suffix}"+ args['processType'],
    )

    otq_input_files_key_result = s3_helper.list_files_in_folder(
        args['s3TCASecIdBucket'],
        Attributes.PROCESS_AUTOMATION_S3_PATH.value +"/"+Attributes.GLUE_JOB_NAME.value +"/logs/"
        + "otq_request_input_actual_" + file_name_suffix + args['processType']+".csv"
    )
    otq_input_files_key = otq_input_files_key_result[0]

    s3_helper.copy_file(
        args['s3TCASecIdBucket'],
        args['s3TCASecIdBucket'],
        otq_input_files_key,
        Attributes.PROCESS_AUTOMATION_S3_PATH.value +"/"+Attributes.GLUE_JOB_NAME.value +"/logs/"
        + "otq_request_input_converted_" + file_name_suffix + args['processType']+".csv/"
        + file_name,
    )

    s3_helper.delete_files_from_s3(
        args['s3TCASecIdBucket'],
        otq_input_files_key_result
    )

    otq_ipnut_file_converted_path = (
        "s3://"
        +args['s3TCASecIdBucket']
        + "/"
        + Attributes.PROCESS_AUTOMATION_S3_PATH.value
        + "/"
        + Attributes.GLUE_JOB_NAME.value
        + "/logs/otq_request_input_converted_"
        + file_name_suffix +"_"
        + args['processType']
        + ".csv"
    )
    util.copy_s3_file_to_temp_folder(
        otq_ipnut_file_converted_path,
        Attributes.OTQ_REQUEST_FILE_TEMP_PATH.value,
    )


def set_up_onetick_env(args:dict):
    tas_onetick_util.copy_ontieck_client(args['sysLevel'], "glue", path_prefix=None)
    from taslibrary.tas_onetick_query_helper import TasOneTickQueryHelper


def otq_response(spark: SparkSession, input_df: DataFrame, sym_prefix: str, file_name_prefix: str, result_prefix: str, args: dict) -> DataFrame:
    start_index = 1
    end_index = 1 + Attributes.OTQ_REQUEST_BATCH_SIZE.value
    final_response_df = None
    df_count = input_df.count()

    logger.info(f"Running OTQ request loop for prefix: {sym_prefix} - Total count: {df_count}")

    while start_index <= df_count:
        logger.info(f"Processing batch from {start_index} to {end_index}")

        # Filter the batch
        batch_df = input_df.filter(
            (col("DATA_ID") >= start_index) & (col("DATA_ID") < end_index)
        )

        # Define batch file name
        file_name = f"{file_name_prefix}_{start_index}_{end_index}.csv"

        write_otq_request_to_tmp(batch_df, s3_helper, result_prefix, file_name, args)

        # Construct the correct localfile path
        localfile_path = (
            f"s3://{args['s3TCASecIdBucket']}/"
            f"{Attributes.PROCESS_AUTOMATION_S3_PATH.value}/"
            f"{Attributes.GLUE_JOB_NAME.value}/logs/"
            f"otq_request_input_converted_{result_prefix}_{args['processType']}.csv/{file_name}"
        )

        # Prepare OTQ params
        query_params = {
            "localfile": localfile_path,
            "otq_reader": "/tmp/one_tick_otqs_sec/CSV_READERS.otq::with_range_datetimes",
            "input_timezone": "GMT",
            "sym_prefix": f"{sym_prefix}::::",
            "date_format": "%Y-%m-%d",
            "query_date_field": "order_start_date",
            "query_date_end_field": "order_end_date",
            "symbol_col": "SYMBOL_NAME",
            "uid": "DATA_ID",
            "symbol_date_field": None
        }

        # Run OTQ query
        logger.info(f"Running OTQ query on file: {file_name}")
        otq_result = otq_helper.otq.run(
            "/tmp/one_tick_otqs_sec/Symbol_lookup.otq::lookup_by_symbol",
            query_params=query_params,
            output_structure="OutputStructure.symbol_result_list"
        )

        partial_df = tas_onetick_util.otq_from_list_to_df(otq_result)
        selected_headers_pdf = partial_df[Attributes.OTQ_RESPONSE_HEADERS.value]
        spark_schema = glue_helper.map_spark_schema_from_input_df(Attributes.otq_response_schema.value)
        spark_partial_df = spark.createDataFrame(selected_headers_pdf, schema=spark_schema)

        if final_response_df is None:
            final_response_df = spark_partial_df
        else:
            final_response_df = final_response_df.union(spark_partial_df)

        start_index = end_index
        end_index += Attributes.OTQ_REQUEST_BATCH_SIZE.value

    logger.info(f"Final OTQ merged response count: {final_response_df.count()}")
    return final_response_df


# def otq_response(spark, sym_prefix:str, file_name:str, result_prefix:str, args:dict):
#     otq_nm = "/tmp/one_tick_otqs_sec/Symbol_lookup.otq"
#     graph_name = "lookup_by_symbol"
#     query_params = {
#         "localfile": f"{Attributes}",
#         "otq_reader": "/tmp/one_tick_otqs_sec/CSV_READERS.otq::with_range_datetimes",
#         "input_timezone": "GMT",
#         "sym_prefix": f"{sym_prefix}::::",
#         "date_format": "%Y-%m-%d",
#         "query_date_field": "order_start_date",
#         "query_date_end_field": "order_end_date",
#         "symbol_col": "SYMBOL_NAME",
#         "uid": "DATA_ID",
#         "symbol_date_field": None
#     }
#     logger.info(f"Running OTQ query: {otq_nm} with params: {query_params}")
#     get_otq_results = otq_helper.otq.run(
#         f"{otq_nm}::{graph_name}",
#         query_params=query_params,
#         output_structure="OutputStructure.symbol_result_list"
#     )
#     logger.info("OTQ query completed successfully.")
#     otq_results_df = tas_onetick_util.otq_from_list_to_df(get_otq_results)
#     logger.info(f"OTQ result DataFrame count: {otq_results_df.count()}")
#     selected_headers_pdf = otq_results_df[Attributes.OTQ_RESPONSE_HEADERS.value]
#     final_response_df = convert_otq_response_to_df(
#         selected_headers_pdf,
#         spark)
#     final_response_df.printSchema()
#     logger.info(f"Final OTQ response DataFrame count: {final_response_df.count()}")
#     return final_response_df


def convert_otq_response_to_df(otq_response_df: DataFrame, spark: SparkSession) -> DataFrame:
    spark_schema = glue_helper.map_spark_schema_from_input_df(Attributes.otq_response_schema.value)
    df = spark.createDataFrame(otq_response_df, schema=spark_schema)
    print(f"Schema for Converted Df from Pandas to Spark RDD, {df.printSchema()}")
    df.printSchema()
    return df


def join_onetick_response(onetick_sedol_respones_df: DataFrame, onetick_ticker_respones_df: DataFrame, alloc_securities_srm_srmmf_df: DataFrame) -> DataFrame:
    load_reference_data_to_etl(
        onetick_sedol_respones_df,
        Attributes.SECURITY_IDENTIFIERS_ETL_DB_NAME.value,
        Attributes.ONETICK_SEDOL_RESULTS_TABLE.value,
        Attributes.SECURITY_IDENTIFIERS_ETL_LOCATION.value,
    )

    load_reference_data_to_etl(
        onetick_ticker_respones_df,
        Attributes.SECURITY_IDENTIFIERS_ETL_DB_NAME.value,
        Attributes.ONETICK_TICKER_RESULTS_TABLE.value,
        Attributes.SECURITY_IDENTIFIERS_ETL_LOCATION.value,
    )
    logger.info("Joining OneTick response with SRM_EQUITY and SRM_MF")

    merged_onetick_results_df = onetick_sedol_respones_df.union(onetick_ticker_respones_df)
    logger.info(f"OneTick response DataFrame count: {merged_onetick_results_df.count()}")
    merged_onetick_results_df = merged_onetick_results_df.withColumn(
        "DATA_ID", col("DATA_ID").cast(IntegerType()))
    merged_onetick_results_df.printSchema()

    securities_identifiers_df = alloc_securities_srm_srmmf_df.join(
        merged_onetick_results_df,
        on=["DATA_ID"],
        how="inner"
    )

    securities_identifiers_df.printSchema()
    logger.info(f"Final merged DataFrame count: {securities_identifiers_df.count()}")
    securities_identifiers_df.show(10, False)
    return securities_identifiers_df


def load_reference_data_to_etl(
    df: DataFrame,
    db_name: str,
    table_name: str,
    tca_bucket_path: str):
    logger.info(f"Loading data to Glue catalog table: {db_name}.{table_name}")
    util.delete_tables_and_data(
        db_name = db_name,
        table_name = table_name,
        s3_bucket = args['s3TCASecIdBucket'],
        s3_lcation = tca_bucket_path
    )

    glue_helper.write_df_to_iceberg_table(
        df = df,
        target_path = "s3://"+args['s3TCASecIdBucket'] + "/" + tca_bucket_path+"/"
        +table_name, 
        db_name = db_name,
        db_table = table_name,
        db_catalog = args['platformAccountId'],
        write_mode = "overwrite")


def load_eq_security_identifiers_data(args:dict):
    logger.info("Loading EQ security identifiers data")
    util.delete_tables_and_data(
        db_name = Attributes.SECURITY_IDENTIFIERS_DB_NAME.value,
        table_name = Attributes.SECURITY_TRADE_DATE_IDENTIFIERS_TABLE.value,
        s3_bucket = args['s3TCASecIdBucket'],
        s3_lcation = Attributes.TCA_DATA_PRODUCT_PATH.value,
        catalog_id = args['platformAccountId']
    )
    glue_helper.write_df_to_iceberg_table(
        df = eq_sec_trade_date_identifiers,
        target_path = "s3://"+args['s3TCASecIdBucket'] + "/" + Attributes.TCA_DATA_PRODUCT_PATH.value,
        db_name = Attributes.SECURITY_IDENTIFIERS_DB_NAME.value,
        db_table = Attributes.SECURITY_TRADE_DATE_IDENTIFIERS_TABLE.value,
        db_catalog = args['platformAccountId'],
        write_mode = "overwrite"
    )

if __name__ == "__main__":
    logger = log_service.LogService(
        app_prefix=Attributes.APP_PREFIX.value,
        job_name=Attributes.GLUE_JOB_NAME.value
    ).get_logger()
    util = Utils(logger)
    args = util.get_args_dict(sys_args_list=Attributes.SYS_ARGS_LIST.value)
    glue_helper = glue_service.GlueService(logger)
    s3_helper = s3_service.S3Service(logger, args["sysLevel"])
    sns_helper = sns_service.SNSHelper(args["sysLevel"])
    spark = util.get_spark_session()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    otq_helper = set_up_onetick_env(args)

    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PySpark Version: {pyspark.__version__}")
    logger.info(f"Job Arguments: {args}")

    logger.info("Glue Job Initialized")
    logger.info(f"Starting security trade date identifiers ETL job. Process type: {args['processType']}")

    if args["processType"] == "historic":
        allocations_df = extract_allocations(spark, glue_helper, args)
        securities_partitions_df = extract_securities_partitions_df(spark, glue_helper, args)
        allocations_securities_partitions_df = alloc_with_securities_partitions(allocations_df, securities_partitions_df)
        
        # Extract securities data
        securities_df = extract_securities(spark, glue_helper, args)
        
        # Join allocations with securities and SRM
        allocations_securities_df = join_allocations_with_security(spark, glue_helper, args)
        alloc_securities_srm_srmmf_df = extract_and_join_orders_sec_srm(allocations_df, securities_df, s3_helper, args)

        #Prepare OneTick input DataFrame
        onetick_sedol_input_df = onetick_input_df_with_sedol(alloc_securities_srm_srmmf_df, s3_helper, args)
        onetick_ticker_input_df = onetick_input_df_with_ticker(alloc_securities_srm_srmmf_df, s3_helper, args)

        # Write OneTick input DataFrame to temporary location
        #write_otq_request_to_tmp(onetick_sedol_input_df, "sedol", "onetick_sedol_data_input.csv")
        otq_sedol_response_df = otq_response(spark, onetick_sedol_input_df, "SED::::", "onetick_sedol_data_input.csv", "sedol_", args=args)
        #write_otq_request_to_tmp(onetick_ticker_input_df, "ticker", "onetick_ticker_data_input.csv")
        otq_ticker_response_df = otq_response(spark, onetick_ticker_input_df, "BTKR::::", "onetick_ticker_data_input.csv", "ticker_", args=args)

        eq_sec_trade_date_identifiers = join_onetick_response(otq_sedol_response_df, otq_ticker_response_df, allocations_securities_partitions_df)

        otq_sedol_response = otq_response(spark, "SED", "sedol.csv", "sedol", args)
        otq_ticker_response = otq_response(spark, "BTKR", "ticker.csv", "ticker", args)

        final_merged_result = join_onetick_response(otq_sedol_response, otq_ticker_response, allocations_securities_partitions_df)

        load_eq_security_identifiers_data(args)
        logger.info("ETL job completed successfully")
    
    elif args["processType"] == "daily":
        pass
    else:
        logger.error(f"Process type {args['processType']} is not supported.")
        raise InvalidProcessTypeException(f"Invalid process type: {args['processType']}")
    
spark.stop()
logger.info("Spark session stopped")
