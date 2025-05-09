import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import *

from attributes import Attributes
from taslibrary import glue_service, s3_service, log_service, sns_helper as sns_service
from taslibrary.ingestion_exceptions.manager import InvalidProcessTypeException
from utils import Utils

window_all = Window.orderBy(monotonically_increasing_id())

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


def extract_and_join_orders_sec_srm(spark, glue_helper, s3_helper, allocations_securities_df: DataFrame, args: dict) -> DataFrame:
    logger.info("Joining with SRM_EQUITY and SRM_MF using Spark SQL from SQL file")

    allocations_securities_df = allocations_securities_df.withColumn(
        "previous_trade_date_est", date_add(col("trade_date_est"), -1)
    )
    allocations_securities_df = allocations_securities_df.withColumn(
        "previous_trade_date_est_str", date_format(col("previous_trade_date_est"), "yyyyMMdd")
    )
    allocations_securities_df.createOrReplaceTempView("alloc_securities_tmp")

    srm_eq_df = glue_helper.read_csv_to_df(
        spark,
        "s3://vgi-invcan-eng-tca-us-east-1/TCA_Business_Automation_SG/data_infrastructure/process_automation/Q01-glue_tca_eq_security_trade-date_identifiers/csv/srm_equity_eng.csv"
    )
    srm_eq_df.createOrReplaceTempView("srm_equity_tmp")

    srm_mf_df = glue_helper.read_csv_to_df(
        spark,
        "s3://vgi-invcan-eng-tca-us-east-1/TCA_Business_Automation_SG/data_infrastructure/process_automation/Q01-glue_tca_eq_security_trade-date_identifiers/csv/srm_mf_eng.csv"
    )
    srm_mf_df.createOrReplaceTempView("srm_mf_tmp")

    alloc_securities_srm_df = glue_helper.read_sql_to_df(
        spark,
        args["awsApplicationPayloadS3Bucket"],
        Attributes.SQL_SCRIPTS_PATH.value + Attributes.ALLOC_SECURITY_SRM_JOIN_SQL.value
    )

    logger.info(f"Final SRM_EQUITY + SRM_MF join result count: {alloc_securities_srm_df.count()}")

    s3_helper.upload_process_logs_spdf(
        alloc_securities_srm_df.limit(10000),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "historic_alloc_securities_srm_enriched"
    )

    glue_helper.write_df_to_glue_catalog_table(
        df=alloc_securities_srm_df,
        target_path="s3://vgi-invcan-eng-tca-us-east-1/etl/gis_tas_eq_etl/sec_trade_date_identifiers",
        db_name="gis_tas_eq_etl",
        db_table="security_identifiers",
        db_catalog="885905049688",
        write_mode="overwrite"
    )

    return alloc_securities_srm_df


def onetick_input_df(enriched_df: DataFrame, s3_helper, args: dict) -> DataFrame:
    subset_df = enriched_df.withColumn("SYMBOL_NAME", col("sedol")) \
        .withColumn(
            "order_start_date",
            to_timestamp(concat_ws(" ", col("trade_date_local").cast("string"), lit("00:00")), "yyyy-MM-dd HH:mm")
        ).withColumn(
            "order_end_date",
            to_timestamp(concat_ws(" ", col("trade_date_local").cast("string"), lit("23:59")), "yyyy-MM-dd HH:mm")
        ).withColumn(
            "sym_date", lit(datetime.today().strftime("%Y-%m-%d"))
        ).withColumn(
            "DATA_ID", row_number().over(window_all)
        ).select(
            "DATA_ID",
            "sec_id",
            col("sec_currency").alias("currency"),
            "ticker",
            "sedol",
            "trade_date_local",
            date_format(col("order_start_date"), "yyyy-MM-dd HH:mm:ss").alias("order_start_Date"),
            date_format(col("order_end_date"), "yyyy-MM-dd HH:mm:ss").alias("order_end_Date"),
            "sym_date",
            "SYMBOL_NAME"
        )

    s3_helper.upload_process_logs_spdf(
        subset_df.limit(20),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "srm_enriched_export"
    )

    return subset_df

def onetick_input_df_with_ticker(enriched_df: DataFrame, s3_helper, args: dict) -> DataFrame:
    filtered_df = enriched_df.filter(col("currency").isin("THB", "AUD"))

    transformed_df = filtered_df.withColumn(
        "SYMBOL_NAME",
        when(
            (col("currency") == "THB") & col("ticker").endsWith("/F"),
            regexp_replace(col("ticker"), "/F$", " TB")
        ).when(
            (col("currency") == "AUD") & col("ticker").rlike("(ZZ|BB|YY|XX\\*?|0A|0B|O|DA|-W)$"),
            regexp_replace(col("ticker"), "(ZZ|BB|YY|XX\\*?|0A|0B|O|DA|-W)$", " AU")
        ).otherwise(col("ticker"))
    ).withColumn(
        "order_start_date",
        to_timestamp(concat_ws(" ", col("trade_date_local").cast("string"), lit("00:00")), "yyyy-MM-dd HH:mm")
    ).withColumn(
        "order_end_date",
        to_timestamp(concat_ws(" ", col("trade_date_local").cast("string"), lit("23:59")), "yyyy-MM-dd HH:mm")
    ).withColumn(
        "sym_date", lit(datetime.today().strftime("%Y-%m-%d"))
    ).withColumn(
        "DATA_ID", row_number().over(window_all)
    ).select(
        "DATA_ID",
        "sec_id",
        col("sec_currency").alias("currency"),
        "ticker",
        "sedol",
        "trade_date_local",
        date_format(col("order_start_date"), "yyyy-MM-dd HH:mm:ss").alias("order_start_Date"),
        date_format(col("order_end_date"), "yyyy-MM-dd HH:mm:ss").alias("order_end_Date"),
        "sym_date",
        "SYMBOL_NAME"
    )

    s3_helper.upload_process_logs_spdf(
        transformed_df.limit(20),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "srm_enriched_export_ticker"
    )

    return transformed_df


def otq_test(sym_prefix:str, onetick_datasubset_test: DataFrame, s3_helper, args: dict) -> DataFrame:
    print("Import OneTick libraries from S3")

    ot_key_id = "40d1148-bf89-420d-84ca-ecl8e23674ef"
    ot_arn = "arn:aws:kms:us-east-1:503680829471:key"
    ot_s3_loc = "s3://vgi-invcan-eng-tca-us-east-1/tmp/test_onetick"

    cmd = f"aws s3 cp {ot_s3_loc}/ /tmp --sse aws:kms --sse-kms-key-id {ot_arn}/{ot_key_id} --recursive"
    return_code = os.system(cmd)
    if return_code != 0:
        print(f"Error with return code: {return_code}")
        raise Exception(f"Failed to copy OneTickClient from S3: {ot_s3_loc}")

    print("Execute command to hard link to onetick directory")
    cmd = "ln -s /usr/lib64/libpython3.11.so.1.0 /tmp/one_tick/bin/libpython3.11.so"
    return_code = os.system(cmd)
    if return_code != 0:
        raise Exception("Failed to hardlink onetick directory")
    print("Hard link to onetick directory successful")

    print("Importing OneTick Query Wrapper")
    sys.path.insert(0, '/tmp/util')
    import taslibrary.tas_onetick_utility as tas_onetick_util
    from taslibrary.tas_onetick_query_helper import TasOneTickQueryHelper

    otq_helper = TasOneTickQueryHelper(sys_level="eng", sys_type="glue")
    os.mkdir('/tmp/input')

    limited_df = onetick_datasubset_test.limit(50)
    pandas_df = limited_df.toPandas()
    pandas_df.to_csv('/tmp/input/sample_subset_data.csv', sep=',', header=True, index=False)

    otq_path = "/tmp/one_tick_otqs_sec/Symbol_lookup.otq"
    graph_name = "lookup_by_symbol"
    query_params = {
        "localfile": "/tmp/input/sample_subset_data.csv",
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

    logger.info(f"Running OTQ query: {otq_path} with params: {query_params}")
    print(f"Running OTQ query: {otq_path} with params: {query_params}")

    otq_test_response = otq_helper.otq.run(
        f"{otq_path}::{graph_name}",
        query_params=query_params
    )

    logger.info("OTQ query completed successfully.")
    print("OTQ query completed successfully.")

    otq_test_df = tas_onetick_util.otq_from_list_to_df(otq_test_response)
    s3_helper.upload_process_logs_spdf(
        otq_test_df,
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "otq_results"
    )

    print("OTQ Result Count:", otq_test_df.count())
    return otq_test_df




def merge_enriched_with_otq_results(
    spark: SparkSession,
    otq_test_df: pd.DataFrame,
    enriched_df: DataFrame,
    s3_helper,
    glue_helper,
    args: dict
) -> DataFrame:
    """
    Converts OTQ results (pandas DataFrame) to Spark DataFrame,
    assigns DATA_ID, joins with enriched_df on DATA_ID, and uploads preview logs.
    """
    from pyspark.sql.window import Window

    # Convert pandas to Spark
    spark_otq_df = spark.createDataFrame(otq_test_df)
    print(f"Schema for Converted Df from Pandas to Spark RDD, {spark_otq_df.printSchema()}")

    # Add DATA_ID to Spark OTQ DataFrame
    enriched_df = enriched_df.withColumn(
        "DATA_ID", row_number().over(Window.orderBy(monotonically_increasing_id()))
    )

    # Join with enriched_df on DATA_ID
    merged_df = enriched_df.join(spark_otq_df, on="DATA_ID", how="inner")

    # Upload log of merged result (sample)
    s3_helper.upload_process_logs_spdf(
        merged_df.limit(100),
        args["s3TCASecIdBucket"].replace("s3://", ""),
        args["JOB_NAME"],
        "final_merged_output"
    )
    glue_helper.write_df_to_iceberg_table(
    df=merged_df,
    target_path="s3://",  
    db_name="",                       
    db_table="",                   
    db_catalog="",                    
    write_mode="overwrite",                       
    partition_cols=None                           
    )
    return merged_df


def main():
    global logger
    logger = log_service.LogService(
        app_prefix=Attributes.APP_PREFIX.value,
        job_name=Attributes.GLUE_JOB_NAME.value
    ).get_logger()

    util = Utils(logger)
    args = util.get_args_dict(sys_args_list=Attributes.SYS_ARGS_LIST.value)

    spark = util.get_spark_session()
    glue_helper = glue_service.GlueService(logger)
    s3_helper = s3_service.S3Service(logger, args["sysLevel"])
    sns_helper = sns_service.SNSHelper(args["sysLevel"])

    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PySpark Version: {pyspark.__version__}")
    logger.info(f"Job Arguments: {args}")
    logger.info("Glue Job Initialized")
    logger.info(f"Starting security trade date identifiers ETL job. Process type: {args['processType']}")

    if args["processType"] == "historic":
        allocations_df = extract_allocations(spark, glue_helper, s3_helper, args)
        securities_df = extract_securities(spark, s3_helper, args)
        allocations_securities_df = join_orders_with_security(allocations_df, securities_df, s3_helper, args)
        enriched_df = extract_and_join_orders_sec_srm(spark, glue_helper, s3_helper, allocations_securities_df, args)
        otq_input_df = onetick_input_df(enriched_df, s3_helper, args)
        otq_ticker_df = onetick_input_df_with_ticker(enriched_df, s3_helper, args)
        otq_result_pandas_df = otq_test("SED", otq_input_df, s3_helper, args)
        otq_ticker_pandas_df = otq_test("BTKR", otq_ticker_df, s3_helper, args)
        final_df = merge_enriched_with_otq_results(spark, otq_result_pandas_df, enriched_df, s3_helper, glue_helper, args)
        final_ticker_df = merge_enriched_with_otq_results(spar, otq_ticker_pandas_df, enriched_df, s3_helper, glue_helper, args)
        logger.info("security trade date identifiers job completed successfully")
    elif args["processType"] == "daily":
        pass
    else:
        logger.error("Process Type not in (historic, daily) - Invalid Process Type")
        raise InvalidProcessTypeException(args['processType'])

    spark.stop()
    logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()
