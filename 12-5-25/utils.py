import sys
from awsglue.utils import getResolvedOptions
from taslibrary.glue_service import GlueService
from taslibrary.s3_service import S3Service
from attributes import Attributes
import os
import pandas as pd
from pyspark.context import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession, DataFrame, Window


class Utils:
    def __init__(self, logger):
        self.logger = logger
        self.args = self.get_args_dict(Attributes.SYS_ARGS_LIST.value)
        self.glue_helper = GlueService(self.logger)
        self.s3_helper = S3Service(self.logger, self.args["syslevel"])

    def get_args_dict(self, sys_args_list) -> dict:
        args = getResolvedOptions(sys.argv, sys_args_list)
        self.logger.info(f"GET ARGS: {args}")
        return {k: v for k, v in args.items() if k in sys_args_list}

    def get_spark_session(self) -> SparkSession:
        bucket_name = f"s3://{self.args['S3TCASecIdBucket']}/"
        conf_list = self.s3_helper.set_spark_conf(
            bucket_name=bucket_name,
            warehouse_location=Attributes.DEFAULT_WAREHOUSE_LOCATION.value,
            catalog_id=self.args["platformAccountId"],
            iam_id=self.args["invwcmAccountId"]
        )

        conf_list.append(
            ("hive.metastore.glue.catalogid", self.args["platformAccountId"])
        )
        self.logger.info(f"Added hive.metastore.glue.catalogid to conf_list")

        self.logger.info(f"SPARK CONF_LIST: {conf_list}")
        conf = SparkConf().setAll(conf_list)

        spark = (
            SparkSession.builder
            .appName(self.args["JOB_NAME"])
            .config(conf=conf)
            .enableHiveSupport()
            .getOrCreate()
        )

        self.logger.info("Spark Session Created")
        return spark


    def otq_test(self, sym_prefix:str, onetick_datasubset_test: DataFrame, s3_helper, args: dict) -> DataFrame:
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

        self.logger.info(f"Running OTQ query: {otq_path} with params: {query_params}")
        print(f"Running OTQ query: {otq_path} with params: {query_params}")

        otq_test_response = otq_helper.otq.run(
            f"{otq_path}::{graph_name}",
            query_params=query_params
        )

        self.logger.info("OTQ query completed successfully.")
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
        args: dict,
        has_col: bool = False,
        is_final: bool = False
    ) -> DataFrame:
        from pyspark.sql.window import Window
        from pyspark.sql.functions import col, row_number, monotonically_increasing_id

        # Convert pandas OTQ result to Spark DataFrame
        spark_otq_df = spark.createDataFrame(otq_test_df)
        print(f"Schema for Converted Df from Pandas to Spark RDD, {spark_otq_df.printSchema()}")

        # Rename non-DATA_ID columns to avoid name collisions
        if is_final:
            spark_otq_df = spark_otq_df.select([
                col(c).alias(f"otq_{c}") if c != "DATA_ID" else col(c)
                for c in spark_otq_df.columns
            ])

        # Add DATA_ID to enriched_df if not present
        if not has_col:
            enriched_df = enriched_df.withColumn(
                "DATA_ID", row_number().over(Window.orderBy(monotonically_increasing_id()))
            )

        # Join on DATA_ID
        merged_df = enriched_df.join(spark_otq_df, on="DATA_ID", how="inner")

        # Upload preview to S3
        s3_helper.upload_process_logs_spdf(
            merged_df.limit(100),
            args["s3TCASecIdBucket"].replace("s3://", ""),
            args["JOB_NAME"],
            "final_merged_output"
        )

        # Write full result to Iceberg table
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
