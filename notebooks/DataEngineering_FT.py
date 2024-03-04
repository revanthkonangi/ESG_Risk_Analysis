# Databricks notebook source
# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install /dbfs/FileStore/sdk/dev/MLCoreSDK-0.5.96-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md <b> User Inputs

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

from MLCORE_SDK import mlclient
from pyspark.sql import functions as F

# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# DE SPECIFIC PARAMETERS
primary_keys = solution_config["data_engineering"]["data_engineering_ft"]["primary_keys"]
features_table_name = solution_config["data_engineering"]["data_engineering_ft"]["features_table_name"]
features_dbfs_path = solution_config["data_engineering"]["data_engineering_ft"]["features_dbfs_path"]

is_scheduled = solution_config["data_engineering"]["data_engineering_ft"].get("is_scheduled",False)
batch_size = solution_config["data_engineering"]["data_engineering_ft"].get("batch_size",100)

# COMMAND ----------

if is_scheduled:
    mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = "de")

# COMMAND ----------

pickle_file_path = f"/mnt/FileStore/{db_name}"
dbutils.fs.mkdirs(pickle_file_path)
print(f"Created directory : {pickle_file_path}")
pickle_file_path = f"/dbfs/{pickle_file_path}/{features_table_name}.pickle"

# COMMAND ----------

try : 
  with open(pickle_file_path, "rb") as handle:
      obj_properties = pickle.load(handle)
      print(f"Instance loaded successfully")
except Exception as e:
  print(f"Exception while loading cache : {e}")
  obj_properties = {}
print(f"Existing Cache : {obj_properties}")

# COMMAND ----------

if not obj_properties :
  start_marker = 1
elif obj_properties and obj_properties.get("end_marker",0) == 0:
  start_marker = 1
else :
  start_marker = obj_properties["end_marker"] + 1
end_marker = start_marker + batch_size - 1

print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

features_df = spark.read.load(features_dbfs_path).filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

features_df.display()

# COMMAND ----------

spark_df = features_df

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# Add Date, timestamp, Id column
now = datetime.now()
date = now.strftime("%m-%d-%Y")
spark_df = spark_df.withColumn('timestamp', F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"))
spark_df = spark_df.withColumn("date", F.lit(date))
spark_df = spark_df.withColumn('date', to_date_(F.col('date')))
w = Window.orderBy(F.monotonically_increasing_id())
spark_df = spark_df.withColumn("id", F.row_number().over(w))

# COMMAND ----------

print(f"HIVE METASTORE DATABASE NAME : {db_name}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

# COMMAND ----------

features_df = spark_df

# COMMAND ----------

features_df.count()

# COMMAND ----------

# DBTITLE 1,ADD A MONOTONICALLY INREASING COLUMN - "id"
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in features_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  features_df = features_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

features_df.createOrReplaceTempView(features_table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == features_table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{features_table_name} AS SELECT * FROM {features_table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO hive_metastore.{db_name}.{features_table_name} SELECT * FROM {features_table_name}");

# COMMAND ----------

from pyspark.sql import functions as F
features_hive_table_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{features_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(f"Features Hive Path : {features_hive_table_path}")

# COMMAND ----------

stagemetrics.end()

# COMMAND ----------

stagemetrics.print_report()

# COMMAND ----------

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

# COMMAND ----------

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md <b> Use MLCore SDK to register Features and Ground Truth Tables

# COMMAND ----------

mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = features_table_name,
    num_rows = features_df.count(),
    cols = features_df.columns,
    column_datatype = features_df.dtypes,
    table_schema = features_df.schema,
    primary_keys = primary_keys,
    table_path = features_hive_table_path,
    table_type="internal",
    table_sub_type="Source",
    env = "dev",
    compute_usage_metrics = compute_metrics)

# COMMAND ----------


