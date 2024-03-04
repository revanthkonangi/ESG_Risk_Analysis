# Databricks notebook source
# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install /dbfs/FileStore/sdk/Revanth/MLCoreSDK-0.5.96-py3-none-any.whl --force-reinstall

# COMMAND ----------

# DBTITLE 1,Uncomment in case you want to register the data in feature store.
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

try : 
    env = dbutils.widgets.get("env")
    task = dbutils.widgets.get("task")
except :
    env, task = "dev","fe"
print(f"Input environment : {env}")
print(f"Input task : {task}")

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

from MLCORE_SDK import mlclient

# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# JOB SPECIFIC PARAMETERS FOR FEATURE PIPELINE
if task.lower() == "fe":
    features_dbfs_path = solution_config["feature_pipelines"]["feature_pipelines_ft"]["features_dbfs_path"]
    transformed_features_table_name = solution_config["feature_pipelines"]["feature_pipelines_ft"]["transformed_features_table_name"]
    is_scheduled = solution_config["feature_pipelines"]["feature_pipelines_ft"]["is_scheduled"]
    batch_size = int(solution_config["feature_pipelines"]["feature_pipelines_ft"].get("batch_size",500))
    cron_job_schedule = solution_config["feature_pipelines"]["feature_pipelines_ft"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["feature_pipelines"]["feature_pipelines_ft"]["primary_keys"]
else:
    # JOB SPECIFIC PARAMETERS FOR DATA PREP DEPLOYMENT
    features_dbfs_path = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["features_dbfs_path"]
    transformed_features_table_name = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["transformed_features_table_name"]
    is_scheduled = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["is_scheduled"]
    batch_size = int(solution_config["data_prep_deployments"]["data_prep_deployment_ft"].get("batch_size",500))
    cron_job_schedule = solution_config["data_prep_deployments"]["data_prep_deployment_ft"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["data_prep_deployments"]["data_prep_deployment_ft"]["primary_keys"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### FEATURE ENGINEERING

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### FEATURE ENGINEERING on Feature Data

# COMMAND ----------

# DBTITLE 1,Load the data
features_df = spark.sql(f"SELECT * FROM {db_name}.{features_dbfs_path}")

# COMMAND ----------

from pyspark.sql import functions as F
import pickle

# COMMAND ----------

if is_scheduled:
  pickle_file_path = f"/mnt/FileStore/{db_name}"
  dbutils.fs.mkdirs(pickle_file_path)
  print(f"Created directory : {pickle_file_path}")
  pickle_file_path = f"/dbfs/{pickle_file_path}/{transformed_features_table_name}.pickle"

  try : 
    with open(pickle_file_path, "rb") as handle:
        obj_properties = pickle.load(handle)
        print(f"Instance loaded successfully")
  except Exception as e:
    print(f"Exception while loading cache : {e}")
    obj_properties = {}
  print(f"Existing Cache : {obj_properties}")

  if not obj_properties :
    start_marker = 1
  elif obj_properties and obj_properties.get("end_marker",0) == 0:
    start_marker = 1
  else :
    start_marker = obj_properties["end_marker"] + 1
  end_marker = start_marker + batch_size - 1
else :
  start_marker = 1
  end_marker = features_df.count()

  print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

# DBTITLE 1,Perform some feature engineering step. 
FT_DF = features_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

# DBTITLE 1,Exit the job if there is no new data
if not FT_DF.first():
  dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

if task.lower() != "fe":
    # Calling job run add for DPD job runs
    mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = task)

# COMMAND ----------

data = FT_DF.toPandas()

# COMMAND ----------

import numpy as np

# COMMAND ----------

data.replace('null', np.nan, inplace=True)

# Count null values in each column
null_counts = data.isnull().sum()

# COMMAND ----------

null_counts

# COMMAND ----------

data.info()

# COMMAND ----------

mode_value_col2 = data['Full_Time_Employees'].mode()[0]  # Calculate mode for 'col2'

# Fill null values in 'col2' with the mode
data['Full_Time_Employees'].fillna(mode_value_col2, inplace=True)

# COMMAND ----------

data['Full_Time_Employees'] = data['Full_Time_Employees'].str.replace(',', '')

# Convert back to float
data['Full_Time_Employees'] = data['Full_Time_Employees'].astype(float)

# Fill NA values with the median
data['Full_Time_Employees'].fillna(data['Full_Time_Employees'].median(), inplace=True)


# COMMAND ----------

columns_to_impute = ['Environment_Risk_Score', 'Governance_Risk_Score','Social_Risk_Score','Controversy_Score']

# Impute missing values with the median for the specified columns
for col in columns_to_impute:
    median_value = data[col].median()
    data[col].fillna(median_value, inplace=True)

# COMMAND ----------

data.info()

# COMMAND ----------

mode_value_col2 = data['Controversy_Level'].mode()[0]  # Calculate mode for 'col2'

# Fill null values in 'col2' with the mode
data['Controversy_Level'].fillna(mode_value_col2, inplace=True)

# COMMAND ----------

data.info()

# COMMAND ----------

data['Pincode'] = data['Address'].str.extract(r'(\b\d{5}\b)').astype(float)

# COMMAND ----------

data.info()

# COMMAND ----------

columns_to_impute = ['Controversy_Level','Sector','Industry']

# Impute missing values with the mode for the specified columns
for col in columns_to_impute:
    mode_value = data[col].mode()[0]  # Calculate mode for each column
    data[col].fillna(mode_value, inplace=True)

print()

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

columns_to_encode = ['Controversy_Level','Sector','Industry']

# COMMAND ----------

label_encoder = LabelEncoder()

# Iterate over each column and label encode
for col in columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

# COMMAND ----------

FT_DF = spark.createDataFrame(data)

# COMMAND ----------

FT_DF.display()

# COMMAND ----------

FT_DF.createOrReplaceTempView(transformed_features_table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == transformed_features_table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS hive_metastore.{db_name}.{transformed_features_table_name} AS SELECT * FROM {transformed_features_table_name}")
else :
  print(F"UPDATING TABLE")
  spark.sql(f"INSERT INTO hive_metastore.{db_name}.{transformed_features_table_name} SELECT * FROM {transformed_features_table_name}");

# COMMAND ----------

from pyspark.sql import functions as F
features_hive_table_path = spark.sql(f"desc formatted hive_metastore.{db_name}.{transformed_features_table_name}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
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

# MAGIC %md
# MAGIC
# MAGIC ### REGISTER THE FEATURES ON MLCORE
# MAGIC

# COMMAND ----------

# DBTITLE 1,Register Features Transformed Table
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = transformed_features_table_name,
    num_rows = FT_DF.count(),
    cols = FT_DF.columns,
    column_datatype = FT_DF.dtypes,
    table_schema = FT_DF.schema,
    primary_keys = primary_keys,
    table_path = features_hive_table_path,
    table_type="internal",
    table_sub_type="Source",
    request_type = task,
    env = env,
    batch_size = batch_size,
    quartz_cron_expression = cron_job_schedule,
    compute_usage_metrics = compute_metrics)

# COMMAND ----------

if is_scheduled:
  obj_properties['end_marker'] = end_marker
  with open(pickle_file_path, "wb") as handle:
      pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Instance successfully saved successfully")

# COMMAND ----------

# try :
#     #define media artifacts path
#     media_artifacts_path = utils.get_media_artifact_path(sdk_session_id = sdk_session_id,
#         dbutils = dbutils,
#         env = env)
    
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Custom_EDA", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#             "input_table_path" : features_hive_table_path,
#             "media_artifacts_path" : media_artifacts_path,
#             })
# except Exception as e:
#     print(f"Exception while triggering EDA notebook : {e}")
