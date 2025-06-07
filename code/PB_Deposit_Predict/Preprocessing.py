from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Bank Marketing Data Preprocessing") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 加载数据
data_path = "/spark/bank-additional-full.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True, sep=";", quote='"')

# 查看原始列名
print("Original columns:", df.columns)

# 重命名列，将特殊字符替换为下划线
for old_col in df.columns:
    new_col = old_col.replace(".", "_")  # 替换点号为下划线
    df = df.withColumnRenamed(old_col, new_col)

# 查看重命名后的列名
print("Renamed columns:", df.columns)

# -------------------------------
# 1. 数据清洗
# -------------------------------

# 替换分类变量中的 "unknown" 为 None
for column in df.columns:
    if df.select(column).dtypes[0][1] == "string":
        df = df.withColumn(
            column,
            when(col(column) == "unknown", None).otherwise(col(column))
        )

# 填充缺失值
for col_name in df.columns:
    if df.select(col_name).dtypes[0][1] == "double":
        mean_val = df.select(mean(col(col_name))).collect()[0][0]
        df = df.fillna({col_name: mean_val})
    elif df.select(col_name).dtypes[0][1] == "string":
        mode_val = df.groupBy(col_name).count().orderBy("count", ascending=False).first()[0]
        df = df.fillna({col_name: mode_val})

# 过滤异常值
df = df.filter((col("age") >= 18) & (col("age") <= 100))

# -------------------------------
# 2. 移除 duration 列
# -------------------------------

# 移除 duration 列，因为它不适合用于现实预测模型
df = df.drop("duration")

# -------------------------------
# 3. 特征工程
# -------------------------------

# 定义分类列和数值列
categorical_cols = ["job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "poutcome"]
numeric_cols = ["age", "campaign", "pdays", "previous",
                "emp_var_rate", "cons_price_idx", "cons_conf_idx",
                "euribor3m", "nr_employed"]

# 使用 StringIndexer 和 OneHotEncoder 处理分类特征
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(df) for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded") for col in categorical_cols]

# 标准化数值特征
assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_features")

# 合并所有特征
assembler_inputs = [f"{col}_encoded" for col in categorical_cols] + ["scaled_features"]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# 目标变量编码
label_indexer = StringIndexer(inputCol="y", outputCol="label").fit(df)

# -------------------------------
# 4. 构建流水线
# -------------------------------

pipeline = Pipeline(stages=indexers + encoders + [assembler_numeric, scaler, assembler, label_indexer])

# 执行流水线
pipeline_model = pipeline.fit(df)
preprocessed_data = pipeline_model.transform(df)

# 保存pipeline模型以便将来使用
pipeline_model_save_path = "/spark/pipeline_model"
pipeline_model.write().overwrite().save(pipeline_model_save_path)
print(f"Pipeline model saved to: {pipeline_model_save_path}")
# 查看预处理后的数据
preprocessed_data.select("features", "label").show(truncate=False)

# -------------------------------
# 5. 数据拆分
# -------------------------------

train_data, val_data, test_data = preprocessed_data.randomSplit([0.7, 0.15, 0.15], seed=42)

# 仅对训练数据进行过采样
train_pandas = train_data.toPandas()
X = train_pandas["features"].apply(lambda x: x.toArray()).tolist()
y = train_pandas["label"]

# 使用 SMOTE 进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将过采样后的数据转换回 PySpark DataFrame
from pyspark.ml.linalg import Vectors

resampled_data = [(Vectors.dense(row), label) for row, label in zip(X_resampled, y_resampled)]
resampled_df = spark.createDataFrame(resampled_data, ["features", "label"])

# 查看数据集大小
print(f"Original training set size: {train_data.count()}")
print(f"Resampled training set size: {resampled_df.count()}")
print(f"Validation set size: {val_data.count()}")
print(f"Testing set size: {test_data.count()}")

# -------------------------------
# 6. 保存预处理后的数据
# -------------------------------
resampled_df.write.mode("overwrite").parquet("/spark/train_data.parquet")
val_data.write.mode("overwrite").parquet("/spark/val_data.parquet")
test_data.write.mode("overwrite").parquet("/spark/test_data.parquet")

spark.stop()
