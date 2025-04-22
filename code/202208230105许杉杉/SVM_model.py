from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.storagelevel import StorageLevel
import matplotlib.pyplot as plt  # 直接导入 matplotlib
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # 直接导入 scikit-learn

# -------------------------------
# 1. 初始化 SparkSession
# -------------------------------
spark = SparkSession.builder \
    .appName("Bank Marketing SVM Model") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# 减少日志输出
spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# 2. 加载预处理后的数据
# -------------------------------
train_data_path = "/spark/train_data.parquet"
val_data_path = "/spark/val_data.parquet"
test_data_path = "/spark/test_data.parquet"

# 动态调整分区数
def load_and_optimize(path):
    df = spark.read.parquet(path)
    # 每个分区目标大小为 10MB (可以根据实际数据调整)
    target_partition_size_mb = 10
    num_partitions = max(1, int(df.count() * 1e-3 / target_partition_size_mb))
    return df.repartition(num_partitions)

train_data = load_and_optimize(train_data_path)
val_data = load_and_optimize(val_data_path)
test_data = load_and_optimize(test_data_path)

print(f"训练集大小: {train_data.count()}")
print(f"验证集大小: {val_data.count()}")
print(f"测试集大小: {test_data.count()}")

# 特征标准化并持久化
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
scaler_model = scaler.fit(train_data)

train_data = scaler_model.transform(train_data).persist(StorageLevel.MEMORY_ONLY)
val_data = scaler_model.transform(val_data).persist(StorageLevel.MEMORY_ONLY)
test_data = scaler_model.transform(test_data).persist(StorageLevel.MEMORY_ONLY)

# 触发持久化
train_data.count()
val_data.count()
test_data.count()

# -------------------------------
# 3. 初始化 SVM 模型
# -------------------------------
svm = LinearSVC(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=100,
    standardization=False  # 已经手动标准化
)

# -------------------------------
# 4. 简化超参数网格
# -------------------------------
param_grid = (
    ParamGridBuilder()
    .addGrid(svm.regParam, [0.01, 0.1])  # 只保留两个正则化参数
    .addGrid(svm.maxIter, [50])          # 固定最大迭代次数
    .build()
)

# -------------------------------
# 5. 定义评估器
# -------------------------------
evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# -------------------------------
# 6. 简化交叉验证设置
# -------------------------------
cv = CrossValidator(
    estimator=svm,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,        # 减少折数
    parallelism=4      # 根据集群核心数调整
)

# -------------------------------
# 7. 训练模型
# -------------------------------
print("开始训练SVM模型...")
cv_model = cv.fit(train_data)

# 输出最佳参数
best_model = cv_model.bestModel
print("\n最佳参数:")
print(f"正则化参数(regParam): {best_model.getRegParam()}")
print(f"最大迭代次数(maxIter): {best_model.getMaxIter()}")

# -------------------------------
# 8. 验证集评估
# -------------------------------
print("\n验证集评估:")
val_predictions = cv_model.transform(val_data)

# 计算评估指标
def calculate_metrics(predictions):
    total = predictions.count()
    tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

    metrics = {
        "准确率(Accuracy)": (tp + tn) / total,
        "精确率(Precision)": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "召回率(Recall)": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "AUC": evaluator.evaluate(predictions)
    }
    metrics["F1分数"] = 2 * (metrics["精确率(Precision)"] * metrics["召回率(Recall)"]) / \
                        (metrics["精确率(Precision)"] + metrics["召回率(Recall)"]) \
        if (metrics["精确率(Precision)"] + metrics["召回率(Recall)"]) > 0 else 0
    return metrics

val_metrics = calculate_metrics(val_predictions)
print("\n验证集性能指标:")
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

# -------------------------------
# 9. 测试集评估
# -------------------------------
print("\n测试集评估:")
test_predictions = cv_model.transform(test_data)
test_metrics = calculate_metrics(test_predictions)

print("\n测试集性能指标:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# -------------------------------
# 10. 可视化结果
# -------------------------------
def plot_curves_spark(predictions, title_suffix=""):
    # 提取数据为 RDD 格式
    pd_pred = predictions.select("label", "rawPrediction").rdd.map(lambda row: (row.label, row.rawPrediction[1])).collect()
    y_true = [x[0] for x in pd_pred]
    y_scores = [x[1] for x in pd_pred]

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # PR 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # 绘制图形
    plt.figure(figsize=(12, 5))

    # ROC 图
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve {title_suffix}')
    plt.legend(loc="lower right")

    # PR 图
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='PR Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {title_suffix}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# 调用绘图函数
print("\n生成验证集性能曲线...")
plot_curves_spark(val_predictions, title_suffix="(Validation Set)")

print("\n生成测试集性能曲线...")
plot_curves_spark(test_predictions, title_suffix="(Test Set)")

# -------------------------------
# 11. 模型保存
# -------------------------------
model_save_path = "/spark/svm_model_tuned"
cv_model.write().overwrite().save(model_save_path)
print(f"\n模型已保存到: {model_save_path}")

# -------------------------------
# 12. 资源清理
# -------------------------------
train_data.unpersist()
val_data.unpersist()
test_data.unpersist()

spark.stop()
print("Spark会话已关闭")