from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# -------------------------------
# 1. 初始化 SparkSession
# -------------------------------
spark = SparkSession.builder \
    .appName("Bank Marketing SVM Model") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# 2. 加载预处理后的数据
# -------------------------------
train_data = spark.read.parquet("/spark/train_data.parquet")
val_data = spark.read.parquet("/spark/val_data.parquet")
test_data = spark.read.parquet("/spark/test_data.parquet")

# 特征标准化
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model = scaler.fit(train_data)

train_data = scaler_model.transform(train_data)
val_data = scaler_model.transform(val_data)
test_data = scaler_model.transform(test_data)

# -------------------------------
# 3. 初始化 SVM 模型
# -------------------------------
svm = LinearSVC(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=100,
    standardization=False,
    threshold=0.3  # 调整阈值以偏向正类（提高召回率）
)

# -------------------------------
# 4. 定义超参数网格
# -------------------------------
param_grid = (
    ParamGridBuilder()
    .addGrid(svm.regParam, [0.01, 0.1, 0.5])  # 正则化参数范围
    .addGrid(svm.maxIter, [50, 100, 200])     # 最大迭代次数选项
    .build()
)

# -------------------------------
# 5. 定义评估器
# -------------------------------
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

# -------------------------------
# 6. 交叉验证设置
# -------------------------------
cv = CrossValidator(
    estimator=svm,
    estimatorParamMaps=param_grid,
    evaluator=evaluator_f1,  # 使用 F1 分数作为优化目标
    numFolds=3,
    parallelism=4
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
print(f"分类阈值(threshold): {best_model.getThreshold()}")

# -------------------------------
# 8. 验证集和测试集评估
# -------------------------------
def evaluate_metrics(data, predictions):
    metrics = {
        "准确率(Accuracy)": evaluator_f1.evaluate(predictions, {evaluator_f1.metricName: "accuracy"}),
        "精确率(Precision)": evaluator_f1.evaluate(predictions, {evaluator_f1.metricName: "weightedPrecision"}),
        "召回率(Recall)": evaluator_f1.evaluate(predictions, {evaluator_f1.metricName: "weightedRecall"}),
        "F1分数": evaluator_f1.evaluate(predictions),
        "AUC": BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction").evaluate(predictions)
    }
    print(f"\n{data}性能指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

val_predictions = cv_model.transform(val_data)
evaluate_metrics("验证集", val_predictions)

test_predictions = cv_model.transform(test_data)
evaluate_metrics("测试集", test_predictions)

# -------------------------------
# 9. 可视化结果
# -------------------------------
def plot_curves(predictions, title_suffix=""):
    pd_pred = predictions.select("label", "rawPrediction").rdd.map(lambda row: (row.label, row.rawPrediction[1])).collect()
    y_true, y_scores = zip(*pd_pred)

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
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve {title_suffix}')
    plt.legend(loc="lower right")

    # PR 图
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {title_suffix}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

print("\n生成验证集性能曲线...")
plot_curves(val_predictions, "(Validation Set)")

print("\n生成测试集性能曲线...")
plot_curves(test_predictions, "(Test Set)")

# -------------------------------
# 10. 模型保存与资源清理
# -------------------------------
model_save_path = "/spark/svm_model_tuned"
cv_model.write().overwrite().save(model_save_path)
print(f"\n模型已保存到: {model_save_path}")

spark.stop()
print("Spark会话已关闭")