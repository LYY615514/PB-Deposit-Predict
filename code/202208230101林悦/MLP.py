from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import numpy as np

# 初始化 SparkSession
spark = SparkSession.builder \
   .appName("Bank Marketing Model Training with Hyperparameter Tuning (MLP)") \
   .getOrCreate()

# 调整日志级别为 ERROR，减少不必要的日志输出
spark.sparkContext.setLogLevel("ERROR")

print("SparkSession initialized successfully!")

# -------------------------------
# 1. 加载预处理后的数据
# -------------------------------

train_data_path = "/spark/train_data.parquet"
val_data_path = "/spark/val_data.parquet"
test_data_path = "/spark/test_data.parquet"

train_data = spark.read.parquet(train_data_path).repartition(6)
val_data = spark.read.parquet(val_data_path).repartition(6)
test_data = spark.read.parquet(test_data_path).repartition(6)

# -------------------------------
# 2. 定义多层感知器模型
# -------------------------------

# 定义神经网络的层数和每层的神经元数量
layers = [46, 64, 32, 2]  # 输入层 -> 隐藏层 -> 输出层

mlp = MultilayerPerceptronClassifier(
    layers=layers,
    featuresCol="features",
    labelCol="label",
    maxIter=100,  # 最大迭代次数
    blockSize=128,  # 每次梯度下降的样本块大小
    seed=42  # 随机种子
)

# -------------------------------
# 3. 定义超参数网格
# -------------------------------

paramGrid = (
    ParamGridBuilder()
    .addGrid(mlp.maxIter, [50, 100])  # 尝试不同的最大迭代次数
    .addGrid(mlp.blockSize, [64, 128])  # 尝试不同的批量大小
    .build()
)

# -------------------------------
# 4. 定义评估器
# -------------------------------

evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")

# -------------------------------
# 5. 定义交叉验证器
# -------------------------------

cross_validator = CrossValidator(
    estimator=mlp,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator_auc,
    numFolds=3  # 使用 3 折交叉验证
)

# -------------------------------
# 6. 训练模型（带超参数调优）
# -------------------------------

print("Training the MLP model with hyperparameter tuning...")
cv_model = cross_validator.fit(train_data)

# 获取最佳模型
best_mlp_model = cv_model.bestModel

# 打印最佳超参数
print(f"Best maxIter: {best_mlp_model.getMaxIter()}")
print(f"Best blockSize: {best_mlp_model.getBlockSize()}")

# -------------------------------
# 7. 在验证集上进行预测
# -------------------------------

print("Making predictions on the validation set...")
val_predictions = best_mlp_model.transform(val_data)

# 查看预测结果
val_predictions.select("features", "label", "prediction", "probability").show(truncate=False)

# -------------------------------
# 8. 模型评估（验证集）
# -------------------------------

# 使用 AUC 评估模型性能
val_auc = evaluator_auc.evaluate(val_predictions)
print(f"Validation set AUC: {val_auc}")

# 使用准确率、精确率、召回率和 F1 分数评估模型性能
evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

val_accuracy = evaluator_multi.evaluate(val_predictions, {evaluator_multi.metricName: "accuracy"})
val_precision = evaluator_multi.evaluate(val_predictions, {evaluator_multi.metricName: "weightedPrecision"})
val_recall = evaluator_multi.evaluate(val_predictions, {evaluator_multi.metricName: "weightedRecall"})
val_f1_score = evaluator_multi.evaluate(val_predictions, {evaluator_multi.metricName: "f1"})

print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Precision: {val_precision}")
print(f"Validation Recall: {val_recall}")
print(f"Validation F1 Score: {val_f1_score}")

# -------------------------------
# 9. 在测试集上进行预测
# -------------------------------

print("Making predictions on the test set...")
test_predictions = best_mlp_model.transform(test_data)

# 查看预测结果
test_predictions.select("features", "label", "prediction", "probability").show(truncate=False)

# -------------------------------
# 10. 模型评估（测试集）
# -------------------------------

# 使用 AUC 评估模型性能
test_auc = evaluator_auc.evaluate(test_predictions)
print(f"Test set AUC: {test_auc}")

# 使用准确率、精确率、召回率和 F1 分数评估模型性能
test_accuracy = evaluator_multi.evaluate(test_predictions, {evaluator_multi.metricName: "accuracy"})
test_precision = evaluator_multi.evaluate(test_predictions, {evaluator_multi.metricName: "weightedPrecision"})
test_recall = evaluator_multi.evaluate(test_predictions, {evaluator_multi.metricName: "weightedRecall"})
test_f1_score = evaluator_multi.evaluate(test_predictions, {evaluator_multi.metricName: "f1"})

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1_score}")

# -------------------------------
# 11. 绘制 ROC 和 PR 曲线
# -------------------------------

def plot_roc_pr_curves(predictions, title_suffix=""):
    """
    绘制 ROC 和 PR 曲线。
    :param predictions: 包含 'label' 和 'probability' 列的 DataFrame。
    :param title_suffix: 曲线标题后缀。
    """
    # 提取标签和概率
    labels_and_probs = predictions.select("label", "probability").rdd.map(
        lambda row: (float(row["label"]), float(row["probability"][1]))
    ).collect()

    # 分离标签和概率
    y_true, y_scores = zip(*labels_and_probs)

    # 计算 ROC 曲线
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({title_suffix})')
    plt.legend(loc="lower right")

    # 计算 PR 曲线
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    # 绘制 PR 曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({title_suffix})')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# 绘制验证集的 ROC 和 PR 曲线
plot_roc_pr_curves(val_predictions, title_suffix="Validation Set")

# 绘制测试集的 ROC 和 PR 曲线
plot_roc_pr_curves(test_predictions, title_suffix="Test Set")

# -------------------------------
# 12. 保存训练好的模型
# -------------------------------

model_save_path = "/spark/mlp_model_tuned"

# 使用 overwrite() 方法覆盖已有路径
best_mlp_model.write().overwrite().save(model_save_path)
print(f"Model saved successfully to: {model_save_path}")

# -------------------------------
# 13. 关闭 SparkSession
# -------------------------------

spark.stop()