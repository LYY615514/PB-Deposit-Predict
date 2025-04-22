from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import numpy as np

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Bank Marketing Model Training with Hyperparameter Tuning") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# -------------------------------
# 1. 加载预处理后的数据
# -------------------------------

train_data_path = "/spark/train_data.parquet"
val_data_path = "/spark/val_data.parquet"
test_data_path = "/spark/test_data.parquet"

train_data = spark.read.parquet(train_data_path)
val_data = spark.read.parquet(val_data_path)
test_data = spark.read.parquet(test_data_path)

# -------------------------------
# 2. 定义随机森林模型
# -------------------------------

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    seed=42  # 随机种子，确保结果可复现
)

# -------------------------------
# 3. 定义超参数网格
# -------------------------------

paramGrid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100])  # 树的数量
    .addGrid(rf.maxDepth, [5, 10])  # 树的最大深度
    .build()
)

# -------------------------------
# 4. 定义评估器
# -------------------------------

evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                              metricName="areaUnderROC")

# -------------------------------
# 5. 定义交叉验证器
# -------------------------------

cross_validator = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator_auc,
    numFolds=3  # 使用 3 折交叉验证
)

# -------------------------------
# 6. 训练模型（带超参数调优）
# -------------------------------

print("Training the random forest model with hyperparameter tuning...")
cv_model = cross_validator.fit(train_data)

# 获取最佳模型
best_rf_model = cv_model.bestModel

# 打印最佳超参数
print(f"Best numTrees: {best_rf_model.getNumTrees}")
print(f"Best maxDepth: {best_rf_model.getMaxDepth}")

# -------------------------------
# 7. 在验证集上进行预测
# -------------------------------

print("Making predictions on the validation set...")
val_predictions = best_rf_model.transform(val_data)

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
# 9. 绘制 ROC 和 PR 曲线
# -------------------------------

def plot_roc_pr_curves(predictions, title_suffix=""):
    # 提取标签和预测概率
    roc_data = predictions.select("label", "rawPrediction").rdd.map(
        lambda row: (float(row["label"]), float(row["rawPrediction"][1]))).collect()
    labels, scores = zip(*roc_data)

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    # 计算 ROC 数据
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 计算 PR 数据
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制 ROC 曲线
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Receiver Operating Characteristic (ROC) Curve - {title_suffix}')
    ax1.legend(loc="lower right")

    # 绘制 PR 曲线
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {title_suffix}')
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


print("Plotting ROC and PR Curves for Validation Set...")
plot_roc_pr_curves(val_predictions, title_suffix="Validation Set")

# -------------------------------
# 10. 在测试集上进行预测
# -------------------------------

print("Making predictions on the test set...")
test_predictions = best_rf_model.transform(test_data)

# 查看预测结果
test_predictions.select("features", "label", "prediction", "probability").show(truncate=False)

# -------------------------------
# 11. 模型评估（测试集）
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
# 12. 绘制 ROC 和 PR 曲线（测试集）
# -------------------------------

print("Plotting ROC and PR Curves for Test Set...")
plot_roc_pr_curves(test_predictions, title_suffix="Test Set")

# -------------------------------
# 13. 保存训练好的模型
# -------------------------------

model_save_path = "/spark/random_forest_model_tuned"

# 使用 overwrite() 方法覆盖已有路径
best_rf_model.write().overwrite().save(model_save_path)
print(f"Model saved successfully to: {model_save_path}")

# -------------------------------
# 14. 关闭 SparkSession
# -------------------------------

spark.stop()