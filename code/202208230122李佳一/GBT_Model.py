from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# 初始化 SparkSession
spark = SparkSession.builder \
   .appName("Bank Marketing Model Training with GBT Hyperparameter Tuning") \
   .getOrCreate()

# 调整日志级别为 ERROR，减少不必要的日志输出
spark.sparkContext.setLogLevel("ERROR")

print("SparkSession initialized successfully!")

# 1. 加载预处理后的数据
train_data_path = "/spark/train_data.parquet"
val_data_path = "/spark/val_data.parquet"
test_data_path = "/spark/test_data.parquet"

train_data = spark.read.parquet(train_data_path).repartition(6)
val_data = spark.read.parquet(val_data_path).repartition(6)
test_data = spark.read.parquet(test_data_path).repartition(6)

# 2. 定义梯度提升树模型
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    seed=42,  # 随机种子，确保结果可复现
    maxIter=80  # 最大迭代次数（树的数量）
)

# 3. 定义超参数网格
paramGrid = (
    ParamGridBuilder()
   .addGrid(gbt.maxDepth, [5])  # 树的最大深度
   .addGrid(gbt.stepSize, [0.01])  # 学习率
   .build()
)

# 4. 定义评估器
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# 5. 定义交叉验证器
cross_validator = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator_auc,
    numFolds=2  # 减少折数以加速训练
)

# 6. 训练模型（带超参数调优）
print("Training the gradient boosted tree model with hyperparameter tuning...")
cv_model = cross_validator.fit(train_data)

# 获取最佳模型
best_gbt_model = cv_model.bestModel

# 打印最佳超参数
print(f"Best maxDepth: {best_gbt_model.getMaxDepth()}")
print(f"Best stepSize: {best_gbt_model.getStepSize()}")

# 7. 在验证集上进行预测
print("Making predictions on the validation set...")
val_predictions = best_gbt_model.transform(val_data)

# 查看预测结果
val_predictions.select("features", "label", "prediction", "probability").show(truncate=False)

# 8. 模型评估（验证集）
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

# 9. 在测试集上进行预测
print("Making predictions on the test set...")
test_predictions = best_gbt_model.transform(test_data)

# 查看预测结果
test_predictions.select("features", "label", "prediction", "probability").show(truncate=False)

# 10. 模型评估（测试集）
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

# 11. 绘制 ROC 曲线（采样部分数据以减少计算量）
print("Plotting ROC and PR curves...")
sampled_val_predictions = val_predictions.sample(False, 0.1).collect()  # 采样 10% 的数据
labels = [row.label for row in sampled_val_predictions]
probs = [row.probability[1] for row in sampled_val_predictions]

# 计算 ROC 曲线
fpr_val, tpr_val, _ = roc_curve(labels, probs)
roc_auc_val = auc(fpr_val, tpr_val)

# 计算 PR 曲线
precision_val, recall_val, _ = precision_recall_curve(labels, probs)
pr_auc_val = auc(recall_val, precision_val)

# 绘制 ROC 和 PR 曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC curve (area = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Validation Set')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val, label=f'PR curve (area = {pr_auc_val:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve - Validation Set')
plt.legend(loc="lower right")

plt.show()

# 12. 保存训练好的模型
model_save_path = "/spark/gbt_model_tuned"

# 使用 overwrite() 方法覆盖已有路径
best_gbt_model.write().overwrite().save(model_save_path)
print(f"Model saved successfully to: {model_save_path}")

# 13. 关闭 SparkSession
spark.stop()