# customer_segmentation.py

from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import json


def perform_customer_segmentation(spark, model_save_path, test_data_path):
    # 加载保存的随机森林模型
    loaded_model = RandomForestClassificationModel.load(model_save_path)
    print(f"Model loaded successfully from: {model_save_path}")

    # 加载测试数据或生产环境数据
    test_data = spark.read.parquet(test_data_path).repartition(6)

    # 使用加载的模型进行预测
    predictions = loaded_model.transform(test_data)

    # 客户分层逻辑
    def customer_segmentation(probability):
        prob_value = probability[1]  # 提取正类（1 类）的概率
        if prob_value > 0.8:
            return "High Potential"
        elif 0.5 <= prob_value <= 0.8:
            return "Medium Potential"
        else:
            return "Low Potential"

    # 注册 UDF
    segment_udf = udf(customer_segmentation, StringType())

    # 添加客户分层列
    segmented_predictions = predictions.withColumn("potential_segment", segment_udf(predictions["probability"]))

    # 分层统计
    segment_stats = segmented_predictions.groupBy("potential_segment").count().orderBy("potential_segment")

    # 将分层统计结果转换为 Pandas DataFrame
    segment_stats_pd = segment_stats.toPandas()

    # 为ECharts准备数据 - 确保数据类型正确
    segment_chart_data = {
        'categories': segment_stats_pd["potential_segment"].tolist(),
        'values': [int(x) for x in segment_stats_pd["count"].tolist()],  # 确保是整数
        'colors': ['#91cc75', '#fac858', '#ee6666']  # 绿色、橙色、红色
    }

    # 保存分层结果到文件系统
    output_save_path_parquet = "/spark/customer_segmentation_results.parquet"
    segmented_predictions.write.mode("overwrite").parquet(output_save_path_parquet)
    print(f"Segmentation results saved as Parquet to: {output_save_path_parquet}")

    return segment_stats_pd, segment_chart_data  # 直接返回字典，不转换为JSON