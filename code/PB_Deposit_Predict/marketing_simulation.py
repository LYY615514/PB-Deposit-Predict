# marketing_simulation.py

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import json


def perform_marketing_simulation(spark, model_save_path, test_data_path):
    # 加载已训练的模型
    loaded_model = RandomForestClassificationModel.load(model_save_path)
    print(f"Model loaded from: {model_save_path}")

    # 加载测试数据或新客户数据
    test_data = spark.read.parquet(test_data_path).repartition(6)

    # 使用模型进行预测
    predictions = loaded_model.transform(test_data)

    # 客户分层逻辑（Potential Segment）
    def customer_segmentation(probability, threshold_high=0.8, threshold_medium=0.5):
        prob_value = probability[1]
        if prob_value > threshold_high:
            return "High Potential"
        elif threshold_medium <= prob_value <= threshold_high:
            return "Medium Potential"
        else:
            return "Low Potential"

    segment_udf = udf(customer_segmentation, StringType())
    segmented_predictions = predictions.withColumn("potential_segment", segment_udf(col("probability")))

    # 营销策略模拟函数
    strategy_1 = segmented_predictions.filter(col("potential_segment") == "High Potential")
    strategy_2 = segmented_predictions.filter(
        (col("potential_segment") == "High Potential") |
        (col("potential_segment") == "Medium Potential")
    )
    strategy_3 = segmented_predictions.sample(withReplacement=False, fraction=0.3, seed=42)

    # 计算每个策略的总人数和预计转化人数
    def calculate_conversion_stats(df, strategy_name):
        total = df.count()
        converted = df.filter(col("prediction") == 1.0).count()
        conversion_rate = round(converted / total, 4) if total > 0 else 0
        return {
            "Strategy": strategy_name,
            "Total Customers": total,
            "Expected Conversions": converted,
            "Conversion Rate": conversion_rate
        }

    # 获取策略统计
    results = [
        calculate_conversion_stats(strategy_1, "High Potential Only"),
        calculate_conversion_stats(strategy_2, "High + Medium Potential"),
        calculate_conversion_stats(strategy_3, "Random Sampling (30%)")
    ]

    # 转换为 Pandas DataFrame 方便可视化
    import pandas as pd
    results_df = pd.DataFrame(results)

    # 为ECharts准备数据
    marketing_chart_data = {
        'strategies': results_df["Strategy"].tolist(),
        'customers': [int(x) for x in results_df["Total Customers"].tolist()],
        'conversions': [int(x) for x in results_df["Expected Conversions"].tolist()],
        'rates': [float(x) for x in results_df["Conversion Rate"].tolist()]
    }

    output_path = "/spark/marketing_simulation_results.parquet"
    segmented_predictions.write.mode("overwrite").parquet(output_path)

    return results_df, marketing_chart_data