# risk_warning.py

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, when, count, sum, avg, expr, lit, concat_ws, array, struct, rank, datediff, \
    current_date, monotonically_increasing_id
from pyspark.sql.types import FloatType, StringType, ArrayType, MapType
import pandas as pd


def perform_risk_warning(spark, segmentation_results_path, historical_data_path=None):
    """
    Enhanced risk warning system with multiple improvements
    """
    # 加载分层结果数据
    segmented_data = spark.read.parquet(segmentation_results_path)

    segmented_data = segmented_data.withColumn("customer_id", monotonically_increasing_id())

    # ===== 1. RISK CONFIGURATION =====
    # 配置风险阈值
    churn_risk_config = {
        "high_potential": {"pdays_threshold": 30, "campaign_threshold": 2},
        "medium_potential": {"pdays_threshold": 60, "campaign_threshold": 3},
        "low_potential": {"pdays_threshold": 90, "campaign_threshold": 5}
    }

    credit_risk_config = {
        "high": {"conf_idx_threshold": -40},
        "medium": {"conf_idx_threshold": -20}
    }

    # ===== 2. IMPROVED RISK SCORING =====
    # 计算流失风险分数 (0-100 scale)
    risk_data = segmented_data.withColumn(
        "churn_risk_score",
        when(col("potential_segment") == "Low Potential",
             # 分数越高风险越大
             (when(col("pdays") > churn_risk_config["low_potential"]["pdays_threshold"], 40).otherwise(20)) +
             (when(col("campaign") > churn_risk_config["low_potential"]["campaign_threshold"], 30).otherwise(15)) +
             (when(col("previous") < 2, 30).otherwise(10))
             ).when(col("potential_segment") == "Medium Potential",
                    (when(col("pdays") > churn_risk_config["medium_potential"]["pdays_threshold"], 35).otherwise(15)) +
                    (when(col("campaign") > churn_risk_config["medium_potential"]["campaign_threshold"], 25).otherwise(
                        10)) +
                    (when(col("poutcome_index") < 0.5, 25).otherwise(5))
                    ).when(col("potential_segment") == "High Potential",
                           (when(col("pdays") > churn_risk_config["high_potential"]["pdays_threshold"], 25).otherwise(
                               10)) +
                           (when(col("previous") < 1, 15).otherwise(5))
                           ).otherwise(10)
    )

    # 计算信用风险分数
    risk_data = risk_data.withColumn(
        "credit_risk_score",
        (when(col("cons_conf_idx") < credit_risk_config["high"]["conf_idx_threshold"], 50).
         when(col("cons_conf_idx") < credit_risk_config["medium"]["conf_idx_threshold"], 30).
         otherwise(10)) +
        (when(col("default") == "yes", 50).otherwise(0)) +
        (when(col("loan") == "yes", 15).otherwise(0)) +
        (when(col("housing") == "yes", 15).otherwise(0))
    )

    # ===== 3. CATEGORICAL RISK LEVELS WITH CONFIDENCE =====
    # 将分数转换为分类，并添加置信度
    risk_data = risk_data.withColumn(
        "churn_risk",
        when(col("churn_risk_score") >= 70, "High Churn Risk").
        when(col("churn_risk_score") >= 40, "Medium Churn Risk").
        otherwise("Low Churn Risk")
    ).withColumn(
        "churn_confidence",
        when(col("churn_risk_score") >= 85, "Very High").
        when(col("churn_risk_score") >= 70, "High").
        when(col("churn_risk_score") >= 55, "Medium").
        when(col("churn_risk_score") >= 40, "Low").
        otherwise("Very Low")
    )

    risk_data = risk_data.withColumn(
        "credit_risk",
        when(col("credit_risk_score") >= 60, "High Credit Risk").
        when(col("credit_risk_score") >= 30, "Medium Credit Risk").
        otherwise("Low Credit Risk")
    ).withColumn(
        "credit_confidence",
        when(col("credit_risk_score") >= 75, "Very High").
        when(col("credit_risk_score") >= 60, "High").
        when(col("credit_risk_score") >= 45, "Medium").
        when(col("credit_risk_score") >= 30, "Low").
        otherwise("Very Low")
    )

    # ===== 4. COMBINED RISK ASSESSMENT =====
    # 创建综合风险指标
    risk_data = risk_data.withColumn(
        "composite_risk_score",
        (col("churn_risk_score") * 0.6) + (col("credit_risk_score") * 0.4)
    ).withColumn(
        "overall_risk_level",
        when(col("composite_risk_score") >= 65, "High Risk").
        when(col("composite_risk_score") >= 35, "Medium Risk").
        otherwise("Low Risk")
    )

    # ===== 5. ACTIONABLE RECOMMENDATIONS =====
    # 基于风险档案生成定制化行动建议
    risk_data = risk_data.withColumn(
        "recommended_actions",
        when((col("churn_risk") == "High Churn Risk") & (col("credit_risk") == "High Credit Risk"),
             "紧急：安排立即个人联系。提供客户保留计划和财务咨询。"
             ).when((col("churn_risk") == "High Churn Risk") & (col("credit_risk") != "High Credit Risk"),
                    "高优先级：启动客户保留活动。考虑基于以往互动的个性化优惠。"
                    ).when((col("churn_risk") != "High Churn Risk") & (col("credit_risk") == "High Credit Risk"),
                           "需要关注：密切监控账户。考虑主动信贷咨询或重组选项。"
                           ).when(
            (col("churn_risk") == "Medium Churn Risk") & (col("credit_risk") == "Medium Credit Risk"),
            "中等关注：纳入下次外联活动。评估产品适合度和满意度。"
            ).otherwise(
            "常规：通过标准营销渠道保持定期互动。"
        )
    )

    # ===== 6. RISK PRIORITIZATION =====
    # 创建风险客户优先级列表
    window_spec = Window.orderBy(col("composite_risk_score").desc())
    risk_data = risk_data.withColumn("risk_priority_rank", rank().over(window_spec))

    # ===== 7. RISK SUMMARY STATISTICS =====
    # 计算各风险细分的详细统计
    risk_summary = risk_data.groupBy("overall_risk_level").agg(
        count("*").alias("customer_count"),
        avg("churn_risk_score").alias("avg_churn_score"),
        avg("credit_risk_score").alias("avg_credit_score"),
        avg("composite_risk_score").alias("avg_composite_score")
    ).orderBy("overall_risk_level")

    # ===== 8. OUTPUT PREPARATION =====
    # 准备风险统计和图表数据用于可视化
    risk_counts = risk_data.groupBy("churn_risk", "credit_risk", "overall_risk_level").count().orderBy(
        "churn_risk", "credit_risk", "overall_risk_level"
    ).toPandas()

    # 为ECharts准备数据
    churn_risk_data = risk_data.groupBy("churn_risk").count().orderBy("churn_risk").toPandas()
    credit_risk_data = risk_data.groupBy("credit_risk").count().orderBy("credit_risk").toPandas()
    overall_risk_data = risk_data.groupBy("overall_risk_level").count().orderBy("overall_risk_level").toPandas()

    # 不同风险级别的颜色
    risk_colors = {
        "High": "#ee6666",
        "Medium": "#fac858",
        "Low": "#91cc75"
    }

    risk_chart_data = {
        'churn': {
            'categories': churn_risk_data["churn_risk"].tolist(),
            'values': [int(x) for x in churn_risk_data["count"].tolist()],
            'colors': [risk_colors["High"], risk_colors["Medium"], risk_colors["Low"]]
        },
        'credit': {
            'categories': credit_risk_data["credit_risk"].tolist(),
            'values': [int(x) for x in credit_risk_data["count"].tolist()],
            'colors': [risk_colors["High"], risk_colors["Medium"], risk_colors["Low"]]
        },
        'overall': {
            'categories': overall_risk_data["overall_risk_level"].tolist(),
            'values': [int(x) for x in overall_risk_data["count"].tolist()],
            'colors': [risk_colors["High"], risk_colors["Medium"], risk_colors["Low"]]
        }
    }

    # ===== 9. SAVE RESULTS =====
    # 保存详细风险结果到Parquet文件
    output_path = "/spark/risk_warning_results.parquet"
    risk_data.write.mode("overwrite").parquet(output_path)
    print(f"Enhanced risk warning results saved to: {output_path}")

    return risk_counts, risk_chart_data