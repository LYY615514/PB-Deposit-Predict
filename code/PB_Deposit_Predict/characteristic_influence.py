# characteristic_influence.py

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml import PipelineModel
import numpy as np
import json


def analyze_feature_importance(spark, model_save_path, train_data_path):
    # 加载已训练的随机森林模型
    loaded_model = RandomForestClassificationModel.load(model_save_path)
    print(f"Model loaded from: {model_save_path}")

    # 获取特征重要性数组
    importances = loaded_model.featureImportances.toArray()
    print("Feature Importances (raw):", importances)

    # 尝试加载pipeline模型以获取准确的特征名称
    try:
        pipeline_model = PipelineModel.load("/spark/pipeline_model")

        # 加载一个样本数据来提取特征列名
        sample_data = spark.read.parquet(train_data_path).limit(1)
        transformed_data = pipeline_model.transform(sample_data)

        # 从VectorAssembler的metadata中提取特征名称
        feature_names = []
        if "features" in transformed_data.schema:
            metadata = transformed_data.schema["features"].metadata
            if "ml_attr" in metadata:
                attrs = metadata["ml_attr"]["attrs"]
                # 收集所有特征名称（数值和分类）
                for attr_type in attrs:
                    for attr in attrs[attr_type]:
                        feature_names.append(attr["name"])

        if len(feature_names) == len(importances):
            print(f"Successfully extracted {len(feature_names)} feature names from pipeline metadata")
            all_feature_names = feature_names
        else:
            raise ValueError(f"Extracted {len(feature_names)} feature names but model has {len(importances)} features")

    except Exception as e:
        print(f"Could not extract feature names from pipeline: {str(e)}")
        print("Falling back to manual feature name mapping...")

        # 定义分类列和数值列
        categorical_cols = ["job", "marital", "education", "default", "housing", "loan",
                            "contact", "month", "day_of_week", "poutcome"]
        numeric_cols = ["age", "campaign", "pdays", "previous",
                        "emp_var_rate", "cons_price_idx", "cons_conf_idx",
                        "euribor3m", "nr_employed"]

        # 尝试从原始数据中获取分类特征的唯一值
        try:
            original_data_path = "/spark/bank-additional-full.csv"
            original_data = spark.read.csv(original_data_path, header=True, inferSchema=True, sep=";", quote='"')

            # 创建一个空列表来存储所有特征名称
            all_feature_names = []

            # 处理分类列的所有可能值
            for col in categorical_cols:
                col_replaced = col.replace(".", "_")  # 替换点号为下划线，与预处理保持一致
                unique_values = original_data.select(col_replaced).distinct().rdd.flatMap(lambda x: x).collect()
                for value in unique_values:
                    if value is not None and value != "unknown":  # 排除None和"unknown"
                        all_feature_names.append(f"{col}_{value}")

            # 添加数值列
            all_feature_names.extend(numeric_cols)

            print(f"Generated {len(all_feature_names)} feature names from original data")
        except Exception as e:
            print(f"Error reading original data: {str(e)}")
            # 如果无法读取原始数据，使用一个简单的特征名列表
            all_feature_names = [f"feature_{i}" for i in range(len(importances))]
            print(f"Using generic feature names: {len(all_feature_names)} features")

    # 打印特征名称映射
    print("\nMapped feature names:")
    feature_mapping = {}
    for i, name in enumerate(all_feature_names):
        if i < len(importances):  # 确保不会越界
            feature_mapping[i] = name
            print(f"{i}: {name}")

    # 确保特征数量匹配
    if len(importances) != len(all_feature_names):
        print(
            f"Warning: Mismatch between number of features ({len(all_feature_names)}) and feature importances ({len(importances)}).")

        # 如果特征名称不足，用通用名称填充
        if len(all_feature_names) < len(importances):
            original_count = len(all_feature_names)
            for i in range(original_count, len(importances)):
                feature_name = f"feature_{i}"
                all_feature_names.append(feature_name)
                feature_mapping[i] = feature_name
                print(f"{i}: {feature_name} (auto-generated)")
        else:
            # 如果特征名称过多，截断
            all_feature_names = all_feature_names[:len(importances)]

    # 排序并输出 Top N 最重要的特征
    indices = np.argsort(importances)[::-1]  # 按重要性降序排列索引
    sorted_importances = importances[indices]
    sorted_feature_names = [all_feature_names[i] for i in indices]

    # 打印前 15 个最重要特征
    top_n = min(15, len(sorted_importances))
    top_features = []
    for i in range(top_n):
        top_features.append(f"{i + 1}. {sorted_feature_names[i]}: {sorted_importances[i]:.4f}")

    # 为ECharts准备数据
    feature_chart_data = {
        'features': sorted_feature_names[:top_n],
        'importances': [float(x) for x in sorted_importances[:top_n].tolist()]  # 确保是浮点数
    }

    return top_features, feature_chart_data