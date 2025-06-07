# app.py

from flask import Flask, render_template, request, jsonify
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import traceback
import characteristic_influence
import customer_segmentation
import marketing_simulation
import risk_warning

# 初始化SparkSession
spark = SparkSession.builder \
    .appName("Bank Marketing Customer Segmentation") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 设置日志级别为ERROR以减少不必要的输出
spark.sparkContext.setLogLevel("ERROR")

# 导入prediction_service
from prediction_service import predict_customer

app = Flask(__name__)

# 模型和数据路径配置
MODEL_PATH = "/spark/random_forest_model_tuned"
PIPELINE_PATH = "/spark/pipeline_model"
TEST_DATA_PATH = "/spark/test_data.parquet"
TRAIN_DATA_PATH = "/spark/train_data.parquet"
SEGMENTATION_RESULTS_PATH = "/spark/customer_segmentation_results.parquet"

# 加载pipeline模型
try:
    pipeline_model = PipelineModel.load(PIPELINE_PATH)
    print(f"Pipeline model loaded successfully from: {PIPELINE_PATH}")
except Exception as e:
    print(f"Warning: Could not load pipeline model: {str(e)}")
    print("You need to save the pipeline model from preprocessing.py")
    pipeline_model = None

# 加载测试数据
test_data = spark.read.parquet(TEST_DATA_PATH).repartition(6)

# 执行各种分析
segment_stats_pd, segment_chart_data = customer_segmentation.perform_customer_segmentation(
    spark, MODEL_PATH, TEST_DATA_PATH)

top_features, feature_chart_data = characteristic_influence.analyze_feature_importance(
    spark, MODEL_PATH, TRAIN_DATA_PATH)

risk_counts, risk_chart_data = risk_warning.perform_risk_warning(
    spark, SEGMENTATION_RESULTS_PATH)

results_df, marketing_chart_data = marketing_simulation.perform_marketing_simulation(
    spark, MODEL_PATH, TEST_DATA_PATH)

# 页面介绍文本
intro_texts = {
    'index': '本项目巧妙运用随机森林模型，深度剖析银行营销客户数据。依据客户的购买潜力，精准地将客户划分为高、中、低三个等级，同时深入分析模型中各特征的重要性，为您清晰呈现影响客户购买决策的关键因素。此外，精心构建的风险预警系统，能够全面评估和预警客户的流失风险与信用风险，为银行营销决策提供有力支持。',
    'segment': '在这个页面，我们借助强大的随机森林模型，对银行营销客户进行细致的分层。通过精准计算客户的购买潜力，将他们明确地分为高、中、低三个等级。这有助于银行针对性地制定营销策略，提高营销效率，挖掘潜在客户价值。',
    'feature': '此页面聚焦于随机森林模型中各特征的重要性分析。我们深入研究每个特征对客户购买决策的影响程度，找出那些起着关键作用的因素。这将帮助银行更好地理解客户行为，优化营销方案，提升营销效果。',
    'risk': '这里是我们的风险预警系统展示页面。我们结合客户分层结果和多个关键指标，构建了一套全面的风险评估体系。通过该系统，能够及时准确地评估客户的流失风险和信用风险，并提供相应的风险预警，为银行的风险管理提供有力保障。',
    'showall': '本项目运用先进的随机森林模型，全方位服务于银行营销客户分析。在客户分层方面，依据购买潜力将客户分为高、中、低三个等级；在特征分析上，深入探究各特征对购买决策的影响；同时，通过精心构建的风险预警系统，对客户的流失风险和信用风险进行实时评估和预警。这里展示了项目的所有核心内容，为您提供全面、深入的银行营销客户洞察。',
    'marketing_simulation': '这里展示了营销模拟的结果，帮助银行评估不同营销策略的效果。',
    'predict': '在这个页面，您可以输入客户信息，系统将预测该客户是否会购买定期存款产品，并提供客户潜力评估、风险预警和个性化营销建议。填写表单时，可以参考每个字段的说明和推荐值范围。'
}


@app.route('/')
def index():
    return render_template('index.html',
                           segment_stats=segment_stats_pd.to_html(),
                           segment_chart_data=segment_chart_data,
                           top_features=top_features,
                           feature_chart_data=feature_chart_data,
                           risk_counts=risk_counts.to_html(),
                           risk_chart_data=risk_chart_data,
                           show_all=False,
                           show_segment=False,
                           show_feature=False,
                           show_risk=False,
                           show_marketing_simulation=False,
                           intro_text=intro_texts['index'])


@app.route('/segment')
def show_segment():
    return render_template('index.html',
                           segment_stats=segment_stats_pd.to_html(),
                           segment_chart_data=segment_chart_data,
                           top_features=top_features,
                           feature_chart_data=feature_chart_data,
                           risk_counts=risk_counts.to_html(),
                           risk_chart_data=risk_chart_data,
                           show_all=False,
                           show_segment=True,
                           show_feature=False,
                           show_risk=False,
                           show_marketing_simulation=False,
                           intro_text=intro_texts['segment'])


@app.route('/feature')
def show_feature():
    return render_template('index.html',
                           segment_stats=segment_stats_pd.to_html(),
                           segment_chart_data=segment_chart_data,
                           top_features=top_features,
                           feature_chart_data=feature_chart_data,
                           risk_counts=risk_counts.to_html(),
                           risk_chart_data=risk_chart_data,
                           show_all=False,
                           show_segment=False,
                           show_feature=True,
                           show_risk=False,
                           show_marketing_simulation=False,
                           intro_text=intro_texts['feature'])


@app.route('/risk')
def show_risk():
    return render_template('index.html',
                           segment_stats=segment_stats_pd.to_html(),
                           segment_chart_data=segment_chart_data,
                           top_features=top_features,
                           feature_chart_data=feature_chart_data,
                           risk_counts=risk_counts.to_html(),
                           risk_chart_data=risk_chart_data,
                           show_all=False,
                           show_segment=False,
                           show_feature=False,
                           show_risk=True,
                           show_marketing_simulation=False,
                           intro_text=intro_texts['risk'])


@app.route('/showall')
def show_all():
    marketing_stats = results_df.to_html()
    return render_template('index.html',
                           segment_stats=segment_stats_pd.to_html(),
                           segment_chart_data=segment_chart_data,
                           top_features=top_features,
                           feature_chart_data=feature_chart_data,
                           risk_counts=risk_counts.to_html(),
                           risk_chart_data=risk_chart_data,
                           show_all=True,
                           show_segment=False,
                           show_feature=False,
                           show_risk=False,
                           show_marketing_simulation=True,
                           intro_text=intro_texts['showall'],
                           marketing_stats=marketing_stats,
                           marketing_chart_data=marketing_chart_data)


@app.route('/marketing_simulation')
def show_marketing_simulation():
    marketing_stats = results_df.to_html()
    return render_template('index.html',
                           segment_stats=segment_stats_pd.to_html(),
                           segment_chart_data=segment_chart_data,
                           top_features=top_features,
                           feature_chart_data=feature_chart_data,
                           risk_counts=risk_counts.to_html(),
                           risk_chart_data=risk_chart_data,
                           show_all=False,
                           show_segment=False,
                           show_feature=False,
                           show_risk=False,
                           show_marketing_simulation=True,
                           marketing_stats=marketing_stats,
                           marketing_chart_data=marketing_chart_data,
                           intro_text=intro_texts['marketing_simulation'])


@app.route('/predict', methods=['GET'])
def show_prediction_form():
    return render_template('predict_form.html',
                           intro_text=intro_texts['predict'])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        result_data = predict_customer(request.form, spark, pipeline_model, test_data)
        return render_template('prediction_result.html', result=result_data)
    except Exception as e:
        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


# app.py 中的 risk_dashboard 路由修改
@app.route('/risk_dashboard')
def risk_dashboard():
    try:
        # 从风险预警结果中获取高风险客户
        risk_data = spark.read.parquet("/spark/risk_warning_results.parquet")

        # 检查是否存在 risk_priority_rank 列
        if "risk_priority_rank" in risk_data.columns:
            top_at_risk = risk_data.filter(col("risk_priority_rank") <= 50).select(
                "customer_id", "overall_risk_level", "churn_risk", "credit_risk",
                "composite_risk_score", "recommended_actions", "risk_priority_rank"
            ).orderBy("risk_priority_rank").toPandas()
        else:
            # 如果没有优先级排名，按综合风险分数排序
            top_at_risk = risk_data.select(
                "customer_id", "overall_risk_level", "churn_risk", "credit_risk",
                "composite_risk_score", "recommended_actions"
            ).orderBy(col("composite_risk_score").desc()).limit(50).toPandas()

        # 综合风险分数分布
        score_stats = risk_data.select("composite_risk_score").describe().toPandas()

        # 创建分数区间统计
        score_ranges = [
            ("0-20", risk_data.filter((col("composite_risk_score") >= 0) & (col("composite_risk_score") < 20)).count()),
            ("20-40",
             risk_data.filter((col("composite_risk_score") >= 20) & (col("composite_risk_score") < 40)).count()),
            ("40-60",
             risk_data.filter((col("composite_risk_score") >= 40) & (col("composite_risk_score") < 60)).count()),
            ("60-80",
             risk_data.filter((col("composite_risk_score") >= 60) & (col("composite_risk_score") < 80)).count()),
            ("80-100",
             risk_data.filter((col("composite_risk_score") >= 80) & (col("composite_risk_score") <= 100)).count())
        ]

        score_chart_data = {
            'bins': [x[0] for x in score_ranges],
            'counts': [x[1] for x in score_ranges]
        }

        # 创建风险相关性热力图数据
        risk_indicators = ["churn_risk_score", "credit_risk_score", "cons_conf_idx", "pdays", "campaign", "previous"]
        heatmap_data = []

        # 检查列是否存在
        available_indicators = [col_name for col_name in risk_indicators if col_name in risk_data.columns]

        for i in range(len(available_indicators)):
            for j in range(len(available_indicators)):
                try:
                    corr = float(risk_data.stat.corr(available_indicators[i], available_indicators[j]))
                    heatmap_data.append([i, j, round(corr, 2)])
                except:
                    heatmap_data.append([i, j, 0.0])

        corr_chart_data = {
            'indicators': [x.replace("_score", "").replace("_", " ") for x in available_indicators],
            'data': heatmap_data
        }

        return render_template('risk_dashboard.html',
                               top_at_risk=top_at_risk.to_html(classes='table table-striped table-hover'),
                               # 移除trend_chart_data参数
                               score_chart_data=score_chart_data,
                               corr_chart_data=corr_chart_data,
                               intro_text="高级风险预警仪表板提供全面的客户风险评估，显示流失风险和信用风险的详细分析。您可以查看风险分布、关联性分析和优先级最高的风险客户名单，以便及时采取行动。")

    except Exception as e:
        print(f"Error in risk dashboard: {str(e)}")
        return render_template('error.html', error_message=f"风险仪表板加载失败: {str(e)}")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')