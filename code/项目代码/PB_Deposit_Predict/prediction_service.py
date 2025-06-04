# prediction_service.py
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when, lit
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, DoubleType, FloatType

model_save_path = "/spark/random_forest_model_tuned"
loaded_model = None  # 初始化为None，延迟加载

def load_model(spark):
    """加载模型，如果尚未加载"""
    global loaded_model
    if loaded_model is None:
        print(f"Loading model from {model_save_path}")
        loaded_model = RandomForestClassificationModel.load(model_save_path)
    return loaded_model

def preprocess_input_data(input_df, pipeline_model, test_data):
    """预处理输入数据"""
    try:
        if pipeline_model:
            return pipeline_model.transform(input_df)
        else:
            print("使用后备预处理方法")
            example = test_data.limit(1)
            return example.withColumn("features", example.select("features").collect()[0][0])
    except Exception as e:
        print(f"预处理失败: {str(e)}")
        return test_data.limit(1)

def predict_customer(form_data, spark, pipeline_model, test_data):
    """预测客户购买概率并生成营销建议"""
    # 确保模型已加载
    model = load_model(spark)

    # 创建输入数据的schema
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("job", StringType(), True),
        StructField("marital", StringType(), True),
        StructField("education", StringType(), True),
        StructField("default", StringType(), True),
        StructField("housing", StringType(), True),
        StructField("loan", StringType(), True),
        StructField("contact", StringType(), True),
        StructField("month", StringType(), True),
        StructField("day_of_week", StringType(), True),
        StructField("campaign", IntegerType(), True),
        StructField("pdays", IntegerType(), True),
        StructField("previous", IntegerType(), True),
        StructField("poutcome", StringType(), True),
        StructField("emp_var_rate", FloatType(), True),
        StructField("cons_price_idx", FloatType(), True),
        StructField("cons_conf_idx", FloatType(), True),
        StructField("euribor3m", FloatType(), True),
        StructField("nr_employed", FloatType(), True),
        StructField("y", StringType(), True)  # 这是目标变量的占位符
    ])

    # 从表单数据创建一个单行DataFrame
    data = [(
        int(form_data.get('age')),
        form_data.get('job'),
        form_data.get('marital'),
        form_data.get('education'),
        form_data.get('default'),
        form_data.get('housing'),
        form_data.get('loan'),
        form_data.get('contact'),
        form_data.get('month'),
        form_data.get('day_of_week'),
        int(form_data.get('campaign')),
        int(form_data.get('pdays')),
        int(form_data.get('previous')),
        form_data.get('poutcome'),
        float(form_data.get('emp_var_rate')),
        float(form_data.get('cons_price_idx')),
        float(form_data.get('cons_conf_idx')),
        float(form_data.get('euribor3m')),
        float(form_data.get('nr_employed')),
        'unknown'  # 目标变量占位符
    )]

    # 创建DataFrame
    input_df = spark.createDataFrame(data, schema)

    # 对输入数据进行预处理
    transformed_data = preprocess_input_data(input_df, pipeline_model, test_data)

    # 使用模型进行预测
    prediction_result = model.transform(transformed_data)

    # 提取预测结果
    result = prediction_result.select("prediction", "probability").collect()[0]
    prediction = result["prediction"]
    probability = float(result["probability"][1])  # 提取正类（yes）的概率

    # 确定客户潜力
    potential = "Low Potential"
    if probability > 0.8:
        potential = "High Potential"
    elif probability >= 0.5:
        potential = "Medium Potential"

    # 确定风险预警
    churn_risk = "Low Churn Risk"
    credit_risk = "Low Credit Risk"

    pdays = int(form_data.get('pdays'))
    campaign = int(form_data.get('campaign'))
    previous = int(form_data.get('previous'))
    cons_conf_idx = float(form_data.get('cons_conf_idx'))
    default = form_data.get('default')
    loan = form_data.get('loan')
    housing = form_data.get('housing')

    # 流失风险逻辑
    if potential == "Low Potential" and (pdays > 90 or campaign > 5) and previous < 2:
        churn_risk = "High Churn Risk"
    elif potential == "Medium Potential" and (pdays > 60 or campaign > 3):
        churn_risk = "Medium Churn Risk"
    elif potential == "High Potential" and pdays > 30:
        churn_risk = "Medium Churn Risk"

    # 信用风险逻辑
    if cons_conf_idx < -40 or default == "yes":
        credit_risk = "High Credit Risk"
    elif cons_conf_idx >= -40 and cons_conf_idx < -20 and (loan == "yes" or housing == "yes"):
        credit_risk = "Medium Credit Risk"

    # 基于客户特征的营销建议
    marketing_recommendation = ""
    if potential == "High Potential":
        marketing_recommendation = "高级营销活动：直接个人电话联系，提供优质的定期存款产品，降低利率门槛"
    elif potential == "Medium Potential":
        marketing_recommendation = "标准营销活动：先发送邮件，然后电话跟进，提供常规定期存款产品"
    else:
        marketing_recommendation = "基础营销活动：群发邮件提供基本定期存款信息，定期更新"

    if churn_risk == "High Churn Risk":
        marketing_recommendation += "。紧急提示：增加客户保留激励措施，如专属利率或增值服务"
    elif churn_risk == "Medium Churn Risk":
        marketing_recommendation += "。建议：包含忠诚度奖励，如期限灵活性或特殊优惠"

    if credit_risk == "High Credit Risk":
        marketing_recommendation += "。风险提示：推荐保守型金融产品，降低风险敞口"

    # 准备结果数据
    return {
        "prediction": "是" if prediction == 1.0 else "否",
        "probability": f"{probability:.2%}",
        "potential": potential,
        "churn_risk": churn_risk,
        "credit_risk": credit_risk,
        "marketing_recommendation": marketing_recommendation,
        # 添加客户信息摘要
        "customer_summary": {
            "age": form_data.get('age'),
            "job": form_data.get('job'),
            "marital": form_data.get('marital'),
            "education": form_data.get('education'),
            "housing": form_data.get('housing'),
            "loan": form_data.get('loan')
        }
    }
