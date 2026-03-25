"""
BrandBrain 系统配置
"""
import os

# ===== DeepSeek API =====
try:
    import streamlit as st
    DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
except:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ===== 模型配置 =====
MODEL_SAVE_DIR = "saved_models"

# BrandFormer 默认超参数
BRANDFORMER_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "n_encoder_layers": 2,
    "d_ff": 128,
    "dropout": 0.1,
    "lstm_hidden": 64,
    "lstm_layers": 2,
    "quantiles": [0.1, 0.5, 0.9],
}

# 训练默认参数
TRAIN_CONFIG = {
    "batch_size": 32,
    "max_epochs": 100,
    "learning_rate": 1e-3,
    "patience": 10,          # 早停轮数
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "weight_decay": 1e-4,
}

# ===== 多粒度配置 =====
GRANULARITY_CONFIG = {
    "daily": {
        "patch_size": 7,          # 周级别patch
        "lookback": 60,           # 回溯60天
        "forecast_horizon": 14,   # 预测14天
        "freq": "D",
    },
    "weekly": {
        "patch_size": 4,
        "lookback": 26,           # 回溯26周
        "forecast_horizon": 8,    # 预测8周
        "freq": "W",
    },
    "monthly": {
        "patch_size": 3,
        "lookback": 24,           # 回溯24月
        "forecast_horizon": 6,    # 预测6月
        "freq": "ME",
    },
}

# ===== LightGBM 配置 =====
LGBM_CONFIG = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 20,
    "verbose": -1,
}

# ===== 可视化配置 =====
PLOT_THEME = "plotly_white"
COLOR_PALETTE = {
    "actual": "#2196F3",
    "predicted": "#FF5722",
    "upper": "#FF9800",
    "lower": "#4CAF50",
    "feature_pos": "#E53935",
    "feature_neg": "#1E88E5",
    "background": "#FAFAFA",
}

# ===== 特征工程配置 =====
TIME_FEATURES = ["day_of_week", "day_of_month", "week_of_year", "month", "quarter", "is_weekend"]
HOLIDAY_COUNTRIES = ["CN"]   # 中国节假日

# ===== Agent 工具定义 =====
AGENT_TOOLS = [
    {
        "name": "query_forecast",
        "description": "查询指定时间范围的销量预测，返回预测值和置信区间",
        "input_schema": {
            "type": "object",
            "properties": {
                "horizon": {"type": "integer", "description": "预测步长（天/周/月数）"},
                "price": {"type": "number", "description": "产品单价（可选，覆盖当前值）"},
                "ad_spend": {"type": "number", "description": "广告投入（可选）"},
            },
            "required": ["horizon"],
        },
    },
    {
        "name": "explain_prediction",
        "description": "解释最近一次预测的特征重要性，返回SHAP值分析",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "返回前N个重要特征", "default": 10},
            },
            "required": [],
        },
    },
    {
        "name": "simulate_scenario",
        "description": "模拟不同营销策略下的销量变化，支持调节价格、折扣、广告等参数",
        "input_schema": {
            "type": "object",
            "properties": {
                "price_change_pct": {"type": "number", "description": "价格变化百分比，如 -10 表示降价10%"},
                "discount_rate": {"type": "number", "description": "折扣率(0-1)，如 0.2 表示8折"},
                "ad_spend_change_pct": {"type": "number", "description": "广告费用变化百分比"},
                "has_promotion": {"type": "boolean", "description": "是否有促销活动"},
            },
            "required": [],
        },
    },
    {
        "name": "analyze_trend",
        "description": "分析销量趋势、季节性规律和异常点",
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["trend", "seasonality", "anomaly", "correlation"],
                    "description": "分析类型",
                },
            },
            "required": ["analysis_type"],
        },
    },
    {
        "name": "get_data_summary",
        "description": "获取当前数据集的统计摘要，包括基本统计量、数据质量等",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
