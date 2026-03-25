"""
模块1：数据导入
- CSV/Excel上传
- 表格预览
- 自动识别时间列和目标列
- 列类型推断
"""
import io
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Tuple, List, Optional


# ──────────────────────────────────────────────
# 时间列检测
# ──────────────────────────────────────────────

_TIME_KEYWORDS = ["date", "time", "datetime", "week", "month", "year", "日期", "时间", "年", "月", "周"]
_TARGET_KEYWORDS = ["sales", "sale", "revenue", "amount", "qty", "quantity", "volume",
                    "销量", "销售额", "销售", "收入", "数量", "金额"]


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    """优先用关键词匹配，其次尝试解析"""
    cols = df.columns.tolist()
    for col in cols:
        if any(kw in col.lower() for kw in _TIME_KEYWORDS):
            return col
    # 尝试解析dtype
    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # 尝试字符串解析
    for col in cols:
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].head(10))
                return col
            except Exception:
                pass
    return None


def detect_target_col(df: pd.DataFrame, time_col: str) -> Optional[str]:
    """找销量/收入目标列"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)

    for col in numeric_cols:
        if any(kw in col.lower() for kw in _TARGET_KEYWORDS):
            return col
    # 默认取第一个正数较大的数值列
    for col in numeric_cols:
        if df[col].mean() > 10:
            return col
    return numeric_cols[0] if numeric_cols else None


def infer_granularity(df: pd.DataFrame, date_col: str) -> str:
    """推断时间粒度"""
    dates = pd.to_datetime(df[date_col]).sort_values().drop_duplicates()
    if len(dates) < 2:
        return "daily"
    diffs = dates.diff().dropna().dt.days
    median_diff = diffs.median()
    if median_diff <= 1.5:
        return "daily"
    elif median_diff <= 8:
        return "weekly"
    else:
        return "monthly"


# ──────────────────────────────────────────────
# 文件读取
# ──────────────────────────────────────────────

def load_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """读取上传文件，返回(DataFrame, 文件名)"""
    name = uploaded_file.name
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="gbk")
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"不支持的文件格式: {name}")
    return df, name


def auto_parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """自动解析时间列"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    return df


# ──────────────────────────────────────────────
# Streamlit渲染
# ──────────────────────────────────────────────

def render_data_import():
    st.header("📂 数据导入")
    st.caption("支持 CSV / Excel，自动识别时间列与销量目标列")

    # ── 示例数据按钮 ──
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "上传你的销售数据文件",
            type=["csv", "xlsx", "xls"],
            help="列应包含：日期、销量、价格、广告投入等特征",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📊 使用示例数据", use_container_width=True)

    if use_sample:
        try:
            sample_path = Path(__file__).parent.parent / "data" / "sample_data.csv"
            df = pd.read_csv(sample_path)
            st.session_state["raw_df"] = df
            st.session_state["file_name"] = "sample_data.csv"
            st.success("已加载示例数据（BrandX 2021-2024 日频销售数据）")
        except FileNotFoundError:
            st.error("示例数据文件不存在，请先运行 `python data/generate_sample.py`")
            return

    if uploaded is not None:
        try:
            df, fname = load_file(uploaded)
            st.session_state["raw_df"] = df
            st.session_state["file_name"] = fname
            st.success(f"文件上传成功: {fname}")
        except Exception as e:
            st.error(f"文件读取失败: {e}")
            return

    if "raw_df" not in st.session_state:
        st.info('请上传数据文件或点击"使用示例数据"开始体验')
        return

    df = st.session_state["raw_df"]

    # ── 基本信息 ──
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("行数", f"{len(df):,}")
    m2.metric("列数", len(df.columns))
    m3.metric("数值列", df.select_dtypes(include=np.number).shape[1])
    m4.metric("缺失值", int(df.isnull().sum().sum()))

    # ── 列配置 ──
    st.markdown("### 列配置")

    # 自动检测
    auto_date = detect_time_col(df)
    auto_target = detect_target_col(df, auto_date or "")
    all_cols = df.columns.tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox(
            "⏱️ 时间列",
            all_cols,
            index=all_cols.index(auto_date) if auto_date in all_cols else 0,
        )
    with c2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target_col = st.selectbox(
            "🎯 目标列（销量）",
            numeric_cols,
            index=numeric_cols.index(auto_target) if auto_target in numeric_cols else 0,
        )
    with c3:
        auto_gran = infer_granularity(df, date_col)
        granularity = st.selectbox(
            "📅 时间粒度",
            ["daily", "weekly", "monthly"],
            index=["daily", "weekly", "monthly"].index(auto_gran),
        )

    # 外生变量选择
    st.markdown("#### 外生变量（特征）选择")
    exclude = {date_col, target_col}
    candidate_exog = [c for c in numeric_cols if c not in exclude]
    # 默认全选
    exog_cols = st.multiselect(
        "历史外生变量（模型输入特征）",
        candidate_exog,
        default=candidate_exog,
        help="这些列在历史窗口内已知，会被用于预测",
    )
    # 未来已知变量（节假日、促销计划等）
    binary_cols = [c for c in candidate_exog if df[c].nunique() <= 5]
    future_exog_cols = st.multiselect(
        "未来已知变量（节假日/促销计划）",
        candidate_exog,
        default=[c for c in binary_cols if c not in ["price", "competitor_price"]],
        help="这些变量在预测期内已知（如节假日标志、促销计划）",
    )

    # ── 保存配置 ──
    if st.button("✅ 确认配置，进入预处理", type="primary", use_container_width=True):
        try:
            df_parsed = auto_parse_dates(df, date_col)
            st.session_state.update({
                "df": df_parsed,
                "date_col": date_col,
                "target_col": target_col,
                "granularity": granularity,
                "exog_cols": exog_cols,
                "future_exog_cols": future_exog_cols,
                "config_done": True,
            })
            st.success("配置已保存！请切换到「预处理」Tab")
            st.balloons()
        except Exception as e:
            st.error(f"解析失败: {e}")

    # ── 数据预览 ──
    st.markdown("### 数据预览")
    st.dataframe(df.head(100), use_container_width=True)

    # ── 列统计 ──
    with st.expander("📊 列统计信息"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)
