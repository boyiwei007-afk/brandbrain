"""
模块2：数据预处理 + 可视化分析
- 缺失值处理
- 异常值检测（IQR + Z-score）
- 特征工程（时间特征 + 节假日）
- 处理前后多视图对比（核心增强）
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────
# 缺失值处理
# ──────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame, strategy: str = "interpolate") -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    missing_before = df.isnull().sum().to_dict()
    numeric_cols = df.select_dtypes(include=np.number).columns

    if strategy == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "drop":
        df = df.dropna()

    missing_after = df.isnull().sum().to_dict()
    return df, {"before": missing_before, "after": missing_after}


# ──────────────────────────────────────────────
# 异常值检测
# ──────────────────────────────────────────────

def detect_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    if method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    elif method == "zscore":
        valid = series.dropna()
        z = np.abs(stats.zscore(valid))
        mask = pd.Series(False, index=series.index)
        mask.loc[valid.index] = z > 3
        return mask
    return pd.Series(False, index=series.index)


def handle_outliers(df: pd.DataFrame, col: str, method: str = "cap") -> pd.DataFrame:
    df = df.copy()
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    if method == "cap":
        df[col] = df[col].clip(lower, upper)
    elif method == "remove":
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


# ──────────────────────────────────────────────
# 特征工程
# ──────────────────────────────────────────────

def add_time_features(df: pd.DataFrame, date_col: str, granularity: str = "daily") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    return df


def add_holiday_features(df: pd.DataFrame, date_col: str, country: str = "CN") -> pd.DataFrame:
    df = df.copy()
    try:
        import holidays
        cn_holidays = holidays.country_holidays(country)
        dates = pd.to_datetime(df[date_col])
        df["is_public_holiday"] = dates.apply(lambda d: int(d in cn_holidays))
        df["days_to_holiday"] = dates.apply(
            lambda d: min(
                [abs((d - pd.Timestamp(h)).days) for h in cn_holidays
                 if abs((d - pd.Timestamp(h)).days) <= 14],
                default=14,
            )
        )
    except ImportError:
        df["is_public_holiday"] = 0
        df["days_to_holiday"] = 0
    return df


def engineer_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    granularity: str = "daily",
    add_holidays: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    new_cols = []
    before = set(df.columns)
    df = add_time_features(df, date_col, granularity)
    new_cols += list(set(df.columns) - before)
    if add_holidays:
        before = set(df.columns)
        df = add_holiday_features(df, date_col)
        new_cols += list(set(df.columns) - before)
    return df, new_cols


# ──────────────────────────────────────────────
# 原始数据可视化
# ──────────────────────────────────────────────

def plot_sales_overview(df: pd.DataFrame, date_col: str, target_col: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["销量时序", "30日滚动均值与波动区间"],
                        vertical_spacing=0.12, row_heights=[0.6, 0.4])
    x, y = df[date_col], df[target_col]
    fig.add_trace(go.Scatter(x=x, y=y, name="实际销量",
                             line=dict(color="#2196F3", width=1.5),
                             fill="tozeroy", fillcolor="rgba(33,150,243,0.08)"), row=1, col=1)
    from numpy.polynomial import polynomial as P
    idx = np.arange(len(y))
    c = P.polyfit(idx, y.fillna(0), 1)
    trend = P.polyval(idx, c)
    fig.add_trace(go.Scatter(x=x, y=trend, name="趋势线",
                             line=dict(color="#FF5722", dash="dash", width=2)), row=1, col=1)
    roll_mean = y.rolling(30, min_periods=1).mean()
    roll_std = y.rolling(30, min_periods=1).std().fillna(0)
    fig.add_trace(go.Scatter(x=x, y=roll_mean, name="30日均值",
                             line=dict(color="#9C27B0", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(roll_mean + roll_std) + list((roll_mean - roll_std)[::-1]),
        fill="toself", fillcolor="rgba(156,39,176,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="±1σ区间",
    ), row=2, col=1)
    fig.update_layout(template="plotly_white", height=550, title="销量总览",
                      legend=dict(orientation="h", y=-0.05))
    return fig


def plot_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    y = df[target_col].dropna()
    fig = make_subplots(rows=1, cols=3, subplot_titles=["分布直方图", "箱线图", "QQ图"])
    fig.add_trace(go.Histogram(x=y, nbinsx=50, name="频率",
                               marker_color="#2196F3", opacity=0.75), row=1, col=1)
    fig.add_trace(go.Box(y=y, name="销量", marker_color="#FF5722", boxpoints="outliers"), row=1, col=2)
    (osm, osr), (slope, intercept, r) = stats.probplot(y, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="样本",
                             marker=dict(color="#4CAF50", size=4)), row=1, col=3)
    fig.add_trace(go.Scatter(x=[osm[0], osm[-1]],
                             y=[slope * osm[0] + intercept, slope * osm[-1] + intercept],
                             name="理论正态", line=dict(color="red", dash="dash")), row=1, col=3)
    fig.update_layout(template="plotly_white", height=350, title="销量分布分析", showlegend=False)
    return fig


def plot_seasonality(df: pd.DataFrame, date_col: str, target_col: str) -> go.Figure:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["month"] = dt.dt.month
    df["dow"] = dt.dt.dayofweek
    fig = make_subplots(rows=1, cols=2, subplot_titles=["月度平均销量", "星期平均销量"])
    monthly = df.groupby("month")[target_col].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig.add_trace(go.Bar(x=[month_names[m-1] for m in monthly["month"]], y=monthly[target_col],
                         name="月均", marker_color=px.colors.sequential.Blues[2:]), row=1, col=1)
    daily = df.groupby("dow")[target_col].mean().reset_index()
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig.add_trace(go.Bar(x=[dow_names[d] for d in daily["dow"]], y=daily[target_col],
                         name="星期均值", marker_color=px.colors.sequential.Oranges[2:]), row=1, col=2)
    fig.update_layout(template="plotly_white", height=350, title="季节性分析", showlegend=False)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, exog_cols: List[str]) -> go.Figure:
    cols = [target_col] + [c for c in exog_cols if c in df.columns]
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="行: %{y}<br>列: %{x}<br>相关系数: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(template="plotly_white", height=450, title="特征相关性矩阵",
                      xaxis_tickangle=-45)
    return fig


def plot_outliers(df: pd.DataFrame, date_col: str, target_col: str) -> go.Figure:
    y = df[target_col]
    outlier_mask = detect_outliers(y, method="iqr")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col][~outlier_mask], y=y[~outlier_mask],
                             mode="lines+markers", name="正常值",
                             line=dict(color="#2196F3", width=1.5), marker=dict(size=3)))
    fig.add_trace(go.Scatter(x=df[date_col][outlier_mask], y=y[outlier_mask],
                             mode="markers", name=f"异常值 ({outlier_mask.sum()}个)",
                             marker=dict(color="red", size=10, symbol="x")))
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    fig.add_hline(y=upper, line_dash="dash", line_color="orange", annotation_text=f"上限 {upper:.0f}")
    fig.add_hline(y=lower, line_dash="dash", line_color="orange", annotation_text=f"下限 {lower:.0f}")
    fig.update_layout(template="plotly_white", height=400, title="异常值检测（IQR方法）")
    return fig


def plot_decompose(df: pd.DataFrame, date_col: str, target_col: str, period: int = 7) -> go.Figure:
    from scipy.ndimage import uniform_filter1d
    y = df[target_col].fillna(method="ffill").values.astype(float)
    trend = uniform_filter1d(y, size=period)
    seasonal_raw = y - trend
    n = len(y)
    seasonal = np.zeros(n)
    for i in range(period):
        idxs = list(range(i, n, period))
        seasonal[idxs] = np.mean(seasonal_raw[idxs])
    residual = y - trend - seasonal
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=["原始", "趋势", "季节性", "残差"],
                        vertical_spacing=0.08)
    x = df[date_col]
    for row, (data, color, name) in enumerate(zip(
        [y, trend, seasonal, residual],
        ["#2196F3", "#FF5722", "#9C27B0", "#4CAF50"], ["原始", "趋势", "季节性", "残差"],
    ), start=1):
        fig.add_trace(go.Scatter(x=x, y=data, name=name,
                                 line=dict(color=color, width=1.2)), row=row, col=1)
    fig.update_layout(template="plotly_white", height=700,
                      title=f"时序分解（周期={period}）", showlegend=False)
    return fig


# ──────────────────────────────────────────────
# 处理前后对比（核心新增）
# ──────────────────────────────────────────────

def plot_before_after_timeseries(df_before: pd.DataFrame, df_after: pd.DataFrame,
                                  date_col: str, col: str) -> go.Figure:
    """时序叠加对比"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_before[date_col], y=df_before[col],
                             name="处理前", line=dict(color="#E53935", width=1.5, dash="dot"),
                             opacity=0.7))
    fig.add_trace(go.Scatter(x=df_after[date_col], y=df_after[col],
                             name="处理后", line=dict(color="#1E88E5", width=2)))
    fig.update_layout(template="plotly_white", height=380,
                      title=f"时序对比：{col}",
                      xaxis_title="日期", yaxis_title=col,
                      legend=dict(orientation="h", y=-0.15))
    return fig


def plot_before_after_distribution(df_before: pd.DataFrame, df_after: pd.DataFrame,
                                    col: str, viz_type: str = "histogram") -> go.Figure:
    """分布对比"""
    y_before = df_before[col].dropna()
    y_after = df_after[col].dropna()

    if viz_type == "histogram":
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y_before, name="处理前", opacity=0.6,
                                   nbinsx=40, marker_color="#E53935"))
        fig.add_trace(go.Histogram(x=y_after, name="处理后", opacity=0.6,
                                   nbinsx=40, marker_color="#1E88E5"))
        fig.update_layout(barmode="overlay", template="plotly_white", height=380,
                          title=f"分布对比（直方图）：{col}")

    elif viz_type == "box":
        fig = go.Figure()
        fig.add_trace(go.Box(y=y_before, name="处理前", marker_color="#E53935",
                             boxpoints="outliers"))
        fig.add_trace(go.Box(y=y_after, name="处理后", marker_color="#1E88E5",
                             boxpoints="outliers"))
        fig.update_layout(template="plotly_white", height=380,
                          title=f"箱线图对比：{col}")

    elif viz_type == "violin":
        fig = go.Figure()
        fig.add_trace(go.Violin(y=y_before, name="处理前",
                                box_visible=True, line_color="#E53935", fillcolor="rgba(229,57,53,0.3)"))
        fig.add_trace(go.Violin(y=y_after, name="处理后",
                                box_visible=True, line_color="#1E88E5", fillcolor="rgba(30,136,229,0.3)"))
        fig.update_layout(template="plotly_white", height=380,
                          title=f"小提琴图对比：{col}")

    elif viz_type == "kde":
        from scipy.stats import gaussian_kde
        fig = go.Figure()
        for y_data, name, color in [
            (y_before, "处理前", "#E53935"),
            (y_after, "处理后", "#1E88E5"),
        ]:
            if len(y_data) > 1:
                kde = gaussian_kde(y_data)
                x_range = np.linspace(y_data.min(), y_data.max(), 300)
                fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), name=name,
                                         line=dict(color=color, width=2),
                                         fill="tozeroy",
                                         fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.15)"))
        fig.update_layout(template="plotly_white", height=380,
                          title=f"KDE密度对比：{col}")

    return fig


def render_stats_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame,
                             numeric_cols: List[str]) -> None:
    """统计量对比表"""
    rows = []
    for col in numeric_cols:
        if col not in df_before.columns or col not in df_after.columns:
            continue
        b, a = df_before[col], df_after[col]
        rows.append({
            "列名": col,
            "缺失前": int(b.isnull().sum()),
            "缺失后": int(a.isnull().sum()),
            "均值前": round(float(b.mean()), 3),
            "均值后": round(float(a.mean()), 3),
            "标准差前": round(float(b.std()), 3),
            "标准差后": round(float(a.std()), 3),
            "最小值前": round(float(b.min()), 3),
            "最小值后": round(float(a.min()), 3),
            "最大值前": round(float(b.max()), 3),
            "最大值后": round(float(a.max()), 3),
        })
    if rows:
        cmp_df = pd.DataFrame(rows)
        # 高亮缺失值变化
        def highlight_changes(col):
            if col.name.endswith("后") and col.name.replace("后", "前") in cmp_df.columns:
                before_col = cmp_df[col.name.replace("后", "前")]
                return ["background-color: #e8f5e9" if v != b else "" for v, b in zip(col, before_col)]
            return [""] * len(col)
        st.dataframe(cmp_df.style.apply(highlight_changes), use_container_width=True)


def plot_missing_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame) -> go.Figure:
    """缺失值前后对比"""
    miss_before = df_before.isnull().sum()
    miss_after = df_after.isnull().sum()
    cols_with_missing = miss_before[miss_before > 0].index.tolist()
    if not cols_with_missing:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cols_with_missing,
                         y=[miss_before[c] for c in cols_with_missing],
                         name="处理前", marker_color="#E53935", opacity=0.8))
    fig.add_trace(go.Bar(x=cols_with_missing,
                         y=[miss_after.get(c, 0) for c in cols_with_missing],
                         name="处理后", marker_color="#1E88E5", opacity=0.8))
    fig.update_layout(barmode="group", template="plotly_white", height=350,
                      title="缺失值处理对比", yaxis_title="缺失值数量",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def plot_new_features_correlation(df_after: pd.DataFrame, target_col: str,
                                   new_feats: List[str]) -> go.Figure:
    """新增特征与目标列的相关性"""
    avail = [c for c in new_feats if c in df_after.columns and
             pd.api.types.is_numeric_dtype(df_after[c])][:12]
    if not avail:
        return None
    corrs = [df_after[c].corr(df_after[target_col]) for c in avail]
    colors = ["#4CAF50" if c > 0 else "#F44336" for c in corrs]
    fig = go.Figure(go.Bar(
        x=corrs, y=avail, orientation="h",
        marker_color=colors,
        text=[f"{c:.3f}" for c in corrs],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="gray", line_dash="dash")
    fig.update_layout(template="plotly_white", height=max(300, len(avail) * 28),
                      title=f"新增特征与 {target_col} 的相关系数",
                      xaxis_title="Pearson相关系数")
    return fig


# ──────────────────────────────────────────────
# Streamlit渲染
# ──────────────────────────────────────────────

def render_preprocessing():
    st.header("🔧 数据预处理与可视化分析")

    if "df" not in st.session_state or not st.session_state.get("config_done"):
        st.warning("请先在「数据导入」Tab完成数据配置")
        return

    df = st.session_state["df"].copy()
    date_col = st.session_state["date_col"]
    target_col = st.session_state["target_col"]
    exog_cols = st.session_state["exog_cols"]
    granularity = st.session_state["granularity"]

    # 记录原始数据
    if "original_df" not in st.session_state:
        st.session_state["original_df"] = df.copy()
    df_original = st.session_state["original_df"]

    # ── 缺失值处理 ──
    st.subheader("1️⃣ 缺失值处理")
    missing_total = int(df.isnull().sum().sum())
    col1, col2 = st.columns([2, 1])
    with col1:
        strategy = st.radio(
            "处理策略",
            ["interpolate", "mean", "median", "drop"],
            format_func=lambda x: {
                "interpolate": "时序插值（推荐）",
                "mean": "均值填充",
                "median": "中位数填充",
                "drop": "删除含缺失行",
            }[x],
            horizontal=True,
        )
    with col2:
        st.metric("当前缺失值", missing_total)

    df, missing_info = handle_missing_values(df, strategy)
    if missing_total > 0:
        st.success(f"处理完毕：{missing_total} 个缺失值 → {int(df.isnull().sum().sum())} 个")

    # ── 异常值 ──
    st.subheader("2️⃣ 异常值检测与处理")
    outlier_mask = detect_outliers(df[target_col], method="iqr")
    n_outlier = int(outlier_mask.sum())
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.plotly_chart(plot_outliers(df, date_col, target_col), use_container_width=True)
    with col2:
        st.metric("检测到异常值", n_outlier)
        outlier_method = st.selectbox("处理方式", ["cap（截断）", "保留不处理"])
    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if n_outlier > 0 and outlier_method == "cap（截断）":
            if st.button("应用截断处理"):
                df = handle_outliers(df, target_col, "cap")
                st.success(f"已截断 {n_outlier} 个异常值")

    # ── 特征工程 ──
    st.subheader("3️⃣ 特征工程")
    col1, col2 = st.columns(2)
    with col1:
        add_time = st.checkbox("✅ 添加时间特征（星期/月份/季度/正弦编码）", value=True)
    with col2:
        add_hol = st.checkbox("✅ 添加中国节假日特征", value=True)

    new_feats = []
    if add_time or add_hol:
        df, new_feats = engineer_features(df, date_col, target_col, granularity, add_hol)
        st.success(
            f"新增 {len(new_feats)} 个时间/节假日特征：{', '.join(new_feats[:8])}"
            f"{'...' if len(new_feats) > 8 else ''}"
        )

    # ── 可视化分析（原始数据）──
    st.markdown("---")
    st.subheader("📊 数据可视化分析")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 时序总览", "📊 分布分析", "🌊 季节性", "🔥 相关性", "🔀 分解分析"
    ])
    with tab1:
        st.plotly_chart(plot_sales_overview(df, date_col, target_col), use_container_width=True)
        if exog_cols:
            sel = st.selectbox("选择外生变量与销量对比", exog_cols, key="exog_compare")
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(x=df[date_col], y=df[target_col], name="销量",
                                      line=dict(color="#2196F3")), secondary_y=False)
            fig2.add_trace(go.Scatter(x=df[date_col], y=df[sel], name=sel,
                                      line=dict(color="#FF5722", dash="dot")), secondary_y=True)
            fig2.update_layout(template="plotly_white", height=360, title=f"销量 vs {sel}")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.plotly_chart(plot_distribution(df, target_col), use_container_width=True)
        desc = df[target_col].describe()
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("均值", f"{desc['mean']:.0f}")
        mc2.metric("中位数", f"{desc['50%']:.0f}")
        mc3.metric("标准差", f"{desc['std']:.0f}")
        mc4.metric("最大值", f"{desc['max']:.0f}")
        mc5.metric("最小值", f"{desc['min']:.0f}")

    with tab3:
        st.plotly_chart(plot_seasonality(df, date_col, target_col), use_container_width=True)

    with tab4:
        all_exog = exog_cols + [c for c in new_feats if c in df.columns and
                                 pd.api.types.is_numeric_dtype(df[c])][:6]
        st.plotly_chart(plot_correlation_heatmap(df, target_col, all_exog), use_container_width=True)

    with tab5:
        period_map = {"daily": 7, "weekly": 4, "monthly": 12}
        period = period_map.get(granularity, 7)
        st.plotly_chart(plot_decompose(df, date_col, target_col, period), use_container_width=True)

    # ── 处理前后对比（核心新增）──
    st.markdown("---")
    st.subheader("🔄 处理前后数据对比分析")

    numeric_cols_common = [c for c in df_original.select_dtypes(include=np.number).columns
                           if c in df.columns]

    # 选择对比列 + 可视化方法
    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        compare_col = st.selectbox(
            "选择对比列",
            [target_col] + [c for c in numeric_cols_common if c != target_col],
            key="compare_col_select",
        )
    with ctrl2:
        viz_method = st.radio(
            "可视化方法",
            ["时序叠加", "直方图", "箱线图", "小提琴图", "KDE密度"],
            horizontal=True,
            key="viz_method_select",
        )

    method_map = {"时序叠加": "timeseries", "直方图": "histogram",
                  "箱线图": "box", "小提琴图": "violin", "KDE密度": "kde"}
    method_key = method_map[viz_method]

    if method_key == "timeseries":
        df_orig_for_compare = df_original.copy()
        df_orig_for_compare[date_col] = pd.to_datetime(df_orig_for_compare[date_col])
        st.plotly_chart(
            plot_before_after_timeseries(df_orig_for_compare, df, date_col, compare_col),
            use_container_width=True,
        )
    else:
        # 分布对比
        df_orig_aligned = df_original.reindex(df.index)
        st.plotly_chart(
            plot_before_after_distribution(df_orig_aligned, df, compare_col, method_key),
            use_container_width=True,
        )

    # 子Tab：统计对比 / 缺失值 / 新增特征
    cmp_tab1, cmp_tab2, cmp_tab3 = st.tabs(["📋 统计量对比", "🔴 缺失值对比", "✨ 新增特征相关性"])
    with cmp_tab1:
        render_stats_comparison(df_original, df, numeric_cols_common[:15])
    with cmp_tab2:
        fig_miss = plot_missing_comparison(df_original, df)
        if fig_miss:
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.info("原始数据无缺失值，无需对比")
    with cmp_tab3:
        if new_feats:
            fig_fc = plot_new_features_correlation(df, target_col, new_feats)
            if fig_fc:
                st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("尚未添加新特征")

    # ── 保存处理结果 ──
    st.markdown("---")
    all_new_exog = list(set(exog_cols + new_feats))
    all_new_exog = [c for c in all_new_exog if c in df.columns and
                    pd.api.types.is_numeric_dtype(df[c]) and c != target_col]

    if st.button("💾 保存处理结果，进入模型训练", type="primary", use_container_width=True):
        st.session_state["preprocessed_df"] = df
        st.session_state["exog_cols"] = all_new_exog
        st.session_state["future_exog_cols"] = [
            c for c in st.session_state.get("future_exog_cols", []) if c in df.columns
        ]
        st.session_state["preprocessing_done"] = True
        st.success("预处理完成！请切换到「模型训练」Tab")
