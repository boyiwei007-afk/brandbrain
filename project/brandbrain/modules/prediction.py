"""
模块4：预测分析
- 生成预测曲线（P10/P50/P90）
- SHAP特征重要性（LightGBM）
- 注意力热力图可视化（BrandFormer）
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import traceback


def _safe_date_range(last_date: pd.Timestamp, horizon: int, granularity: str) -> pd.DatetimeIndex:
    """安全构造预测日期序列（避免Timestamp+offset兼容性问题）"""
    freq = {"daily": "D", "weekly": "W", "monthly": "ME"}.get(granularity, "D")
    # 生成 horizon+1 个点，取后 horizon 个（跳过起点）
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]


# ──────────────────────────────────────────────
# 预测曲线
# ──────────────────────────────────────────────

def plot_forecast(
    history_dates: pd.Series,
    history_values: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    q10: List[float],
    q50: List[float],
    q90: List[float],
    title: str = "销量预测",
) -> go.Figure:
    # 日期统一转字符串（避免Plotly对Timestamp做加法运算引起异常）
    hist_x = [str(d)[:10] for d in pd.to_datetime(history_dates)]
    fore_x = [str(d)[:10] for d in forecast_dates]

    fig = go.Figure()
    n_hist = min(180, len(history_values))

    fig.add_trace(go.Scatter(
        x=hist_x[-n_hist:], y=list(history_values.iloc[-n_hist:]),
        name="历史销量", line=dict(color="#2196F3", width=2),
    ))

    # 置信区间填充
    fig.add_trace(go.Scatter(
        x=fore_x + fore_x[::-1],
        y=list(q90) + list(q10[::-1]),
        fill="toself", fillcolor="rgba(255,87,34,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="P10-P90置信区间",
    ))

    fig.add_trace(go.Scatter(
        x=fore_x, y=q50,
        name="预测中位数(P50)", line=dict(color="#FF5722", width=2.5, dash="dash"),
        mode="lines+markers", marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=fore_x, y=q10, name="P10下界",
        line=dict(color="#4CAF50", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=fore_x, y=q90, name="P90上界",
        line=dict(color="#FF9800", width=1.5, dash="dot"),
    ))

    # 预测起点分割线（用 add_shape 避免 Plotly 对字符串日期做 sum 运算）
    if hist_x:
        fig.add_shape(
            type="line",
            x0=hist_x[-1], x1=hist_x[-1],
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="gray", dash="solid", width=1.5),
        )
        fig.add_annotation(
            x=hist_x[-1], y=1, xref="x", yref="paper",
            text="预测起点", showarrow=False,
            yanchor="bottom", font=dict(color="gray", size=11),
        )

    fig.update_layout(
        template="plotly_white", height=450, title=title,
        xaxis_title="日期", yaxis_title="销量",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )
    return fig


# ──────────────────────────────────────────────
# SHAP 可视化
# ──────────────────────────────────────────────

def compute_shap(trainer) -> Optional[Dict]:
    if trainer.lgbm_model is None:
        return None
    try:
        import shap
        lgbm_df = trainer._lgbm_df
        feat_cols = trainer._lgbm_feat_cols
        X = lgbm_df[feat_cols].values
        explainer = shap.TreeExplainer(trainer.lgbm_model)
        shap_values = explainer.shap_values(X)
        return {
            "shap_values": shap_values,
            "X": X,
            "feat_cols": feat_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    except Exception as e:
        st.warning(f"SHAP计算失败: {e}")
        return None


def plot_shap_importance(shap_result: Dict, top_n: int = 15) -> go.Figure:
    feat_cols = shap_result["feat_cols"]
    mean_shap = shap_result["mean_abs_shap"]
    idx = np.argsort(mean_shap)[-top_n:]
    feats = [feat_cols[i] for i in idx]
    vals = mean_shap[idx]
    colors = [f"rgba(229,57,53,{min(1.0, v / (max(vals) + 1e-9) + 0.3)})" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(template="plotly_white", height=max(350, top_n * 25),
                      title=f"SHAP特征重要性（Top {top_n}）",
                      xaxis_title="|SHAP值|（对销量的平均影响）")
    return fig


def plot_shap_beeswarm(shap_result: Dict, top_n: int = 10) -> go.Figure:
    shap_values = shap_result["shap_values"]
    X = shap_result["X"]
    feat_cols = shap_result["feat_cols"]
    mean_shap = shap_result["mean_abs_shap"]
    top_idx = np.argsort(mean_shap)[-top_n:][::-1]
    n_samples = min(500, len(shap_values))
    sample_idx = np.random.choice(len(shap_values), n_samples, replace=False)

    fig = go.Figure()
    for rank, feat_i in enumerate(top_idx):
        sv = shap_values[sample_idx, feat_i]
        fv = X[sample_idx, feat_i]
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        y_jitter = rank + np.random.uniform(-0.35, 0.35, len(sv))
        fig.add_trace(go.Scatter(
            x=sv, y=y_jitter, mode="markers", name=feat_cols[feat_i],
            marker=dict(color=fv_norm, colorscale="RdBu_r", size=4, opacity=0.6,
                        showscale=(rank == 0),
                        colorbar=dict(title="特征值（归一化）", len=0.5) if rank == 0 else None),
            showlegend=False,
        ))
    fig.update_layout(
        template="plotly_white", height=max(400, top_n * 40),
        title="SHAP蜂群图",
        xaxis_title="SHAP值", xaxis_zeroline=True, xaxis_zerolinecolor="gray",
        yaxis=dict(tickmode="array", tickvals=list(range(top_n)),
                   ticktext=[feat_cols[i] for i in top_idx]),
    )
    return fig


# ──────────────────────────────────────────────
# 注意力热力图
# ──────────────────────────────────────────────

def plot_attention_heatmap(attn_dict: Dict, exog_cols: List[str], patch_size: int = 7) -> List[go.Figure]:
    figs = []
    if "cross_attn" in attn_dict:
        ca = np.array(attn_dict["cross_attn"])
        if ca.ndim == 3:
            ca = ca[0]
        n_patches, n_exog_tokens = ca.shape[0], ca.shape[1] if ca.ndim > 1 else 1
        patch_labels = [f"Patch{i+1}" for i in range(n_patches)]
        exog_labels = exog_cols[:n_exog_tokens] if exog_cols else [f"Exog{i}" for i in range(n_exog_tokens)]
        fig1 = go.Figure(go.Heatmap(
            z=ca, x=exog_labels, y=patch_labels, colorscale="Viridis",
            hovertemplate="Patch: %{y}<br>外生变量: %{x}<br>权重: %{z:.4f}<extra></extra>",
        ))
        fig1.update_layout(template="plotly_white", height=350,
                           title="Cross-Attention热力图（时序Patch → 外生变量）",
                           xaxis_tickangle=-45)
        figs.append(fig1)

    if "decoder_attn" in attn_dict:
        da = np.array(attn_dict["decoder_attn"])
        if da.ndim == 3:
            da = da[0]
        horizon, n_patches = da.shape[0], da.shape[1]
        fig2 = go.Figure(go.Heatmap(
            z=da,
            x=[f"Patch{i+1}" for i in range(n_patches)],
            y=[f"预测+{i+1}" for i in range(horizon)],
            colorscale="Blues",
            hovertemplate="预测步: %{y}<br>历史Patch: %{x}<br>权重: %{z:.4f}<extra></extra>",
        ))
        fig2.update_layout(template="plotly_white", height=350,
                           title="Decoder注意力热力图（预测步 → 历史Patch）")
        figs.append(fig2)
    return figs


# ──────────────────────────────────────────────
# Streamlit渲染
# ──────────────────────────────────────────────

def render_prediction():
    st.header("🔮 预测分析")

    if not st.session_state.get("training_done"):
        st.warning("请先完成「模型训练」Tab")
        return

    trainer = st.session_state["trainer"]
    df = st.session_state["preprocessed_df"] if "preprocessed_df" in st.session_state else st.session_state.get("df")
    if df is None:
        st.error("找不到数据，请重新导入")
        return
    date_col = st.session_state["date_col"]
    target_col = st.session_state["target_col"]
    exog_cols = st.session_state["exog_cols"]
    granularity = st.session_state["granularity"]

    from config import GRANULARITY_CONFIG
    cfg = GRANULARITY_CONFIG[granularity]
    max_horizon = cfg["forecast_horizon"] * 2

    st.subheader("预测配置")
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.slider(f"预测步数（{granularity}）",
                            min_value=1, max_value=max_horizon,
                            value=cfg["forecast_horizon"])
    with col2:
        show_history = st.slider("显示历史数据量", 30, min(365, len(df)), 120)

    if st.button("🔮 生成预测", type="primary", use_container_width=True):
        with st.spinner("正在生成预测..."):
            try:
                result = trainer.predict(steps=horizon)
                last_date = pd.to_datetime(df[date_col].iloc[-1])
                forecast_dates = _safe_date_range(last_date, horizon, granularity)

                history_dates = df[date_col].iloc[-show_history:]
                history_values = df[target_col].iloc[-show_history:]

                st.session_state["last_forecast"] = {
                    "result": result,
                    "forecast_dates": forecast_dates,
                    "history_dates": history_dates,
                    "history_values": history_values,
                }

                q50 = result.get("q50", [0] * horizon)
                last_sales = float(history_values.iloc[-1])
                pred_mean = float(np.mean(q50))
                change_pct = (pred_mean - last_sales) / (abs(last_sales) + 1e-8) * 100

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("预测均值(P50)", f"{pred_mean:.0f}")
                m2.metric("最近实际值", f"{last_sales:.0f}")
                m3.metric("预测变化", f"{change_pct:+.1f}%",
                          delta_color="normal" if change_pct >= 0 else "inverse")
                m4.metric("预测步数", horizon)
                st.success("预测完成！")

            except Exception as e:
                st.error(f"预测失败: {e}")
                st.code(traceback.format_exc())
                return

    if "last_forecast" not in st.session_state:
        return

    fc = st.session_state["last_forecast"]
    result = fc["result"]
    q10 = result.get("q10", [])
    q50 = result.get("q50", [])
    q90 = result.get("q90", [])

    tab1, tab2, tab3 = st.tabs(["📈 预测曲线", "🔍 SHAP解释", "🎯 注意力热力图"])

    with tab1:
        fig = plot_forecast(
            fc["history_dates"], fc["history_values"],
            fc["forecast_dates"], q10, q50, q90,
            title=f"未来 {len(q50)} {granularity} 销量预测",
        )
        st.plotly_chart(fig, use_container_width=True)
        pred_df = pd.DataFrame({
            "日期": [str(d)[:10] for d in fc["forecast_dates"]],
            "P10（下界）": [round(v, 1) for v in q10],
            "P50（中位）": [round(v, 1) for v in q50],
            "P90（上界）": [round(v, 1) for v in q90],
        })
        with st.expander("📋 预测数据表"):
            st.dataframe(pred_df, use_container_width=True)
            st.download_button("下载预测结果", pred_df.to_csv(index=False),
                               "forecast.csv", "text/csv")

    with tab2:
        with st.spinner("计算SHAP值..."):
            shap_result = compute_shap(trainer)
        if shap_result:
            top_n = st.slider("显示Top N特征", 5, min(20, len(shap_result["feat_cols"])), 12)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_shap_importance(shap_result, top_n), use_container_width=True)
            with col2:
                st.plotly_chart(plot_shap_beeswarm(shap_result, min(top_n, 10)), use_container_width=True)
            feat_cols = shap_result["feat_cols"]
            mean_shap = shap_result["mean_abs_shap"]
            top3 = np.argsort(mean_shap)[-3:][::-1]
            st.info("**最重要的3个特征：** " +
                    " | ".join([f"**{feat_cols[i]}**（{mean_shap[i]:.2f}）" for i in top3]))
        else:
            st.warning("LightGBM未训练，无法计算SHAP")

    with tab3:
        attn = result.get("attention", {})
        if attn:
            figs = plot_attention_heatmap(attn, exog_cols, cfg["patch_size"])
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)
            st.info("**Cross-Attention**：模型对每个时间Patch关注了哪些外生变量\n\n"
                    "**Decoder Attention**：生成每个预测步时主要依赖了哪些历史时段")
        else:
            st.warning("暂无注意力数据，请点击「生成预测」")
