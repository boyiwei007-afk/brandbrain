"""
模块5：决策模拟
- 调节价格/折扣/广告参数
- 实时模拟销量变化
- 场景对比 + 策略建议
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import traceback


# ──────────────────────────────────────────────
# 场景模拟
# ──────────────────────────────────────────────

def simulate_scenario(
    trainer,
    base_params: Dict,
    scenario_params: Dict,
    horizon: int = None,
) -> Dict:
    from config import GRANULARITY_CONFIG
    cfg = GRANULARITY_CONFIG[trainer.granularity]
    steps = horizon or cfg["forecast_horizon"]

    base_result = trainer.predict(steps=steps,
                                  exog_override=base_params if base_params else None)
    scen_result = trainer.predict(steps=steps,
                                  exog_override=scenario_params if scenario_params else None)

    base_q50 = np.array(base_result.get("q50", [0] * steps))
    scen_q50 = np.array(scen_result.get("q50", [0] * steps))
    scen_q10 = np.array(scen_result.get("q10", [0] * steps))
    scen_q90 = np.array(scen_result.get("q90", [0] * steps))

    total_base = float(base_q50.sum())
    total_scen = float(scen_q50.sum())
    lift = (total_scen - total_base) / (abs(total_base) + 1e-8) * 100

    return {
        "base_q50": base_q50.tolist(),
        "scen_q50": scen_q50.tolist(),
        "scen_q10": scen_q10.tolist(),
        "scen_q90": scen_q90.tolist(),
        "total_base": round(total_base, 1),
        "total_scen": round(total_scen, 1),
        "lift_pct": round(lift, 2),
        "steps": steps,
    }


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def plot_scenario_comparison(sim_result: Dict, granularity: str = "daily") -> go.Figure:
    steps = sim_result["steps"]
    x = list(range(1, steps + 1))
    lift = sim_result["lift_pct"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sim_result["base_q50"], name="基准方案",
                             line=dict(color="#2196F3", width=2.5), mode="lines+markers"))

    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=sim_result["scen_q90"] + sim_result["scen_q10"][::-1],
        fill="toself", fillcolor="rgba(255,87,34,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="场景P10-P90",
    ))

    color = "#4CAF50" if lift >= 0 else "#F44336"
    fig.add_trace(go.Scatter(x=x, y=sim_result["scen_q50"],
                             name=f"模拟方案（{lift:+.1f}%）",
                             line=dict(color=color, width=2.5, dash="dash"),
                             mode="lines+markers"))

    # 差值填充
    base = np.array(sim_result["base_q50"])
    scen = np.array(sim_result["scen_q50"])
    fill_color = "rgba(76,175,80,0.1)" if lift >= 0 else "rgba(244,67,54,0.1)"
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(np.maximum(base, scen)) + list(np.minimum(base, scen)[::-1]),
        fill="toself", fillcolor=fill_color,
        line=dict(color="rgba(0,0,0,0)"), name="差值区域",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_white", height=420,
        title=f"策略模拟对比（总计变化: {lift:+.1f}%）",
        xaxis_title=f"预测步 ({granularity})", yaxis_title="预测销量",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )
    return fig


def plot_roi_waterfall(params_changes: Dict, lift_pct: float, base_sales: float) -> go.Figure:
    items = [(k, v) for k, v in params_changes.items() if abs(v) > 1e-6]
    if not items:
        return go.Figure()

    delta_total = base_sales * lift_pct / 100
    n = len(items)

    fig = go.Figure(go.Waterfall(
        name="销量构成",
        orientation="v",
        measure=["absolute"] + ["relative"] * n + ["total"],
        x=["基准销量"] + [k for k, _ in items] + ["预测销量"],
        y=[base_sales] + [delta_total / n] * n + [base_sales + delta_total],
        connector=dict(line=dict(color="gray", dash="dot")),
        increasing=dict(marker=dict(color="#4CAF50")),
        decreasing=dict(marker=dict(color="#F44336")),
        totals=dict(marker=dict(color="#9C27B0")),
    ))
    fig.update_layout(template="plotly_white", height=350, title="策略效果拆解（瀑布图）",
                      yaxis_title="销量", showlegend=False)
    return fig


def plot_multi_scenario_radar(scenarios_result: Dict, base_sales: float) -> go.Figure:
    """多方案雷达图对比"""
    names = list(scenarios_result.keys())
    lifts = [v["lift_pct"] for v in scenarios_result.values()]
    totals = [v["total_scen"] for v in scenarios_result.values()]

    # 归一化 lift 到 0-1
    lift_norm = [(l - min(lifts)) / (max(lifts) - min(lifts) + 1e-9) for l in lifts]

    fig = go.Figure()
    categories = ["销量提升率", "峰值稳定性", "置信区间宽度", "综合得分"]
    for i, name in enumerate(names):
        r = scenarios_result[name]
        q50 = np.array(r["scen_q50"])
        q10 = np.array(r["scen_q10"])
        q90 = np.array(r["scen_q90"])
        stability = 1 - (q50.std() / (q50.mean() + 1e-8))
        ci_narrow = 1 - ((q90 - q10).mean() / (q50.mean() + 1e-8))
        scores = [
            max(0, lift_norm[i]),
            max(0, min(1, stability)),
            max(0, min(1, ci_narrow)),
            max(0, (lift_norm[i] + stability + ci_narrow) / 3),
        ]
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], theta=categories + [categories[0]],
            fill="toself", name=name, opacity=0.7,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      template="plotly_white", height=380, title="多方案雷达对比")
    return fig


def generate_strategy_suggestion(
    lift_pct: float, scenario_params: Dict, base_params: Dict,
    total_base: float, total_scen: float,
) -> str:
    lines = ["## 策略模拟分析报告\n"]
    lift_emoji = "📈" if lift_pct >= 0 else "📉"
    lines.append(f"{lift_emoji} **总体影响：** 预计销量变化 **{lift_pct:+.1f}%**")
    lines.append(f"- 基准总销量：{total_base:,.0f}")
    lines.append(f"- 模拟总销量：{total_scen:,.0f}")
    lines.append(f"- 绝对变化：{total_scen - total_base:+,.0f}\n")

    lines.append("### 参数变化明细")
    for col, val in scenario_params.items():
        base_val = base_params.get(col, None)
        if base_val is not None and isinstance(base_val, (int, float)):
            change = val - base_val
            change_pct_str = f"（{change / (abs(base_val) + 1e-9) * 100:+.1f}%）" if base_val != 0 else ""
            lines.append(f"- **{col}**：{base_val:.3g} → {val:.3g} {change_pct_str}")
        else:
            lines.append(f"- **{col}**：{val}")

    lines.append("\n### 💡 策略建议")
    if lift_pct > 10:
        lines.append("✅ **强烈推荐执行**：该策略组合预计带来显著销量提升，ROI较高。")
    elif lift_pct > 3:
        lines.append("✅ **建议执行**：该策略有一定提升效果，可结合成本评估后实施。")
    elif lift_pct > -3:
        lines.append("⚠️ **谨慎执行**：预期效果接近中性，建议配合其他激励手段。")
    else:
        lines.append("❌ **不建议执行**：该参数组合预计导致销量下滑，建议调整方案。")

    price_base = base_params.get("price", 0)
    price_scen = scenario_params.get("price", price_base)
    if isinstance(price_base, (int, float)) and price_scen < price_base:
        lines.append(f"- 降价 {(1 - price_scen / price_base) * 100:.1f}%，通常能刺激短期销量，注意毛利率影响")
    if scenario_params.get("is_promotion") == 1:
        lines.append("- 促销期间建议配合社交媒体推广和限时优惠提高转化率")
    if scenario_params.get("ad_spend", 0) > base_params.get("ad_spend", 0):
        lines.append("- 增加广告投入，建议重点投放节假日前1-2周（ROI最高时段）")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Streamlit渲染
# ──────────────────────────────────────────────

def render_decision_simulation():
    st.header("🎮 决策模拟")
    st.caption("调节关键营销参数，实时预测销量变化，辅助制定最优策略")

    if not st.session_state.get("training_done"):
        st.warning("请先完成「模型训练」Tab")
        return

    trainer = st.session_state["trainer"]
    df = st.session_state["preprocessed_df"] if "preprocessed_df" in st.session_state else st.session_state.get("df")
    if df is None:
        st.error("找不到数据，请重新导入")
        return
    exog_cols = st.session_state.get("exog_cols", [])
    granularity = st.session_state["granularity"]
    target_col = st.session_state["target_col"]

    # 计算各变量基准值（中位数）
    base_stats = {}
    for col in exog_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            base_stats[col] = float(df[col].median())

    st.subheader("📊 参数调节面板")
    col1, col2 = st.columns(2)
    scenario_params: Dict = {}
    base_params: Dict = {}

    with col1:
        st.markdown("**价格与折扣**")
        for col_name, label in [("price", "单价"), ("discount_rate", "折扣率（0=无折扣）"),
                                  ("competitor_price", "竞品价格")]:
            if col_name not in base_stats:
                continue
            bv = base_stats[col_name]
            if col_name == "discount_rate":
                nv = st.slider(f"{label}（基准: {bv:.2f}）",
                               min_value=0.0, max_value=0.6, value=float(bv),
                               step=0.05, format="%.2f", key=f"sim_{col_name}")
            else:
                lo, hi = bv * 0.5, bv * 1.5
                nv = st.slider(f"{label}（基准: {bv:.1f}）",
                               min_value=float(lo), max_value=float(hi), value=float(bv),
                               step=float(bv * 0.01), format="%.2f", key=f"sim_{col_name}")
            scenario_params[col_name] = nv
            base_params[col_name] = bv

    with col2:
        st.markdown("**营销与促销**")
        if "ad_spend" in base_stats:
            bv = base_stats["ad_spend"]
            nv = st.slider(f"广告投入（基准: {bv:.0f}）",
                           min_value=0.0, max_value=float(bv * 3), value=float(bv),
                           step=float(max(1, bv * 0.05)), format="%.0f", key="sim_ad_spend")
            scenario_params["ad_spend"] = nv
            base_params["ad_spend"] = bv

        for col_name, label in [("is_promotion", "启动促销活动"), ("is_holiday", "节假日期间")]:
            if col_name in base_stats:
                nv = st.toggle(label, value=False, key=f"sim_{col_name}")
                scenario_params[col_name] = int(nv)
                base_params[col_name] = 0

    from config import GRANULARITY_CONFIG
    cfg = GRANULARITY_CONFIG[granularity]
    horizon = st.slider("模拟预测步数", 1, cfg["forecast_horizon"] * 2,
                        cfg["forecast_horizon"], key="sim_horizon")

    if st.button("🚀 运行场景模拟", type="primary", use_container_width=True):
        with st.spinner("模拟中..."):
            try:
                sim_result = simulate_scenario(trainer, base_params, scenario_params, horizon)
                st.session_state["sim_result"] = sim_result
                st.session_state["sim_params"] = {"base": base_params, "scenario": scenario_params}
            except Exception as e:
                st.error(f"模拟失败: {e}")
                st.code(traceback.format_exc())
                return

    if "sim_result" not in st.session_state:
        return

    sim = st.session_state["sim_result"]
    params = st.session_state["sim_params"]
    lift = sim["lift_pct"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("基准总销量", f"{sim['total_base']:,.0f}")
    m2.metric("模拟总销量", f"{sim['total_scen']:,.0f}")
    m3.metric("销量提升", f"{lift:+.1f}%",
              delta=f"{sim['total_scen'] - sim['total_base']:+,.0f}",
              delta_color="normal" if lift >= 0 else "inverse")
    m4.metric("最近实际销量", f"{float(df[target_col].iloc[-1]):,.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 对比曲线", "💰 效果拆解", "📡 多方案雷达", "💡 策略建议"])

    with tab1:
        st.plotly_chart(plot_scenario_comparison(sim, granularity), use_container_width=True)

    with tab2:
        changes = {k: scenario_params.get(k, 0) - base_params.get(k, 0)
                   for k in scenario_params}
        fig_wf = plot_roi_waterfall(changes, lift, sim["total_base"])
        if fig_wf.data:
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("参数未发生变化")

    with tab3:
        preset_scenarios = {
            "降价10%": {**base_params, "price": base_params.get("price", 100) * 0.9},
            "广告+50%": {**base_params, "ad_spend": base_params.get("ad_spend", 500) * 1.5},
            "促销活动": {**base_params, "is_promotion": 1},
            "组合策略": {
                **base_params,
                "price": base_params.get("price", 100) * 0.95,
                "ad_spend": base_params.get("ad_spend", 500) * 1.3,
                "is_promotion": 1,
            },
            "当前方案": scenario_params,
        }
        radar_data = {}
        comparison_rows = []
        with st.spinner("批量模拟中..."):
            for name, sp in preset_scenarios.items():
                try:
                    r = simulate_scenario(trainer, base_params, sp, horizon)
                    radar_data[name] = r
                    comparison_rows.append({
                        "方案": name,
                        "预计总销量": f"{r['total_scen']:,.0f}",
                        "提升率": f"{r['lift_pct']:+.1f}%",
                        "推荐度": "⭐⭐⭐" if r["lift_pct"] > 10 else "⭐⭐" if r["lift_pct"] > 0 else "⭐",
                    })
                except Exception:
                    pass

        if radar_data:
            st.plotly_chart(plot_multi_scenario_radar(radar_data, sim["total_base"]),
                            use_container_width=True)
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    with tab4:
        suggestion = generate_strategy_suggestion(
            lift, params["scenario"], params["base"], sim["total_base"], sim["total_scen"]
        )
        st.markdown(suggestion)
