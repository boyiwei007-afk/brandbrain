"""
模块6：智能对话 Agent
- Claude claude-sonnet-4-6 Tool Use
- 5个工具：预测/解释/模拟/趋势/数据摘要
- 文字 + 图表联合回复
"""
import json
import traceback
import openai
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Optional, List, Tuple

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, AGENT_TOOLS


def _safe_date_range(last_date: pd.Timestamp, horizon: int, granularity: str) -> pd.DatetimeIndex:
    freq = {"daily": "D", "weekly": "W", "monthly": "ME"}.get(granularity, "D")
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]


# ──────────────────────────────────────────────
# Tool执行器
# ──────────────────────────────────────────────

class AgentToolExecutor:
    def __init__(self, trainer, df: pd.DataFrame, date_col: str,
                 target_col: str, exog_cols: list, granularity: str):
        self.trainer = trainer
        self.df = df
        self.date_col = date_col
        self.target_col = target_col
        self.exog_cols = exog_cols
        self.granularity = granularity

    def execute(self, tool_name: str, tool_input: dict) -> Dict:
        try:
            if tool_name == "query_forecast":
                return self._query_forecast(**tool_input)
            elif tool_name == "explain_prediction":
                return self._explain_prediction(**tool_input)
            elif tool_name == "simulate_scenario":
                return self._simulate_scenario(**tool_input)
            elif tool_name == "analyze_trend":
                return self._analyze_trend(**tool_input)
            elif tool_name == "get_data_summary":
                return self._get_data_summary()
            else:
                return {"error": f"未知工具: {tool_name}"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()[:500]}

    def _query_forecast(self, horizon: int = 14, price: float = None,
                        ad_spend: float = None, **kwargs) -> Dict:
        from config import GRANULARITY_CONFIG
        cfg = GRANULARITY_CONFIG[self.granularity]
        steps = min(int(horizon), cfg["forecast_horizon"] * 2)

        override = {}
        if price is not None:
            override["price"] = float(price)
        if ad_spend is not None:
            override["ad_spend"] = float(ad_spend)

        result = self.trainer.predict(steps=steps, exog_override=override if override else None)
        q50 = result.get("q50", [])
        q10 = result.get("q10", [])
        q90 = result.get("q90", [])

        last_date = pd.to_datetime(self.df[self.date_col].iloc[-1])
        dates = _safe_date_range(last_date, steps, self.granularity)

        return {
            "status": "success",
            "horizon": steps,
            "granularity": self.granularity,
            "dates": [str(d.date()) for d in dates],
            "q10": [round(v, 1) for v in q10[:steps]],
            "q50": [round(v, 1) for v in q50[:steps]],
            "q90": [round(v, 1) for v in q90[:steps]],
            "mean_forecast": round(float(np.mean(q50[:steps])), 1),
            "total_forecast": round(float(np.sum(q50[:steps])), 1),
            "last_actual": round(float(self.df[self.target_col].iloc[-1]), 1),
        }

    def _explain_prediction(self, top_n: int = 10, **kwargs) -> Dict:
        if self.trainer.lgbm_model is None:
            return {"error": "LightGBM未训练，无法计算SHAP"}
        try:
            import shap
            feat_cols = self.trainer._lgbm_feat_cols
            X = self.trainer._lgbm_df[feat_cols].values
            explainer = shap.TreeExplainer(self.trainer.lgbm_model)
            shap_values = explainer.shap_values(X)
            mean_shap = np.abs(shap_values).mean(axis=0)
            top_n = min(int(top_n), len(feat_cols))
            idx = np.argsort(mean_shap)[-top_n:][::-1]
            return {
                "status": "success",
                "top_features": [
                    {"feature": feat_cols[i], "importance": round(float(mean_shap[i]), 4)}
                    for i in idx
                ],
                "explanation": (
                    f"最重要特征是 {feat_cols[idx[0]]}（SHAP={mean_shap[idx[0]]:.3f}），"
                    f"其次是 {feat_cols[idx[1]]}（SHAP={mean_shap[idx[1]]:.3f}）"
                    if len(idx) > 1 else f"最重要特征是 {feat_cols[idx[0]]}"
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    def _simulate_scenario(
        self,
        price_change_pct: float = 0,
        discount_rate: float = None,
        ad_spend_change_pct: float = 0,
        has_promotion: bool = False,
        **kwargs,
    ) -> Dict:
        base: Dict = {}
        scenario: Dict = {}
        for col in self.exog_cols:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                v = float(self.df[col].median())
                base[col] = v
                scenario[col] = v

        if "price" in base and price_change_pct != 0:
            scenario["price"] = base["price"] * (1 + float(price_change_pct) / 100)
        if discount_rate is not None and "discount_rate" in scenario:
            scenario["discount_rate"] = float(discount_rate)
        if "ad_spend" in base and ad_spend_change_pct != 0:
            scenario["ad_spend"] = base["ad_spend"] * (1 + float(ad_spend_change_pct) / 100)
        if has_promotion and "is_promotion" in scenario:
            scenario["is_promotion"] = 1

        from modules.decision_sim import simulate_scenario
        result = simulate_scenario(self.trainer, base, scenario)
        return {
            "status": "success",
            "base_total": result["total_base"],
            "scenario_total": result["total_scen"],
            "lift_pct": result["lift_pct"],
            "recommendation": (
                "强烈推荐" if result["lift_pct"] > 10
                else "建议考虑" if result["lift_pct"] > 0
                else "不建议执行"
            ),
            "applied_changes": {
                "price_change_pct": price_change_pct,
                "discount_rate": discount_rate,
                "ad_spend_change_pct": ad_spend_change_pct,
                "has_promotion": has_promotion,
            },
        }

    def _analyze_trend(self, analysis_type: str = "trend", **kwargs) -> Dict:
        y = self.df[self.target_col].values
        dates = pd.to_datetime(self.df[self.date_col])

        if analysis_type == "trend":
            from numpy.polynomial import polynomial as P
            idx = np.arange(len(y))
            c = P.polyfit(idx, y, 1)
            slope = float(c[1])
            return {
                "status": "success",
                "trend_direction": "上升" if slope > 0 else "下降",
                "slope_per_step": round(slope, 3),
                "total_change_pct": round((y[-1] - y[0]) / (y[0] + 1e-8) * 100, 2),
                "mean_sales": round(float(y.mean()), 1),
                "max_sales": round(float(y.max()), 1),
                "min_sales": round(float(y.min()), 1),
            }
        elif analysis_type == "seasonality":
            df_tmp = self.df.copy()
            df_tmp["month"] = pd.to_datetime(df_tmp[self.date_col]).dt.month
            monthly = df_tmp.groupby("month")[self.target_col].mean()
            return {
                "status": "success",
                "peak_month": int(monthly.idxmax()),
                "low_month": int(monthly.idxmin()),
                "monthly_averages": {str(k): round(v, 1) for k, v in monthly.items()},
            }
        elif analysis_type == "anomaly":
            from scipy import stats as sp_stats
            z_scores = np.abs(sp_stats.zscore(y))
            anomaly_idx = np.where(z_scores > 3)[0]
            return {
                "status": "success",
                "n_anomalies": len(anomaly_idx),
                "anomaly_dates": [str(dates.iloc[i].date()) for i in anomaly_idx[:10]],
                "anomaly_values": [round(float(y[i]), 1) for i in anomaly_idx[:10]],
            }
        elif analysis_type == "correlation":
            corr = {}
            for col in self.exog_cols[:10]:
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                    c = float(self.df[col].corr(self.df[self.target_col]))
                    if not np.isnan(c):
                        corr[col] = round(c, 3)
            return {"status": "success",
                    "correlations": dict(sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True))}
        return {"error": f"未知分析类型: {analysis_type}"}

    def _get_data_summary(self) -> Dict:
        y = self.df[self.target_col]
        return {
            "status": "success",
            "n_rows": len(self.df),
            "n_features": len(self.exog_cols),
            "date_range": {
                "start": str(pd.to_datetime(self.df[self.date_col]).min().date()),
                "end": str(pd.to_datetime(self.df[self.date_col]).max().date()),
            },
            "target_stats": {
                "mean": round(float(y.mean()), 1),
                "std": round(float(y.std()), 1),
                "min": round(float(y.min()), 1),
                "max": round(float(y.max()), 1),
                "median": round(float(y.median()), 1),
            },
            "missing_values": int(self.df.isnull().sum().sum()),
            "granularity": self.granularity,
            "exog_features": self.exog_cols[:15],
        }


# ──────────────────────────────────────────────
# Agent主循环
# ──────────────────────────────────────────────

def run_agent(
    user_message: str,
    executor: AgentToolExecutor,
    history: list,
) -> Tuple[str, list, list]:
    client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    system_prompt = """你是BrandBrain品牌销量分析AI助手，帮助品牌经理深入分析销售数据、预测趋势、模拟策略决策。

可用工具：
- query_forecast：查询未来销量预测（返回P10/P50/P90）
- explain_prediction：解释预测的特征重要性（SHAP值分析）
- simulate_scenario：模拟不同价格/促销/广告策略效果
- analyze_trend：分析销量趋势、季节性规律、异常点、特征相关性
- get_data_summary：获取数据集统计摘要

回答规范：
1. 优先调用工具获取真实数据，再给出有数据支撑的结论
2. 结论具体可操作，避免空泛建议
3. 使用Markdown格式，关键数字加粗
4. 数据问题可以不调工具直接回答"""

    # 转换 tools 为 OpenAI 格式
    openai_tools = []
    for tool in AGENT_TOOLS:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
        })

    # 构建消息（system 放在第一条）
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        if isinstance(m.get("content"), str):
            messages.append(m)
    messages.append({"role": "user", "content": user_message})

    response_text = ""
    tool_results_for_display = []

    for _ in range(6):  # 最多6轮tool调用
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=openai_tools,
            max_tokens=2048,
        )

        msg = response.choices[0].message
        if msg.content:
            response_text += msg.content

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                tool_input = json.loads(tc.function.arguments)
                tool_result = executor.execute(tc.function.name, tool_input)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })
                tool_results_for_display.append({
                    "tool": tc.function.name,
                    "input": tool_input,
                    "result": tool_result,
                })
        else:
            break

    new_history = history[-28:] + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response_text},
    ]
    return response_text, new_history, tool_results_for_display


# ──────────────────────────────────────────────
# Streamlit渲染
# ──────────────────────────────────────────────

def render_agent_dialog():
    st.header("💬 智能对话")
    st.caption(f"由 {DEEPSEEK_MODEL} 驱动，自然语言查询预测、策略建议、数据洞察")

    has_trainer = st.session_state.get("training_done", False)
    if not has_trainer:
        st.warning("模型未训练，部分工具不可用（可体验基础对话）")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []

    # 快捷问题
    st.markdown("**💡 快捷提问：**")
    quick_cols = st.columns(4)
    quick_qs = ["预测未来14天销量", "分析销量趋势和季节性", "降价10%能提升多少销量", "哪些特征对销量影响最大"]
    for i, (col, q) in enumerate(zip(quick_cols, quick_qs)):
        with col:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state["pending_message"] = q
    st.markdown("---")

    # 历史消息
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("charts"):
                for fig in msg["charts"]:
                    st.plotly_chart(fig, use_container_width=True)
            if msg.get("tool_calls"):
                with st.expander("🔧 工具调用详情"):
                    for tc in msg["tool_calls"]:
                        st.json({"工具": tc["tool"], "输入": tc["input"],
                                 "结果": str(tc["result"])[:300]})

    # 输入
    pending = st.session_state.pop("pending_message", None)
    user_input = st.chat_input("输入你的问题，如：下周销量预测？促销活动能带来多少增量？") or pending

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    charts = []
                    if has_trainer:
                        executor = AgentToolExecutor(
                            trainer=st.session_state["trainer"],
                            df=st.session_state["preprocessed_df"] if "preprocessed_df" in st.session_state else st.session_state.get("df"),
                            date_col=st.session_state["date_col"],
                            target_col=st.session_state["target_col"],
                            exog_cols=st.session_state.get("exog_cols", []),
                            granularity=st.session_state["granularity"],
                        )
                        response, new_history, tool_calls = run_agent(
                            user_input, executor, st.session_state["agent_history"]
                        )
                        st.session_state["agent_history"] = new_history

                        # 预测结果图表
                        for tc in tool_calls:
                            if tc["tool"] == "query_forecast" and tc["result"].get("status") == "success":
                                r = tc["result"]
                                df = st.session_state["preprocessed_df"] if "preprocessed_df" in st.session_state else st.session_state.get("df")
                                fig = go.Figure()
                                if df is not None:
                                    tail = df.tail(60)
                                    date_col = st.session_state["date_col"]
                                    target_col = st.session_state["target_col"]
                                    fig.add_trace(go.Scatter(
                                        x=[str(d)[:10] for d in pd.to_datetime(tail[date_col])],
                                        y=list(tail[target_col]),
                                        name="历史", line=dict(color="#2196F3"),
                                    ))
                                fore_x = r["dates"]
                                if r.get("q10") and r.get("q90"):
                                    fig.add_trace(go.Scatter(
                                        x=fore_x + fore_x[::-1],
                                        y=r["q90"] + r["q10"][::-1],
                                        fill="toself", fillcolor="rgba(255,87,34,0.15)",
                                        line=dict(color="rgba(0,0,0,0)"), name="置信区间",
                                    ))
                                fig.add_trace(go.Scatter(
                                    x=fore_x, y=r["q50"],
                                    name="预测P50", line=dict(color="#FF5722", dash="dash"),
                                ))
                                fig.update_layout(template="plotly_white", height=350, title="预测曲线")
                                charts.append(fig)

                    else:
                        client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
                        resp = client.chat.completions.create(
                            model=DEEPSEEK_MODEL, max_tokens=1024,
                            messages=[
                                {"role": "system", "content": "你是BrandBrain品牌分析助手，帮助分析销售数据和营销策略。"},
                                {"role": "user", "content": user_input}
                            ],
                        )
                        response = resp.choices[0].message.content
                        tool_calls = []

                    st.markdown(response)
                    for fig in charts:
                        st.plotly_chart(fig, use_container_width=True)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "tool_calls": tool_calls,
                        "charts": charts,
                    })

                except Exception as e:
                    err = f"Agent出错：{e}"
                    st.error(err)
                    st.code(traceback.format_exc())
                    st.session_state["messages"].append({"role": "assistant", "content": err})

    if st.session_state["messages"]:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["agent_history"] = []
            st.rerun()
