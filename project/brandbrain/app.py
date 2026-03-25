"""
BrandBrain 主应用入口
启动命令：streamlit run app.py
"""
import streamlit as st

# ── 页面配置（必须第一行）──
st.set_page_config(
    page_title="BrandBrain by 韦博懿",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局样式 ──
st.markdown("""
<style>
/* 标题栏 */
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { color: white; margin: 0; font-size: 2rem; }
.main-header p { color: rgba(255,255,255,0.8); margin: 0.3rem 0 0 0; font-size: 0.95rem; }

/* 指标卡片 */
div[data-testid="metric-container"] {
    background: #f8f9ff;
    border: 1px solid #e3e8f0;
    border-radius: 10px;
    padding: 0.8rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}

/* Tab样式 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #e0e0e0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 500;
}

/* 进度条 */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #2196F3, #1565c0);
}

/* 按钮 */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0, #1976d2);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(21,101,192,0.4);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 侧边栏
# ──────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 BrandBrain by 韦博懿")
        st.markdown("**品牌销量智能决策系统**")
        st.markdown("---")

        # 工作流状态
        st.markdown("### 📋 工作流进度")
        steps = [
            ("数据导入", "config_done"),
            ("数据预处理", "preprocessing_done"),
            ("模型训练", "training_done"),
            ("预测分析", "last_forecast"),
        ]
        for i, (name, key) in enumerate(steps, 1):
            done = key in st.session_state and st.session_state[key] is not None
            icon = "✅" if done else "⏳"
            color = "#4CAF50" if done else "#9E9E9E"
            st.markdown(
                f'<p style="color:{color};margin:4px 0">{icon} {i}. {name}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # 当前数据信息
        if "df" in st.session_state:
            df = st.session_state["df"]
            st.markdown("### 📊 当前数据集")
            st.markdown(f"- **文件：** {st.session_state.get('file_name', 'N/A')}")
            st.markdown(f"- **行数：** {len(df):,}")
            st.markdown(f"- **目标列：** {st.session_state.get('target_col', 'N/A')}")
            st.markdown(f"- **粒度：** {st.session_state.get('granularity', 'N/A')}")

            if "preprocessed_df" in st.session_state:
                n_feat = len(st.session_state.get("exog_cols", []))
                st.markdown(f"- **特征数：** {n_feat}")

        st.markdown("---")

        # 模型信息
        if st.session_state.get("training_done"):
            trainer = st.session_state["trainer"]
            st.markdown("### 🤖 模型状态")
            st.markdown(f"- **模型：** BrandFormer")
            st.markdown(f"- **最优Epoch：** {trainer.best_epoch}")
            st.markdown(f"- **设备：** {trainer.device}")
            if trainer.train_losses:
                best_val = min(trainer.val_losses)
                st.markdown(f"- **最优验证损失：** {best_val:.4f}")



# ──────────────────────────────────────────────
# 主标题
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 BrandBrain · 品牌销量智能决策系统</h1>
    <p>数据驱动 · 预测领先 · AI决策 | 从数据到洞察，只需自然语言</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 主Tab布局
# ──────────────────────────────────────────────

tabs = st.tabs([
    "📂 数据导入",
    "🔧 数据预处理",
    "🤖 模型训练",
    "🔮 预测分析",
    "🎮 决策模拟",
    "💬 智能对话",
])

render_sidebar()

with tabs[0]:
    from modules.data_import import render_data_import
    render_data_import()

with tabs[1]:
    from modules.preprocessing import render_preprocessing
    render_preprocessing()

with tabs[2]:
    from modules.model_training import render_model_training
    render_model_training()

with tabs[3]:
    from modules.prediction import render_prediction
    render_prediction()

with tabs[4]:
    from modules.decision_sim import render_decision_simulation
    render_decision_simulation()

with tabs[5]:
    from modules.agent_dialog import render_agent_dialog
    render_agent_dialog()
