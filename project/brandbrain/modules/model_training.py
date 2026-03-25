"""
模块3：模型训练
- 训练配置界面 + 实时进度
- 模型保存/加载（Registry）
- 训练曲线 + 测试集评估
"""
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

MODEL_DIR = Path("saved_models")
REGISTRY_FILE = MODEL_DIR / "model_registry.json"


# ──────────────────────────────────────────────
# 模型注册表
# ──────────────────────────────────────────────

def load_registry() -> List[Dict]:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8")).get("models", [])
        except Exception:
            pass
    return []


def save_registry(models: List[Dict]):
    MODEL_DIR.mkdir(exist_ok=True)
    REGISTRY_FILE.write_text(
        json.dumps({"models": models[:10]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def register_model(info: Dict):
    models = load_registry()
    # 如果同名则更新
    models = [m for m in models if m.get("id") != info.get("id")]
    models.insert(0, info)
    save_registry(models)


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def plot_training_curves(train_losses: list, val_losses: list, best_epoch: int) -> go.Figure:
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name="训练损失",
                             line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="验证损失",
                             line=dict(color="#FF5722", width=2)))
    if best_epoch <= len(train_losses):
        fig.add_vline(x=best_epoch, line_dash="dash", line_color="green",
                      annotation_text=f"最优 Epoch {best_epoch}")
    fig.update_layout(
        template="plotly_white", height=380,
        title="训练曲线（QuantileLoss）",
        xaxis_title="Epoch", yaxis_title="Loss",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_test_evaluation(preds: np.ndarray, actuals: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["预测 vs 实际（测试集）", "残差分布"])
    n = min(len(preds), 200)
    fig.add_trace(go.Scatter(y=actuals[:n], name="实际", line=dict(color="#2196F3")), row=1, col=1)
    fig.add_trace(go.Scatter(y=preds[:n], name="预测", line=dict(color="#FF5722", dash="dash")), row=1, col=1)
    residuals = preds[:n] - actuals[:n]
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40, name="残差",
                               marker_color="#9C27B0", opacity=0.7), row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_layout(template="plotly_white", height=380, title="测试集评估", showlegend=True)
    return fig


# ──────────────────────────────────────────────
# Streamlit渲染
# ──────────────────────────────────────────────

def render_model_training():
    st.header("🤖 模型训练")

    if not st.session_state.get("preprocessing_done"):
        st.warning("请先完成「预处理」Tab的数据处理")
        return

    df = st.session_state["preprocessed_df"]
    date_col = st.session_state["date_col"]
    target_col = st.session_state["target_col"]
    exog_cols = st.session_state["exog_cols"]
    future_exog_cols = st.session_state.get("future_exog_cols", [])
    granularity = st.session_state["granularity"]

    # ── 已保存模型列表 ──
    registry = load_registry()
    if registry:
        st.subheader("📦 已保存的模型")
        for i, m in enumerate(registry):
            met = m.get("metrics", {})
            cols = st.columns([3, 1, 1, 1, 1, 1])
            cols[0].markdown(
                f"**{m.get('saved_at', 'N/A')}** ｜ {m.get('granularity', '')} ｜ "
                f"目标: {m.get('target_col', '')} ｜ Epoch: {m.get('best_epoch', '?')}"
            )
            cols[1].metric("MAE", met.get("mae", "-"))
            cols[2].metric("RMSE", met.get("rmse", "-"))
            cols[3].metric("MAPE%", met.get("mape", "-"))
            with cols[4]:
                if st.button("加载", key=f"load_model_{i}", use_container_width=True):
                    _load_model_from_registry(m)
            with cols[5]:
                if st.button("删除", key=f"del_model_{i}", use_container_width=True):
                    registry.pop(i)
                    save_registry(registry)
                    st.rerun()
        st.markdown("---")

    # ── 已加载模型状态 ──
    if st.session_state.get("training_done"):
        trainer = st.session_state["trainer"]
        st.success(f"当前模型已就绪（最优Epoch: {trainer.best_epoch}）— 可直接切换到预测/决策Tab")
        with st.expander("查看训练曲线"):
            if trainer.train_losses:
                st.plotly_chart(
                    plot_training_curves(trainer.train_losses, trainer.val_losses, trainer.best_epoch),
                    use_container_width=True,
                )
        if st.button("重新训练", use_container_width=True):
            st.session_state["training_done"] = False
            st.rerun()
        return

    # ── 训练参数配置 ──
    st.subheader("训练参数配置")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_epochs = st.slider("最大轮次", 20, 300, 100, step=10)
        batch_size = st.selectbox("批量大小", [16, 32, 64], index=1)
    with col2:
        lr = st.select_slider("学习率", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3,
                               format_func=lambda x: f"{x:.0e}")
        patience = st.slider("早停耐心值", 5, 30, 10)
    with col3:
        d_model = st.selectbox("模型维度 d_model", [32, 64, 128], index=1)
        n_heads = st.selectbox("注意力头数", [2, 4, 8], index=1)

    st.info(
        f"数据集：**{len(df)}** 行 ｜ 目标：**{target_col}** ｜ "
        f"外生变量：**{len(exog_cols)}** 个 ｜ 粒度：**{granularity}**"
    )

    if st.button("🚀 开始训练", type="primary", use_container_width=True):
        from core.trainer import BrandFormerTrainer
        from config import BRANDFORMER_CONFIG, TRAIN_CONFIG

        model_cfg = {**BRANDFORMER_CONFIG, "d_model": d_model, "n_heads": n_heads}
        train_cfg = {**TRAIN_CONFIG, "max_epochs": max_epochs, "batch_size": batch_size,
                     "learning_rate": lr, "patience": patience}

        trainer = BrandFormerTrainer(config=train_cfg)
        trainer.model_cfg = model_cfg

        progress_bar = st.progress(0.0, text="准备中...")
        status_text = st.empty()
        loss_placeholder = st.empty()
        epoch_logs: Dict[str, list] = {"train": [], "val": []}

        def progress_callback(epoch, total, tr_loss, vl_loss):
            progress_bar.progress(epoch / total, text=f"Epoch {epoch}/{total}")
            status_text.markdown(
                f"**Epoch {epoch}** ｜ 训练: `{tr_loss:.4f}` ｜ 验证: `{vl_loss:.4f}`"
            )
            epoch_logs["train"].append(tr_loss)
            epoch_logs["val"].append(vl_loss)
            if epoch % 5 == 0 or epoch == total:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=epoch_logs["train"], name="训练",
                                         line=dict(color="#2196F3")))
                fig.add_trace(go.Scatter(y=epoch_logs["val"], name="验证",
                                         line=dict(color="#FF5722")))
                fig.update_layout(template="plotly_white", height=260,
                                  title="实时训练曲线", showlegend=True,
                                  margin=dict(t=40, b=20))
                loss_placeholder.plotly_chart(fig, use_container_width=True)

        try:
            with st.spinner("正在训练 BrandFormer..."):
                result = trainer.train(
                    df=df, target_col=target_col, date_col=date_col,
                    exog_cols=exog_cols, future_exog_cols=future_exog_cols,
                    granularity=granularity, progress_callback=progress_callback,
                )

            progress_bar.progress(1.0, text="训练完成 ✅")

            # 评估
            with st.spinner("评估测试集..."):
                eval_result = trainer.evaluate()

            # 保存
            model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = f"brandformer_{model_id}.pt"
            lgbm_name = f"lgbm_{model_id}.pkl"
            trainer_name = f"trainer_{model_id}.pkl"
            trainer._save_checkpoint(ckpt_name)
            import joblib
            import joblib as jlb
            jlb.dump(trainer.lgbm_model, MODEL_DIR / lgbm_name)
            trainer.save(trainer_name)

            # 只保留 JSON 可序列化的标量指标，去掉 preds/actuals ndarray
            safe_metrics = {k: float(v) if hasattr(v, "item") else v
                            for k, v in eval_result.items()
                            if k not in ("preds", "actuals")}

            model_info = {
                "id": model_id,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "granularity": granularity,
                "target_col": target_col,
                "exog_cols": exog_cols,
                "future_exog_cols": future_exog_cols,
                "n_rows": len(df),
                "best_epoch": result["best_epoch"],
                "metrics": safe_metrics,
                "checkpoint_file": ckpt_name,
                "lgbm_file": lgbm_name,
                "trainer_file": trainer_name,
            }
            register_model(model_info)

            st.session_state["trainer"] = trainer
            st.session_state["training_done"] = True

            # 结果展示
            st.success(f"训练完成 ✅ 最优Epoch: {result['best_epoch']} ｜ 耗时: {result['elapsed_sec']}s ｜ 已自动保存")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("最优Epoch", result["best_epoch"])
            m2.metric("测试 MAE", eval_result.get("mae", "N/A"))
            m3.metric("测试 RMSE", eval_result.get("rmse", "N/A"))
            m4.metric("测试 MAPE%", eval_result.get("mape", "N/A"))

            st.plotly_chart(
                plot_training_curves(trainer.train_losses, trainer.val_losses, trainer.best_epoch),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_test_evaluation(eval_result["preds"], eval_result["actuals"]),
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"训练出错: {e}")
            st.code(traceback.format_exc())


def _load_model_from_registry(model_info: Dict):
    """从注册表加载模型到session_state"""
    try:
        from core.trainer import BrandFormerTrainer
        trainer_file = model_info.get("trainer_file", "trainer.pkl")
        trainer = BrandFormerTrainer.load(trainer_file)

        st.session_state["trainer"] = trainer
        st.session_state["training_done"] = True

        # 恢复配置（如果session中没有预处理数据，使用注册表中的元数据）
        if "exog_cols" not in st.session_state or not st.session_state["exog_cols"]:
            st.session_state["exog_cols"] = model_info.get("exog_cols", [])
            st.session_state["future_exog_cols"] = model_info.get("future_exog_cols", [])
            st.session_state["granularity"] = model_info.get("granularity", "daily")
            st.session_state["target_col"] = model_info.get("target_col", "sales")

        st.success(f"模型已加载：{model_info.get('saved_at', '')}（最优Epoch: {model_info.get('best_epoch', '?')}）")
        st.rerun()
    except Exception as e:
        st.error(f"加载失败: {e}")
        st.code(traceback.format_exc())
