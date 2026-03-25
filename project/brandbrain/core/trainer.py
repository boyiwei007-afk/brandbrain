"""
BrandFormer训练器
- 数据集构建
- 训练循环 + 早停
- 模型保存/加载
- LightGBM辅助模型
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from pathlib import Path

from core.brand_former import BrandFormer, QuantileLoss, build_model


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SalesDataset(Dataset):
    def __init__(
        self,
        sales: np.ndarray,           # (T,) 归一化后的销量
        exog: np.ndarray,            # (T, n_exog) 外生变量
        future_exog: np.ndarray,     # (T, n_future) 未来已知变量
        lookback: int,
        horizon: int,
    ):
        self.sales = torch.FloatTensor(sales)
        self.exog = torch.FloatTensor(exog)
        self.future_exog = torch.FloatTensor(future_exog)
        self.lookback = lookback
        self.horizon = horizon
        self.valid_indices = list(range(lookback, len(sales) - horizon + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x_hist = self.sales[t - self.lookback : t].unsqueeze(-1)  # (lookback, 1)
        x_exog = self.exog[t - self.lookback : t]                  # (lookback, n_exog)
        x_fut = self.future_exog[t : t + self.horizon]             # (horizon, n_future)
        y = self.sales[t : t + self.horizon].unsqueeze(-1)         # (horizon, 1)
        return x_hist, x_exog, x_fut, y


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────

class BrandFormerTrainer:
    """完整训练流程管理器"""

    def __init__(self, config: dict = None):
        from config import TRAIN_CONFIG, GRANULARITY_CONFIG, BRANDFORMER_CONFIG, MODEL_SAVE_DIR
        self.train_cfg = {**TRAIN_CONFIG, **(config or {})}
        self.gran_cfg = GRANULARITY_CONFIG
        self.model_cfg = BRANDFORMER_CONFIG
        self.save_dir = Path(MODEL_SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_y = StandardScaler()
        self.scaler_exog = StandardScaler()

        self.model: Optional[BrandFormer] = None
        self.lgbm_model = None
        self.exog_cols: List[str] = []
        self.future_exog_cols: List[str] = []
        self.target_col: str = "sales"
        self.date_col: str = "date"
        self.granularity: str = "daily"
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_epoch: int = 0

    # ────────────── 数据准备 ──────────────

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        exog_cols: List[str],
        future_exog_cols: List[str],
        granularity: str = "daily",
    ) -> Dict:
        """预处理并拆分数据集"""
        self.target_col = target_col
        self.date_col = date_col
        self.exog_cols = exog_cols
        self.future_exog_cols = future_exog_cols
        self.granularity = granularity

        cfg = self.gran_cfg[granularity]
        lookback = cfg["lookback"]
        horizon = cfg["forecast_horizon"]

        df = df.sort_values(date_col).reset_index(drop=True)
        sales = df[target_col].values.astype(float)
        exog = df[exog_cols].values.astype(float)
        future = df[future_exog_cols].values.astype(float) if future_exog_cols else np.zeros((len(df), 0))

        # 归一化
        sales_scaled = self.scaler_y.fit_transform(sales.reshape(-1, 1)).flatten()
        exog_scaled = self.scaler_exog.fit_transform(exog)
        # future_exog通常是0/1标志位，不做归一化
        future_scaled = future

        # 拆分
        n = len(sales)
        val_r = self.train_cfg["val_ratio"]
        test_r = self.train_cfg["test_ratio"]
        n_test = max(horizon, int(n * test_r))
        n_val = max(horizon, int(n * val_r))
        n_train = n - n_val - n_test

        def make_ds(start, end):
            return SalesDataset(
                sales_scaled[start:end],
                exog_scaled[start:end],
                future_scaled[start:end],
                lookback, horizon,
            )

        datasets = {
            "train": make_ds(0, n_train + lookback),
            "val": make_ds(n_train, n_train + n_val + lookback),
            "test": make_ds(n_train + n_val, n),
            "full": make_ds(0, n),
        }

        # 保存用于推理的原始数组
        self._sales_scaled = sales_scaled
        self._exog_scaled = exog_scaled
        self._future_scaled = future_scaled
        self._df = df

        return {
            "datasets": datasets,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_exog": len(exog_cols),
            "n_future": len(future_exog_cols),
            "lookback": lookback,
            "horizon": horizon,
        }

    # ────────────── 训练 ──────────────

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        exog_cols: List[str],
        future_exog_cols: List[str],
        granularity: str = "daily",
        progress_callback=None,
    ) -> Dict:
        """主训练入口，返回训练结果摘要"""
        t0 = time.time()

        # 数据准备
        data = self.prepare_data(df, target_col, date_col, exog_cols, future_exog_cols, granularity)
        cfg = self.gran_cfg[granularity]

        # 创建模型
        self.model = build_model(
            n_exog=data["n_exog"],
            n_future_exog=data["n_future"],
            granularity=granularity,
            config=self.model_cfg,
        ).to(self.device)

        criterion = QuantileLoss(self.model_cfg["quantiles"])
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_cfg["max_epochs"], eta_min=1e-5
        )

        bs = self.train_cfg["batch_size"]
        train_loader = DataLoader(data["datasets"]["train"], batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(data["datasets"]["val"], batch_size=bs, shuffle=False)

        best_val = float("inf")
        patience_cnt = 0
        self.train_losses, self.val_losses = [], []

        for epoch in range(1, self.train_cfg["max_epochs"] + 1):
            # ── 训练 ──
            self.model.train()
            tr_loss = 0.0
            for x_hist, x_exog, x_fut, y in train_loader:
                x_hist, x_exog, x_fut, y = (
                    x_hist.to(self.device),
                    x_exog.to(self.device),
                    x_fut.to(self.device),
                    y.to(self.device),
                )
                optimizer.zero_grad()
                preds, _ = self.model(x_hist, x_exog, x_fut if x_fut.size(-1) > 0 else None)
                loss = criterion(preds, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(train_loader)

            # ── 验证 ──
            self.model.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for x_hist, x_exog, x_fut, y in val_loader:
                    x_hist, x_exog, x_fut, y = (
                        x_hist.to(self.device), x_exog.to(self.device),
                        x_fut.to(self.device), y.to(self.device),
                    )
                    preds, _ = self.model(x_hist, x_exog, x_fut if x_fut.size(-1) > 0 else None)
                    vl_loss += criterion(preds, y).item()
            vl_loss /= max(len(val_loader), 1)

            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)
            scheduler.step()

            if progress_callback:
                progress_callback(epoch, self.train_cfg["max_epochs"], tr_loss, vl_loss)

            # ── 早停 ──
            if vl_loss < best_val - 1e-5:
                best_val = vl_loss
                self.best_epoch = epoch
                patience_cnt = 0
                self._save_checkpoint("best_model.pt")
            else:
                patience_cnt += 1
                if patience_cnt >= self.train_cfg["patience"]:
                    break

        # 加载最优权重
        self._load_checkpoint("best_model.pt")

        # 同时训练LightGBM
        lgbm_metrics = self._train_lgbm(df, target_col, date_col, exog_cols, granularity)

        elapsed = time.time() - t0
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": best_val,
            "total_epochs": len(self.train_losses),
            "elapsed_sec": round(elapsed, 1),
            "lgbm_metrics": lgbm_metrics,
            "device": str(self.device),
        }

    # ────────────── LightGBM ──────────────

    def _train_lgbm(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        exog_cols: List[str],
        granularity: str,
    ) -> Dict:
        """训练LightGBM用于SHAP解释和备用预测"""
        from config import LGBM_CONFIG
        cfg = self.gran_cfg[granularity]

        df = df.sort_values(date_col).reset_index(drop=True)
        feat_cols = exog_cols.copy()

        # 添加lag特征
        for lag in [1, 7, 14, 21, 28] if granularity == "daily" else [1, 2, 4, 8]:
            col = f"lag_{lag}"
            df[col] = df[target_col].shift(lag)
            feat_cols.append(col)

        # 滚动统计
        for w in [7, 14] if granularity == "daily" else [2, 4]:
            col = f"roll_mean_{w}"
            df[col] = df[target_col].rolling(w).mean()
            feat_cols.append(col)

        df = df.dropna()
        n = len(df)
        n_test = max(cfg["forecast_horizon"], int(n * self.train_cfg["test_ratio"]))
        n_val = max(cfg["forecast_horizon"], int(n * self.train_cfg["val_ratio"]))

        X = df[feat_cols].values
        y = df[target_col].values
        X_tr, y_tr = X[: n - n_val - n_test], y[: n - n_val - n_test]
        X_val, y_val = X[n - n_val - n_test : n - n_test], y[n - n_val - n_test : n - n_test]
        X_te, y_te = X[n - n_test :], y[n - n_test :]

        lgbm_cfg = {**LGBM_CONFIG}
        early_stop = lgbm_cfg.pop("early_stopping_rounds", 20)

        model = lgb.LGBMRegressor(**lgbm_cfg)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        self.lgbm_model = model
        self._lgbm_feat_cols = feat_cols
        self._lgbm_df = df

        # 评估
        y_pred = model.predict(X_te)
        mae = np.mean(np.abs(y_pred - y_te))
        rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
        mape = np.mean(np.abs((y_pred - y_te) / (y_te + 1e-8))) * 100

        # 保存
        joblib.dump(model, self.save_dir / "lgbm_model.pkl")
        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}

    # ────────────── 预测 ──────────────

    def predict(self, steps: int = None, exog_override: dict = None) -> Dict:
        """
        自回归式预测未来steps步
        exog_override: {col_name: value} 覆盖外生变量（用于决策模拟）
        """
        if self.model is None:
            raise RuntimeError("模型未训练")

        cfg = self.gran_cfg[self.granularity]
        steps = steps or cfg["forecast_horizon"]
        lookback = cfg["lookback"]

        df = self._df.copy()
        if exog_override:
            # 创建未来行，用最后一行填充并覆盖
            last_row = df.iloc[[-1]].copy()
            for col, val in exog_override.items():
                if col in last_row.columns:
                    last_row[col] = val
            future_rows = pd.concat([last_row] * steps, ignore_index=True)
            future_exog = future_rows[self.exog_cols].values.astype(float)
            future_exog_scaled = self.scaler_exog.transform(future_exog)
            future_future = (
                future_rows[self.future_exog_cols].values.astype(float)
                if self.future_exog_cols else np.zeros((steps, 0))
            )
        else:
            # 使用最近的exog值
            last_exog = self._exog_scaled[-lookback:]
            future_exog_scaled = np.tile(self._exog_scaled[-1:], (steps, 1))
            future_future = (
                np.tile(self._future_scaled[-1:], (steps, 1))
                if self._future_scaled.shape[1] > 0 else np.zeros((steps, 0))
            )

        self.model.eval()
        model_horizon = self.model.horizon
        with torch.no_grad():
            x_hist = torch.FloatTensor(self._sales_scaled[-lookback:]).unsqueeze(0).unsqueeze(-1).to(self.device)
            x_exog = torch.FloatTensor(self._exog_scaled[-lookback:]).unsqueeze(0).to(self.device)

            # 对齐到模型horizon（不足则tile，过长则截断）
            if future_future.shape[1] > 0:
                if future_future.shape[0] < model_horizon:
                    pad = np.tile(future_future[-1:], (model_horizon - future_future.shape[0], 1))
                    future_padded = np.concatenate([future_future, pad], axis=0)
                else:
                    future_padded = future_future[:model_horizon]
                x_fut = torch.FloatTensor(future_padded).unsqueeze(0).to(self.device)
            else:
                x_fut = None

            preds_scaled, attn_dict = self.model(x_hist, x_exog, x_fut)
            preds_scaled = preds_scaled.squeeze(0).cpu().numpy()  # (horizon, n_q)

        # 反归一化
        n_q = preds_scaled.shape[1]
        preds = np.zeros_like(preds_scaled)
        for i in range(n_q):
            preds[:, i] = self.scaler_y.inverse_transform(
                preds_scaled[:, i].reshape(-1, 1)
            ).flatten()

        q_names = [f"q{int(q*100)}" for q in self.model.quantiles]
        result = {col: preds[:steps, i].tolist() for i, col in enumerate(q_names)}
        result["attention"] = {k: v.cpu().numpy().tolist() for k, v in attn_dict.items()}
        return result

    def predict_lgbm(self, df_future: pd.DataFrame = None) -> np.ndarray:
        """LightGBM预测（用于SHAP）"""
        if self.lgbm_model is None:
            raise RuntimeError("LightGBM未训练")
        df = self._lgbm_df.copy()
        X = df[self._lgbm_feat_cols].values
        return self.lgbm_model.predict(X)

    # ────────────── 评估 ──────────────

    def evaluate(self) -> Dict:
        """在测试集上评估BrandFormer"""
        if self.model is None:
            raise RuntimeError("模型未训练")

        cfg = self.gran_cfg[self.granularity]
        lookback = cfg["lookback"]
        horizon = cfg["forecast_horizon"]

        sales = self._sales_scaled
        exog = self._exog_scaled
        future = self._future_scaled
        n = len(sales)
        n_test = max(horizon, int(n * self.train_cfg["test_ratio"]))

        preds_50, actuals = [], []
        self.model.eval()
        with torch.no_grad():
            for t in range(n - n_test, n - horizon + 1):
                x_hist = torch.FloatTensor(sales[t - lookback: t]).unsqueeze(0).unsqueeze(-1).to(self.device)
                x_exog = torch.FloatTensor(exog[t - lookback: t]).unsqueeze(0).to(self.device)
                x_fut = torch.FloatTensor(future[t: t + horizon]).unsqueeze(0).to(self.device)

                p, _ = self.model(x_hist, x_exog, x_fut if x_fut.size(-1) > 0 else None)
                p50 = p[0, :, 1].cpu().numpy()  # median
                p50 = self.scaler_y.inverse_transform(p50.reshape(-1, 1)).flatten()
                preds_50.extend(p50)
                actuals.extend(self.scaler_y.inverse_transform(
                    sales[t: t + horizon].reshape(-1, 1)
                ).flatten())

        preds_50 = np.array(preds_50[:n_test])
        actuals = np.array(actuals[:n_test])
        mae = np.mean(np.abs(preds_50 - actuals))
        rmse = np.sqrt(np.mean((preds_50 - actuals) ** 2))
        mape = np.mean(np.abs((preds_50 - actuals) / (actuals + 1e-8))) * 100
        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "preds": preds_50,
            "actuals": actuals,
        }

    # ────────────── 保存/加载 ──────────────

    def _save_checkpoint(self, name: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_y": self.scaler_y,
            "scaler_exog": self.scaler_exog,
            "exog_cols": self.exog_cols,
            "future_exog_cols": self.future_exog_cols,
            "target_col": self.target_col,
            "date_col": self.date_col,
            "granularity": self.granularity,
        }, self.save_dir / name)

    def _load_checkpoint(self, name: str):
        path = self.save_dir / name
        if not path.exists():
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.scaler_y = ckpt["scaler_y"]
        self.scaler_exog = ckpt["scaler_exog"]

    def save(self, name: str = "trainer.pkl"):
        """保存完整Trainer（含模型权重）"""
        self._save_checkpoint("best_model.pt")
        state = {k: v for k, v in self.__dict__.items() if not k.startswith("model") and k != "lgbm_model"}
        joblib.dump(state, self.save_dir / name)

    @classmethod
    def load(cls, name: str = "trainer.pkl") -> "BrandFormerTrainer":
        from config import MODEL_SAVE_DIR
        save_dir = Path(MODEL_SAVE_DIR)
        trainer = cls()
        state = joblib.load(save_dir / name)
        trainer.__dict__.update(state)
        # 重建模型并加载权重
        cfg = trainer.gran_cfg[trainer.granularity]
        trainer.model = build_model(
            n_exog=len(trainer.exog_cols),
            n_future_exog=len(trainer.future_exog_cols),
            granularity=trainer.granularity,
        ).to(trainer.device)
        trainer._load_checkpoint("best_model.pt")
        trainer.lgbm_model = joblib.load(save_dir / "lgbm_model.pkl")
        return trainer
