"""
生成品牌销量示例数据
运行: python data/generate_sample.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_brand_sales(
    start="2021-01-01",
    end="2024-12-31",
    brand="BrandX",
    base_sales=1000,
):
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # --- 趋势 ---
    trend = np.linspace(0, 300, n)

    # --- 年度季节性（圣诞/春节高峰）---
    seasonal_yearly = 200 * np.sin(2 * np.pi * np.arange(n) / 365.25 - np.pi / 2)

    # --- 周季节性（周末低谷）---
    dow = np.array([d.dayofweek for d in dates])
    seasonal_weekly = np.where(dow >= 5, -80, 30)

    # --- 节假日效应 ---
    holiday_boost = np.zeros(n)
    for i, d in enumerate(dates):
        # 春节前后
        if d.month == 1 and 20 <= d.day <= 31:
            holiday_boost[i] = 400
        if d.month == 2 and 1 <= d.day <= 10:
            holiday_boost[i] = 500
        # 双11
        if d.month == 11 and d.day == 11:
            holiday_boost[i] = 800
        # 618
        if d.month == 6 and 15 <= d.day <= 20:
            holiday_boost[i] = 350
        # 国庆
        if d.month == 10 and 1 <= d.day <= 7:
            holiday_boost[i] = 200

    # --- 价格（有价格弹性）---
    price = np.full(n, 99.0)
    # 模拟几次调价
    price[365:730] = 89.0
    price[730:] = 109.0
    # 促销期间降价
    promo_mask = holiday_boost > 0
    price[promo_mask] *= 0.8
    price_effect = -3.0 * (price - 99.0)  # 价格弹性

    # --- 折扣率 ---
    discount_rate = np.zeros(n)
    discount_rate[promo_mask] = 0.2
    discount_effect = 300 * discount_rate

    # --- 广告投放 ---
    ad_spend = np.random.uniform(200, 800, n)
    ad_spend[promo_mask] *= 2.5
    ad_effect = 0.3 * ad_spend

    # --- 是否促销 ---
    is_promotion = promo_mask.astype(int)

    # --- 噪声 ---
    noise = np.random.normal(0, 60, n)

    # --- 合成销量 ---
    sales = (
        base_sales
        + trend
        + seasonal_yearly
        + seasonal_weekly
        + holiday_boost
        + price_effect
        + discount_effect
        + ad_effect
        + noise
    ).clip(50)

    df = pd.DataFrame({
        "date": dates,
        "brand": brand,
        "sales": sales.round(0).astype(int),
        "price": price.round(2),
        "discount_rate": discount_rate.round(3),
        "ad_spend": ad_spend.round(0).astype(int),
        "is_promotion": is_promotion,
        "is_holiday": (holiday_boost > 0).astype(int),
        "temperature": (20 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25) + np.random.normal(0, 2, n)).round(1),
        "competitor_price": (price * np.random.uniform(0.9, 1.1, n)).round(2),
    })
    return df


if __name__ == "__main__":
    df = generate_brand_sales()
    out = Path(__file__).parent / "sample_data.csv"
    df.to_csv(out, index=False)
    print(f"已生成示例数据: {out}  (共 {len(df)} 行)")
    print(df.head())
    print(df.describe())
