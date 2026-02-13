#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
 二層分離温度フォーマット解析 Streamlitアプリ版
===========================================================
"""

import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold


# ===========================================================
# ログ設定
# ===========================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ===========================================================
# Config クラス（Streamlit対応）
# ===========================================================
class Config:
    def __init__(
        self,
        fit_method="akima",
        endpoint_upper=70.0,
        endpoint_lower=-50.0,
        sheet_name=0,
    ):
        # 基本設定
        self.FIT_METHOD = fit_method
        self.SHEET_NAME = sheet_name

        # フォント設定
        self.FONT_FAMILY = "Meiryo"
        self.FONT_SIZE = dict(title=14, label=14, ticks=12, legend=12)

        # グラフ設定
        self.FIGSIZE = (5, 5)
        self.LABELS = dict(x="油分比率 (wt%)", y="二層分離温度 (°C)")
        self.LIMS = dict(x=(0, 100), y=(endpoint_lower, endpoint_upper))
        self.GRID = True

        self.X_LABELPAD = 8
        self.Y_LABELPAD = 8
        self.X_LABELPOS = "center"
        self.Y_LABELPOS = "center"

        self.X_TICK_POS = "bottom"
        self.Y_TICK_POS = "left"
        self.XTICK_PAD = 15
        self.YTICK_PAD = 5

        # プロット設定
        self.MARKER_SIZE = 60
        self.ALPHA_POINTS = 0.9
        self.LINEWIDTH = 2.0
        self.LINESTYLE = "-"

        # 上限・下限
        self.ENDPOINT_UPPER = endpoint_upper
        self.ENDPOINT_LOWER = endpoint_lower
        self.ENDPOINT_ATOL = 1e-9

        # フィッティングパラメータ
        self.SPLINE_S_FACTOR = 5
        self.LOESS_FRAC = 0.35
        self.LOESS_IT = 1
        self.RIDGE_DEGREE = 4
        self.RIDGE_ALPHAS = np.logspace(-4, 4, 17)
        self.SAVGOL_WINDOW = 7
        self.SAVGOL_POLY = 2
        self.ROBUST_DEGREE = 3
        self.HUBER_EPSILON = 1.35
        self.HUBER_ALPHA = 1e-4


# ===========================================================
# フォント設定
# ===========================================================
def apply_global_font_settings(cfg: Config):
    plt.rcParams["font.family"] = cfg.FONT_FAMILY
    plt.rcParams["axes.unicode_minus"] = False


# ===========================================================
# ユーティリティ
# ===========================================================
def safe_name(txt):
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", str(txt)).strip("_") or "noname"


# ===========================================================
# フィッティング処理
# ===========================================================
def fit_predict_curve(x, y, cfg: Config):
    method = cfg.FIT_METHOD.lower()
    if len(x) < 2:
        return None, None

    xl = np.linspace(np.min(x), np.max(x), 200)

    if method == "spline":
        if len(x) <= 3:
            return None, None
        model = UnivariateSpline(x, y, s=cfg.SPLINE_S_FACTOR)
        return xl, model(xl)

    elif method == "akima":
        model = Akima1DInterpolator(x, y)
        return xl, model(xl)

    elif method == "pchip":
        model = PchipInterpolator(x, y)
        return xl, model(xl)

    elif method == "loess":
        out = lowess(y, x, frac=cfg.LOESS_FRAC, it=cfg.LOESS_IT, return_sorted=True)
        return out[:, 0], out[:, 1]

    elif method == "ridge":
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=cfg.RIDGE_DEGREE)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=cfg.RIDGE_ALPHAS, cv=KFold(5)))
        ])
        model.fit(x.reshape(-1, 1), y)
        return xl, model.predict(xl.reshape(-1, 1))

    elif method == "savgol":
        idx = np.argsort(x)
        y_fit = savgol_filter(y[idx], window_length=cfg.SAVGOL_WINDOW, polyorder=cfg.SAVGOL_POLY)
        return np.sort(x), y_fit

    elif method == "robust":
        poly = PolynomialFeatures(degree=cfg.ROBUST_DEGREE)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        huber = HuberRegressor(epsilon=cfg.HUBER_EPSILON, alpha=cfg.HUBER_ALPHA)
        huber.fit(X_poly, y)
        return xl, huber.predict(poly.transform(xl.reshape(-1, 1)))

    return None, None


# ===========================================================
# 軸設定
# ===========================================================
def apply_axis_format(ax, cfg: Config):
    ax.set_xlim(*cfg.LIMS["x"])
    ax.set_ylim(*cfg.LIMS["y"])

    ax.set_xlabel(cfg.LABELS["x"], fontsize=cfg.FONT_SIZE["label"],
                  labelpad=cfg.X_LABELPAD, loc=cfg.X_LABELPOS)
    ax.set_ylabel(cfg.LABELS["y"], fontsize=cfg.FONT_SIZE["label"],
                  labelpad=cfg.Y_LABELPAD, loc=cfg.Y_LABELPOS)

    ax.tick_params(direction="in", which="both",
                   labelsize=cfg.FONT_SIZE["ticks"])
    ax.xaxis.set_ticks_position(cfg.X_TICK_POS)
    ax.yaxis.set_ticks_position(cfg.Y_TICK_POS)
    ax.tick_params(axis="x", pad=cfg.XTICK_PAD)
    ax.tick_params(axis="y", pad=cfg.YTICK_PAD)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    if cfg.GRID:
        ax.grid(True, linestyle=":", linewidth=0.8)


# ===========================================================
# サンプル描画
# ===========================================================
def process_sample(ax, df, base_row, idx, cfg: Config):
    x = pd.to_numeric(df.iloc[base_row + 1], errors="coerce").to_numpy()
    y1 = pd.to_numeric(df.iloc[base_row + 2], errors="coerce").to_numpy()
    y2 = pd.to_numeric(df.iloc[base_row + 3], errors="coerce").to_numpy()

    label = f"Sample{idx+1}"
    if df.shape[1] > 9 and not pd.isna(df.iloc[base_row, 9]):
        label = str(df.iloc[base_row, 9])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    c = colors[idx % len(colors)]
    m = markers[idx % len(markers)]

    scatter_kw = dict(
        s=cfg.MARKER_SIZE,
        alpha=cfg.ALPHA_POINTS,
        color=c,
        marker=m,
        edgecolors="none",
        clip_on=False,
    )

    def mask_main(y):
        return (
            np.isfinite(y)
            & (~np.isclose(y, cfg.ENDPOINT_UPPER, atol=cfg.ENDPOINT_ATOL))
            & (~np.isclose(y, cfg.ENDPOINT_LOWER, atol=cfg.ENDPOINT_ATOL))
        )

    for y in [y1, y2]:
        main = mask_main(y)
        if np.any(main):
            xv, yv = x[main], y[main]
            ax.scatter(xv, yv, label=label, **scatter_kw)
            xl, yl = fit_predict_curve(xv, yv, cfg)
            if xl is not None:
                ax.plot(xl, yl, lw=cfg.LINEWIDTH, color=c)

    return label


# ===========================================================
# 解析実行関数（Streamlitから呼ぶ）
# ===========================================================
def run_analysis(file, cfg: Config):
    apply_global_font_settings(cfg)

    df = pd.read_excel(file, sheet_name=cfg.SHEET_NAME, header=None)

    fig, ax = plt.subplots(figsize=cfg.FIGSIZE)

    for i, base_row in enumerate(range(1, len(df.index), 4)):
        process_sample(ax, df, base_row, i, cfg)

    apply_axis_format(ax, cfg)
    ax.legend(fontsize=cfg.FONT_SIZE["legend"])
    plt.tight_layout()

    return fig


# ===========================================================
# Streamlit UI
# ===========================================================
st.set_page_config(page_title="二層分離温度解析", layout="centered")
st.title("二層分離温度 解析アプリ")

st.sidebar.header("設定")

fit_method = st.sidebar.selectbox(
    "近似方法",
    ["akima", "spline", "pchip", "loess", "ridge", "savgol", "robust"]
)

upper = st.sidebar.slider("上限温度 (°C)", -100, 150, 70)
lower = st.sidebar.slider("下限温度 (°C)", -150, 100, -50)

uploaded = st.file_uploader(
    "Excelファイルをアップロードしてください",
    type=["xlsx", "xlsm"]
)

if uploaded:
    cfg = Config(
        fit_method=fit_method,
        endpoint_upper=upper,
        endpoint_lower=lower,
    )

    with st.spinner("解析中..."):
        fig = run_analysis(uploaded, cfg)

    st.pyplot(fig)

else:
    st.info("Excelファイルをアップロードすると、ここにグラフが表示されます。")
