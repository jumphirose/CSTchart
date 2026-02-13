#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
# フォントユーティリティ
# ===========================================================
def get_available_fonts():
    return sorted({f.name for f in fm.fontManager.ttflist})


def apply_font(font_name):
    fonts = get_available_fonts()
    if font_name in fonts:
        plt.rcParams["font.family"] = font_name
    else:
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


# ===========================================================
# Config
# ===========================================================
class Config:
    def __init__(self, fit_method, upper, lower):
        self.FIT_METHOD = fit_method
        self.ENDPOINT_UPPER = upper
        self.ENDPOINT_LOWER = lower
        self.ENDPOINT_ATOL = 1e-9

        self.FIGSIZE = (5, 5)
        self.LIMS = dict(x=(0, 100), y=(lower, upper))
        self.LABELS = dict(x="油分比率 (wt%)", y="二層分離温度 (°C)")
        self.FONT_SIZE = dict(label=14, ticks=12, legend=12)

        self.MARKER_SIZE = 60
        self.ALPHA_POINTS = 0.9
        self.LINEWIDTH = 2.0

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
# フィッティング
# ===========================================================
def fit_predict_curve(x, y, cfg):
    if len(x) < 2:
        return None, None

    xl = np.linspace(np.min(x), np.max(x), 200)

    if cfg.FIT_METHOD == "akima":
        return xl, Akima1DInterpolator(x, y)(xl)
    if cfg.FIT_METHOD == "pchip":
        return xl, PchipInterpolator(x, y)(xl)
    if cfg.FIT_METHOD == "spline" and len(x) > 3:
        return xl, UnivariateSpline(x, y, s=cfg.SPLINE_S_FACTOR)(xl)
    if cfg.FIT_METHOD == "loess":
        out = lowess(y, x, frac=cfg.LOESS_FRAC, it=cfg.LOESS_IT)
        return out[:, 0], out[:, 1]
    if cfg.FIT_METHOD == "ridge":
        model = Pipeline([
            ("poly", PolynomialFeatures(cfg.RIDGE_DEGREE)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(cfg.RIDGE_ALPHAS, cv=KFold(5))),
        ])
        model.fit(x[:, None], y)
        return xl, model.predict(xl[:, None])
    if cfg.FIT_METHOD == "savgol":
        idx = np.argsort(x)
        return np.sort(x), savgol_filter(y[idx], cfg.SAVGOL_WINDOW, cfg.SAVGOL_POLY)
    if cfg.FIT_METHOD == "robust":
        poly = PolynomialFeatures(cfg.ROBUST_DEGREE)
        X = poly.fit_transform(x[:, None])
        huber = HuberRegressor(epsilon=cfg.HUBER_EPSILON, alpha=cfg.HUBER_ALPHA)
        huber.fit(X, y)
        return xl, huber.predict(poly.transform(xl[:, None]))

    return None, None


# ===========================================================
# サンプル描画（端点・凡例完全制御）
# ===========================================================
def process_sample(ax, df, base_row, idx, cfg):
    x = pd.to_numeric(df.iloc[base_row + 1], errors="coerce").to_numpy()
    y1 = pd.to_numeric(df.iloc[base_row + 2], errors="coerce").to_numpy()
    y2 = pd.to_numeric(df.iloc[base_row + 3], errors="coerce").to_numpy()

    label = f"Sample{idx + 1}"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P"]
    color = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]

    label_used = False

    for y in (y1, y2):
        main = (
            np.isfinite(y)
            & (~np.isclose(y, cfg.ENDPOINT_UPPER, atol=cfg.ENDPOINT_ATOL))
            & (~np.isclose(y, cfg.ENDPOINT_LOWER, atol=cfg.ENDPOINT_ATOL))
        )

        upper = np.isclose(y, cfg.ENDPOINT_UPPER, atol=cfg.ENDPOINT_ATOL)
        lower = np.isclose(y, cfg.ENDPOINT_LOWER, atol=cfg.ENDPOINT_ATOL)

        if np.any(main):
            ax.scatter(
                x[main], y[main],
                s=cfg.MARKER_SIZE,
                color=color,
                marker=marker,
                alpha=cfg.ALPHA_POINTS,
                label=label if not label_used else "_nolegend_",
                zorder=3,
                clip_on=False,
            )
            label_used = True

            xl, yl = fit_predict_curve(x[main], y[main], cfg)
            if xl is not None:
                ax.plot(xl, yl, color=color, lw=cfg.LINEWIDTH, zorder=2)

        # 端点（必ず最前面に描画）
        ax.scatter(
            x[upper], y[upper],
            marker="^", color=color,
            s=cfg.MARKER_SIZE,
            zorder=5, clip_on=False,
        )
        ax.scatter(
            x[lower], y[lower],
            marker="v", color=color,
            s=cfg.MARKER_SIZE,
            zorder=5, clip_on=False,
        )


# ===========================================================
# メイン解析
# ===========================================================
def run_analysis(file, cfg):
    df = pd.read_excel(file, header=None)
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE)

    for i, r in enumerate(range(1, len(df), 4)):
        process_sample(ax, df, r, i, cfg)

    ax.set_xlim(*cfg.LIMS["x"])
    ax.set_ylim(*cfg.LIMS["y"])
    ax.set_xlabel(cfg.LABELS["x"], fontsize=cfg.FONT_SIZE["label"])
    ax.set_ylabel(cfg.LABELS["y"], fontsize=cfg.FONT_SIZE["label"])

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, linestyle=":")

    ax.legend(fontsize=cfg.FONT_SIZE["legend"])
    plt.tight_layout()
    return fig


# ===========================================================
# Streamlit UI
# ===========================================================
st.set_page_config(page_title="二層分離温度解析", layout="centered")
st.title("二層分離温度解析アプリ")

# フォント選択
fonts = get_available_fonts()
font = st.sidebar.selectbox(
    "フォント選択（日本語対応）",
    options=["Meiryo", "IPAexGothic", "IPAGothic", "Noto Sans CJK JP"] + fonts,
    index=0,
)
apply_font(font)

fit_method = st.sidebar.selectbox(
    "近似方法",
    ["akima", "pchip", "spline", "loess", "ridge", "savgol", "robust"]
)

upper = st.sidebar.slider("上限温度 (°C)", -100, 150, 70)
lower = st.sidebar.slider("下限温度 (°C)", -150, 100, -50)

uploaded = st.file_uploader("Excelファイルをアップロード", type=["xlsx", "xlsm"])

if uploaded:
    cfg = Config(fit_method, upper, lower)
    fig = run_analysis(uploaded, cfg)

    st.pyplot(fig)

    # ダウンロード
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)

    st.download_button(
        label="PNGとしてダウンロード",
        data=buf,
        file_name="result.png",
        mime="image/png",
    )
else:
    st.info("Excelファイルをアップロードしてください")
