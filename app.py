#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

from matplotlib.ticker import MultipleLocator
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


# ===========================================================
# フォント管理（同梱フォント前提）
# ===========================================================
def load_font(font_ui: str):
    """
    UIで選択されたフォント名から FontProperties を返す
    フォントはすべて同梱ファイルを直接参照する
    """
    base = Path(__file__).parent

    font_map = {
        "日本語（IPAexGothic）": base / "fonts" / "ipaexg.ttf",
        "英語（Times）": base / "fonts" / "NimbusRomNo9L-Reg.otf",
    }

    font_path = font_map.get(font_ui)

    if font_path is not None and font_path.exists():
        return fm.FontProperties(fname=str(font_path))

    # 最終フォールバック（まず来ない）
    return fm.FontProperties(family="DejaVu Sans")

# ===========================================================
# Config
# ===========================================================
class Config:
    def __init__(self, upper, lower, font_prop, fit_method, fit_params):
        self.UPPER = upper
        self.LOWER = lower
        self.FONT = font_prop
        self.FIT_METHOD = fit_method
        self.FIT_PARAMS = fit_params

        self.FIGSIZE = (5, 5)
        self.MARKER_SIZE = 60
        self.ALPHA = 0.9
        self.LINEWIDTH = 2.0
        self.ATOL = 1e-9


# ===========================================================
# 近似処理
# ===========================================================
def fit_predict_curve(x, y, cfg):
    if len(x) < 2:
        return None, None

    xl = np.linspace(x.min(), x.max(), 200)
    m = cfg.FIT_METHOD
    p = cfg.FIT_PARAMS

    if m == "akima":
        return xl, Akima1DInterpolator(x, y)(xl)

    if m == "pchip":
        return xl, PchipInterpolator(x, y)(xl)

    if m == "spline" and len(x) > 3:
        return xl, UnivariateSpline(x, y, s=p["s"])(xl)

    if m == "savgol":
        idx = np.argsort(x)
        return np.sort(x), savgol_filter(y[idx], p["window"], p["poly"])

    if m == "loess":
        out = lowess(y, x, frac=p["frac"])
        return out[:, 0], out[:, 1]

    return None, None


# ===========================================================
# サンプル描画（元コード忠実）
# ===========================================================
def process_sample(ax, df, base_row, idx, cfg, legend_done):
    x = pd.to_numeric(df.iloc[base_row + 1], errors="coerce").to_numpy()
    y1 = pd.to_numeric(df.iloc[base_row + 2], errors="coerce").to_numpy()
    y2 = pd.to_numeric(df.iloc[base_row + 3], errors="coerce").to_numpy()

    label = (
        str(df.iloc[base_row, 9])
        if df.shape[1] > 9 and not pd.isna(df.iloc[base_row, 9])
        else f"Sample{idx+1}"
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P"]
    color = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]

    for y in (y1, y2):
        main = (
            np.isfinite(y)
            & (~np.isclose(y, cfg.UPPER, atol=cfg.ATOL))
            & (~np.isclose(y, cfg.LOWER, atol=cfg.ATOL))
        )
        upper = np.isclose(y, cfg.UPPER, atol=cfg.ATOL)
        lower = np.isclose(y, cfg.LOWER, atol=cfg.ATOL)

        if np.any(main):
            ax.scatter(
                x[main], y[main],
                s=cfg.MARKER_SIZE,
                marker=marker,
                color=color,
                alpha=cfg.ALPHA,
                label=label if label not in legend_done else "_nolegend_",
                clip_on=False,
                zorder=3,
            )
            legend_done.add(label)

            xl, yl = fit_predict_curve(x[main], y[main], cfg)
            if xl is not None:
                ax.plot(xl, yl, color=color, lw=cfg.LINEWIDTH, zorder=2)

        ax.scatter(
            x[upper], y[upper],
            s=cfg.MARKER_SIZE,
            marker=marker,
            color=color,
            clip_on=False,
            zorder=4,
        )
        ax.scatter(
            x[lower], y[lower],
            s=cfg.MARKER_SIZE,
            marker=marker,
            color=color,
            clip_on=False,
            zorder=4,
        )


# ===========================================================
# メイン解析
# ===========================================================
def run_analysis(file, cfg):
    df = pd.read_excel(file, header=None)
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE)

    legend_done = set()

    for i, r in enumerate(range(1, len(df), 4)):
        process_sample(ax, df, r, i, cfg, legend_done)

    ax.set_xlim(0, 100)
    ax.set_ylim(cfg.LOWER, cfg.UPPER)

    ax.set_xlabel("油分比率 (wt%)", fontproperties=cfg.FONT)
    ax.set_ylabel("二層分離温度 (°C)", fontproperties=cfg.FONT)

    ax.tick_params(direction="in", which="both", length=6, width=1)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(prop=cfg.FONT)

    plt.tight_layout()
    return fig


# ===========================================================
# Streamlit UI
# ===========================================================
st.set_page_config(page_title="二層分離温度解析", layout="centered")
st.title("二層分離温度解析アプリ")

# --- フォント ---
font_ui = st.sidebar.selectbox(
    "フォント",
    ["日本語（IPAexGothic）", "英語（Times New Roman）"]
)
font_prop = load_font(font_ui)

# --- 軸範囲 ---
upper = st.sidebar.number_input("上限温度 (°C)", value=70.0)
lower = st.sidebar.number_input("下限温度 (°C)", value=-50.0)

# --- 近似方式 ---
fit_method = st.sidebar.selectbox(
    "近似方式",
    ["akima", "pchip", "spline", "savgol", "loess"]
)

# --- パラメータ ---
fit_params = {}
if fit_method == "spline":
    fit_params["s"] = st.sidebar.number_input("Spline s", value=5.0)
elif fit_method == "savgol":
    fit_params["window"] = st.sidebar.number_input("Window (odd)", 5, 31, 7, step=2)
    fit_params["poly"] = st.sidebar.number_input("Poly order", 1, 5, 2)
elif fit_method == "loess":
    fit_params["frac"] = st.sidebar.slider("LOESS frac", 0.1, 0.8, 0.35)

uploaded = st.file_uploader(
    "Excelファイルをアップロード",
    type=["xlsx", "xlsm"]
)

if uploaded:
    cfg = Config(upper, lower, font_prop, fit_method, fit_params)
    fig = run_analysis(uploaded, cfg)

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)

    st.download_button(
        "PNG画像をダウンロード",
        data=buf,
        file_name="two_layer_temperature.png",
        mime="image/png",
    )
else:
    st.info("Excelファイルをアップロードしてください")
