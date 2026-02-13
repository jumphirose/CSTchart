#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.ticker import MultipleLocator
from scipy.interpolate import Akima1DInterpolator


# ===========================================================
# Config
# ===========================================================
class Config:
    def __init__(self, upper, lower, font):
        self.UPPER = upper
        self.LOWER = lower
        self.FONT = font

        self.FIGSIZE = (5, 5)
        self.MARKER_SIZE = 60
        self.ALPHA = 0.9
        self.LINEWIDTH = 2.0
        self.ATOL = 1e-9


# ===========================================================
# サンプル描画（元スクリプト忠実再現）
# ===========================================================
def process_sample(ax, df, base_row, idx, cfg, legend_done):
    x = pd.to_numeric(df.iloc[base_row + 1], errors="coerce").to_numpy()
    y1 = pd.to_numeric(df.iloc[base_row + 2], errors="coerce").to_numpy()
    y2 = pd.to_numeric(df.iloc[base_row + 3], errors="coerce").to_numpy()

    # --- サンプル名（Excel由来） ---
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

        # --- 通常点（凡例は1回のみ） ---
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

            # 近似（端点除外）
            xl = np.linspace(x[main].min(), x[main].max(), 200)
            yl = Akima1DInterpolator(x[main], y[main])(xl)
            ax.plot(xl, yl, color=color, lw=cfg.LINEWIDTH, zorder=2)

        # --- 上限・下限点（同色・同マーカー） ---
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
    plt.rcParams["font.family"] = cfg.FONT
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_excel(file, header=None)
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE)

    legend_done = set()

    for i, r in enumerate(range(1, len(df), 4)):
        process_sample(ax, df, r, i, cfg, legend_done)

    # --- 軸設定（元コードと一致） ---
    ax.set_xlim(0, 100)
    ax.set_ylim(cfg.LOWER, cfg.UPPER)
    ax.set_xlabel("油分比率 (wt%)")
    ax.set_ylabel("二層分離温度 (°C)")

    ax.tick_params(direction="in", which="both", length=6, width=1)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend()
    plt.tight_layout()

    return fig


# ===========================================================
# Streamlit UI
# ===========================================================
st.set_page_config(page_title="二層分離温度解析")
st.title("二層分離温度解析アプリ")

# --- フォント選択 ---
font = st.sidebar.selectbox(
    "フォント",
    ["Meiryo", "Times New Roman"]
)

# --- 上下限（数値入力） ---
upper = st.sidebar.number_input("上限温度 (°C)", value=70.0)
lower = st.sidebar.number_input("下限温度 (°C)", value=-50.0)

uploaded = st.file_uploader(
    "Excelファイルをアップロード",
    type=["xlsx", "xlsm"]
)

if uploaded:
    cfg = Config(upper, lower, font)
    fig = run_analysis(uploaded, cfg)

    st.pyplot(fig)

    # --- PNGダウンロード ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)

    st.download_button(
        label="PNG画像をダウンロード",
        data=buf,
        file_name="two_layer_temperature.png",
        mime="image/png",
    )
else:
    st.info("Excelファイルをアップロードしてください")
