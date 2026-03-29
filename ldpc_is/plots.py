from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_boundary_distances(df: pd.DataFrame):
    """Plot alpha*^2 distances by trapping set index."""
    fig, ax = plt.subplots()
    ax.scatter(df["ts_index"], df["squared_distance"])
    ax.set_xlabel("TS index")
    ax.set_ylabel("Boundary squared distance")
    ax.set_title("Boundary ranking")
    return fig


def plot_component_contributions(contrib_df: pd.DataFrame):
    """Plot weighted component contribution to FER and BER."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(contrib_df["comp"], contrib_df["wfe"])
    ax[0].set_title("Weighted FE contribution")
    ax[1].bar(contrib_df["comp"], contrib_df["wbe"])
    ax[1].set_title("Weighted BE contribution")
    return fig


def plot_weight_histogram(weights):
    """Plot histogram of IS weights."""
    fig, ax = plt.subplots()
    ax.hist(weights, bins=40)
    ax.set_xlabel("importance weight")
    ax.set_ylabel("count")
    return fig


def plot_ber_fer_vs_ebn0(df: pd.DataFrame):
    """Plot BER and FER versus Eb/N0 from a dataframe."""
    fig, ax = plt.subplots()
    ax.semilogy(df["ebn0_db"], df["BER_hat"], marker="o", label="BER")
    ax.semilogy(df["ebn0_db"], df["FER_hat"], marker="s", label="FER")
    ax.legend()
    ax.set_xlabel("Eb/N0 (dB)")
    return fig


def plot_mc_vs_is_comparison(df: pd.DataFrame):
    """Plot MC and IS estimates for comparison."""
    fig, ax = plt.subplots()
    ax.semilogy(df["ebn0_db"], df["FER_is"], marker="o", label="FER IS")
    ax.semilogy(df["ebn0_db"], df["FER_mc"], marker="s", label="FER MC")
    ax.legend()
    ax.set_xlabel("Eb/N0 (dB)")
    return fig
