"""Visualization functions for engagement weight analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_heatmap(
    results_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Create a heatmap showing Gamma (Shares) importance across combinations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from analyze_engagement_weights.
    save_path : Path, optional
        Path to save the figure. If None, displays the figure.
    figsize : tuple[int, int], default (10, 6)
        Figure size (width, height).
    """
    # Pivot table for heatmap
    pivot_df = results_df.pivot_table(
        values="Gamma_Shares",
        index="Platform",
        columns="Post Type",
        aggfunc="mean",
    )

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "Gamma (Shares) Weight"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title(
        "Heatmap: Importance of Shares (Gamma) Across Platform-PostType "
        "Combinations",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Post Type", fontsize=12, fontweight="bold")
    plt.ylabel("Platform", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Heatmap saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def create_facet_grid_chart(
    results_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """
    Create a facet grid showing weight changes across Post Types for each Platform.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from analyze_engagement_weights.
    save_path : Path, optional
        Path to save the figure. If None, displays the figure.
    figsize : tuple[int, int], default (15, 10)
        Figure size (width, height).
    """
    # Melt DataFrame for easier plotting
    weight_cols = ["Alpha_Likes", "Beta_Comments", "Gamma_Shares"]
    id_cols = ["Platform", "Post Type"]

    plot_df = results_df.melt(
        id_vars=id_cols,
        value_vars=weight_cols,
        var_name="Weight_Type",
        value_name="Weight_Value",
    )

    # Map weight types to readable names
    weight_mapping = {
        "Alpha_Likes": "Likes (α)",
        "Beta_Comments": "Comments (β)",
        "Gamma_Shares": "Shares (γ)",
    }
    plot_df["Weight_Type"] = plot_df["Weight_Type"].map(weight_mapping)

    # Create facet grid
    platforms = sorted(results_df["Platform"].unique())
    n_platforms = len(platforms)

    fig, axes = plt.subplots(n_platforms, 1, figsize=figsize, sharex=True, sharey=True)

    if n_platforms == 1:
        axes = [axes]

    colors = {
        "Likes (α)": "#3498db",
        "Comments (β)": "#e74c3c",
        "Shares (γ)": "#2ecc71",
    }

    for idx, platform in enumerate(platforms):
        ax = axes[idx]
        platform_data = plot_df[plot_df["Platform"] == platform]

        # Create grouped bar chart
        post_types = sorted(platform_data["Post Type"].unique())
        x_pos = np.arange(len(post_types))
        width = 0.25

        for weight_idx, weight_type in enumerate(
            ["Likes (α)", "Comments (β)", "Shares (γ)"]
        ):
            weight_data = platform_data[platform_data["Weight_Type"] == weight_type]
            values = [
                weight_data[weight_data["Post Type"] == pt]["Weight_Value"].values[0]
                if len(weight_data[weight_data["Post Type"] == pt]) > 0
                else 0
                for pt in post_types
            ]

            ax.bar(
                x_pos + weight_idx * width,
                values,
                width,
                label=weight_type,
                color=colors[weight_type],
                alpha=0.8,
            )

        ax.set_title(f"{platform}", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Normalized Weight", fontsize=10)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(post_types)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim([0, 1])

    plt.suptitle(
        "Engagement Weights Across Post Types by Platform",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.xlabel("Post Type", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Facet grid chart saved to: {save_path}")
    else:
        plt.show()

    plt.close()
