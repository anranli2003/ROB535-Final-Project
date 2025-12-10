import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


WEATHER_ORDER = ["Clone", "Morning", "Rain", "Sunset", "Fog", "Overcast"]
CLASS_ORDER = ["sky", "tree", "building", "road",
               "trafficsign", "trafficlight", "pole",
               "truck", "car", "van", "mIoU"]

# Consistent colors
SAM_COLOR = (24/255, 119/255, 242/255)      # blue
VGGT_COLOR = (99/255, 201/255, 143/255)     # green-ish


def parse_sam_vggt_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse the SAM 3 vs VGGT+CLIP CSV into a tidy DataFrame with columns:
      Algorithm, Weather, Class, IoU
    """
    df_raw = pd.read_csv(csv_path)
    first_col = df_raw.columns[0]

    header_rows = df_raw.index[df_raw[first_col] == "IoUs:"].tolist()
    rows = []

    for idx, hi in enumerate(header_rows):
        # Figure out algorithm name
        if hi == 0:
            algo = first_col.strip()      # "SAM 3"
        else:
            algo = str(df_raw.loc[hi - 1, first_col]).strip()  # "VGGT + CLIP"

        # Class names (from columns 1.. until NaN)
        class_names = []
        for col in df_raw.columns[1:]:
            val = df_raw.loc[hi, col]
            if isinstance(val, str) and val.strip():
                class_names.append(val.strip())
            elif pd.notna(val):
                class_names.append(str(val))

        # Find where this block ends
        if idx + 1 < len(header_rows):
            end = header_rows[idx + 1]
        else:
            end = len(df_raw)

        # Weather rows
        for r in range(hi + 1, end):
            weather = df_raw.loc[r, first_col]
            if not isinstance(weather, str) or not weather.strip():
                continue
            weather = weather.strip()

            for j, col in enumerate(df_raw.columns[1:1 + len(class_names)]):
                cls = class_names[j]
                val = df_raw.loc[r, col]
                if pd.isna(val):
                    continue
                rows.append(
                    {
                        "Algorithm": algo,
                        "Weather": weather,
                        "Class": cls,
                        "IoU": float(val),
                    }
                )

    df = pd.DataFrame(rows)
    # Enforce consistent ordering
    df["Weather"] = pd.Categorical(df["Weather"],
                                   categories=WEATHER_ORDER,
                                   ordered=True)
    df["Class"] = pd.Categorical(df["Class"],
                                 categories=CLASS_ORDER,
                                 ordered=True)
    return df


def plot_per_class_mean_iou(df: pd.DataFrame, out_path: str):
    """
    Figure 1: per-class mean IoU across all weathers,
    two bars per class (SAM 3 vs VGGT+CLIP).
    """
    df_no_miou = df[df["Class"] != "mIoU"]

    mean_by_class_algo = (
        df_no_miou
        .groupby(["Class", "Algorithm"])["IoU"]
        .mean()
        .unstack("Algorithm")
        .loc[[c for c in CLASS_ORDER if c != "mIoU"]]
    )

    classes = mean_by_class_algo.index.tolist()
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2,
           mean_by_class_algo["SAM 3"].values,
           width,
           label="SAM 3",
           color=SAM_COLOR)

    ax.bar(x + width/2,
           mean_by_class_algo["VGGT + CLIP"].values,
           width,
           label="VGGT + CLIP",
           color=VGGT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Per-class Mean IoUs (Averaged Across Weathers)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_miou_delta_vs_clone(df: pd.DataFrame, out_path: str):
    """
    NEW Figure 2: heatmap of Δ mIoU vs Clone for each algorithm.
    Rows: algorithms (SAM 3, VGGT + CLIP)
    Columns: weathers excluding Clone (Morning..Overcast)
    Value: mIoU(weather) - mIoU(Clone)
    Green = better than Clone, Red = worse than Clone.
    """
    miou_df = (
        df[df["Class"] == "mIoU"]
        .pivot(index="Algorithm", columns="Weather", values="IoU")
        .reindex(columns=WEATHER_ORDER)
    )

    # Compute delta vs Clone
    baseline = miou_df["Clone"]
    deltas = miou_df.sub(baseline, axis=0)

    # Drop Clone column for plotting (it's all zeros by definition)
    deltas = deltas.loc[:, ["Morning", "Rain", "Sunset", "Fog", "Overcast"]]

    data = deltas.values
    max_abs = float(np.nanmax(np.abs(data)))
    if max_abs == 0:
        max_abs = 0.05

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Green = better, Red = worse
    im = ax.imshow(data, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(np.arange(deltas.shape[1]))
    ax.set_xticklabels(deltas.columns)
    ax.set_yticks(np.arange(deltas.shape[0]))
    ax.set_yticklabels(deltas.index)

    ax.set_xlabel("Weather (vs Clone)")
    ax.set_ylabel("Algorithm")
    ax.set_title("Δ mIoU Relative to Clone Weather")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Δ mIoU (weather − Clone)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_delta_iou_heatmap(df: pd.DataFrame, out_path: str):
    """
    Figure 3: heatmap of ΔIoU = SAM 3 - VGGT+CLIP
    for each class (rows) and weather (columns),
    using a custom colormap consistent with SAM (blue) and VGGT (green).
    """
    df_no_miou = df[df["Class"] != "mIoU"]

    sam = (
        df_no_miou[df_no_miou["Algorithm"] == "SAM 3"]
        .pivot(index="Class", columns="Weather", values="IoU")
        .reindex(index=[c for c in CLASS_ORDER if c != "mIoU"],
                 columns=WEATHER_ORDER)
    )
    vggt = (
        df_no_miou[df_no_miou["Algorithm"] == "VGGT + CLIP"]
        .pivot(index="Class", columns="Weather", values="IoU")
        .reindex(index=[c for c in CLASS_ORDER if c != "mIoU"],
                 columns=WEATHER_ORDER)
    )

    delta = sam - vggt
    data = delta.values

    max_abs = float(np.nanmax(np.abs(data)))
    if max_abs == 0:
        max_abs = 0.1  # avoid degenerate color scale

    # Custom diverging colormap: VGGT green -> white -> SAM blue
    cmap = LinearSegmentedColormap.from_list(
        "sam_vggt_delta",
        [VGGT_COLOR, (1, 1, 1), SAM_COLOR],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap, vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(np.arange(len(WEATHER_ORDER)))
    ax.set_xticklabels(WEATHER_ORDER)
    ax.set_yticks(np.arange(len(sam.index)))
    ax.set_yticklabels(sam.index)

    ax.set_xlabel("Weather")
    ax.set_ylabel("Class")
    ax.set_title("ΔIoU (SAM 3 − VGGT + CLIP) Across Class and Weather")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ΔIoU (SAM 3 − VGGT + CLIP)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(csv_path: str):
    df = parse_sam_vggt_csv(csv_path)

    plot_per_class_mean_iou(df, "sam_vggt_per_class_mean_iou.png")
    print("Saved sam_vggt_per_class_mean_iou.png")

    plot_miou_delta_vs_clone(df, "sam_vggt_miou_delta_vs_clone.png")
    print("Saved sam_vggt_miou_delta_vs_clone.png")

    plot_delta_iou_heatmap(df, "sam_vggt_delta_iou_heatmap.png")
    print("Saved sam_vggt_delta_iou_heatmap.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_sam_vs_vggt.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])
