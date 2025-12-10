import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_iou_csv(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    rows = []
    current_model = None
    collecting = False

    for _, row in df_raw.iterrows():
        first = row.get("Unnamed: 0")

        if isinstance(first, str) and first.strip() in ["YOLOv8-seg", "SAM 3", "VGGT + CLIP"]:
            current_model = first.strip()
            collecting = False
            continue

        if isinstance(first, str) and first.strip() == "IoUs:":
            collecting = True
            continue

        if collecting:
            if not isinstance(first, str):
                collecting = False
                continue

            weather = first.strip()
            car = float(row["Unnamed: 1"])
            truck = float(row["Unnamed: 2"])
            tlight = float(row["Unnamed: 3"])
            miou = float(row["Unnamed: 4"])

            rows.append(
                {
                    "Algorithm": current_model,
                    "Weather": weather,
                    "Car": car,
                    "Truck": truck,
                    "Traffic Light": tlight,
                    "mIoU": miou,
                }
            )

    df = pd.DataFrame(rows)

    weather_order = ["Clone", "Morning", "Rain", "Sunset", "Fog", "Overcast"]
    df["Weather"] = pd.Categorical(df["Weather"], categories=weather_order, ordered=True)
    df = df.sort_values(["Algorithm", "Weather"]).reset_index(drop=True)

    return df


def plot_per_class_bars(df: pd.DataFrame, class_name: str, out_path: str):
    algorithms = ["YOLOv8-seg", "SAM 3", "VGGT + CLIP"]
    weather_order = ["Clone", "Morning", "Rain", "Sunset", "Fog", "Overcast"]

    # Custom colors (RGB 0–255 → 0–1)
    color_map = {
        "YOLOv8-seg": (207/255, 89/255, 76/255),   # reddish
        "SAM 3":      (24/255, 119/255, 242/255),  # blue
        "VGGT + CLIP": (99/255, 201/255, 143/255), # use default color cycle (green-ish)
    }

    x = np.arange(len(weather_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all values to compute tight y-limits
    all_vals = []

    for i, algo in enumerate(algorithms):
        subset = (
            df[df["Algorithm"] == algo]
            .set_index("Weather")
            .reindex(weather_order)
        )

        y_vals = subset[class_name].values.astype(float)
        all_vals.extend(y_vals.tolist())

        offset = (i - 1) * width
        color = color_map[algo]
        ax.bar(x + offset, y_vals, width, label=algo, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(weather_order)
    ax.set_ylabel("IoU")
    ax.set_xlabel("Weather")
    ax.set_title(f"{class_name} IoUs")

    # Zoom y-axis around data range to exaggerate differences (but not lie)
    all_vals = np.array(all_vals)
    ymin = float(all_vals.min())
    ymax = float(all_vals.max())
    span = ymax - ymin
    if span < 0.05:
        span = 0.05  # avoid degenerate zoom

    pad = 0.15 * span
    ymin_plot = max(0.0, ymin - pad)
    ymax_plot = min(1.0, ymax + pad)
    ax.set_ylim(ymin_plot, ymax_plot)

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(csv_path: str):
    df = parse_iou_csv(csv_path)

    for cls in ["Car", "Truck", "Traffic Light"]:
        out_name = f"iou_{cls.replace(' ', '_').lower()}_by_weather.png"
        plot_per_class_bars(df, cls, out_name)
        print(f"Saved {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_segmentation_iou.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])
