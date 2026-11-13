# FILE PURPOSE: Inspect native source images, unreviewed pixel observations and
# future-dependent source smoothing. This is not an accuracy/control-benefit plot.
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while part := stream.read(4 * 1024**2):
            digest.update(part)
    return digest.hexdigest()


def plot(result, index, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((result / "protocol.json").read_text())
    observations = [
        json.loads(line)
        for line in (result / "observations.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((index / "manifest.json").read_text())
    asset = next(
        r
        for r in manifest["protocol"]["archives"]
        if r["archive_id"] == protocol["archive_id"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    with zipfile.ZipFile(Path(manifest["cache_path"]) / asset["filename"]) as archive:
        for ax, row, label in zip(
            axes[0], [observations[0], observations[-1]], ["First", "Last"]
        ):
            payload = archive.read(row["original_member"])
            assert hashlib.sha256(payload).hexdigest() == row["source_image_sha256"]
            rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
            ax.imshow(rgb)
            points = np.asarray(row["contour_xy"])
            ax.plot(
                *np.vstack([points, points[0]]).T,
                color="#00d9ff",
                linewidth=1.4,
                label="Automatic contour",
            )
            ax.set_title(
                f"{label} native source view · {row['filename_elapsed_s'] / 3600:.2f} h"
            )
            ax.set_axis_off()
            ax.legend(loc="lower left", fontsize=9)
    rows = list(csv.DictReader(io.StringIO((result / "source_series.csv").read_text())))
    log_time = np.array([float(r["Time (h)"]) for r in rows])
    image_time = np.array([r["filename_elapsed_s"] / 3600 for r in observations])
    size = np.sqrt([r["contour_area_px2"] for r in observations])
    ax = axes[1, 0]
    line = ax.plot(
        image_time, size, color="#126190", linewidth=1.6, label="√ contour area (px)"
    )
    ax.set(
        xlabel="Filename elapsed time (h)",
        ylabel="√ contour area (px)",
        title="Recorded co-variation, not an action-effect estimate",
    )
    right = ax.twinx()
    line += right.plot(
        log_time,
        [float(r["Infusion Rate (mL/h)"]) for r in rows],
        color="#b34f18",
        alpha=0.7,
        linewidth=0.9,
        label="Reported infusion (mL/h)",
    )
    right.set_ylabel("Reported infusion (mL/h)", color="#b34f18")
    ax.legend(line, [v.get_label() for v in line], fontsize=8, loc="upper left")
    settings = protocol["smoothing_diagnostic"]
    raw = np.array([float(r["Length (mm)"]) for r in rows])
    changed = raw.copy()
    cutoff = settings["perturbation_cutoff_row"]
    changed[cutoff + 1 :] += settings["perturbation_delta_mm"]
    window = settings["candidate_effective_window"]
    smooth = savgol_filter(raw, window, settings["candidate_polynomial_order"])
    changed_smooth = savgol_filter(
        changed, window, settings["candidate_polynomial_order"]
    )
    around = abs(np.arange(len(rows)) - cutoff) <= window
    ax = axes[1, 1]
    ax.plot(
        log_time[around],
        (changed_smooth - smooth)[around],
        color="#9a335b",
        label="Change in smoothed history",
    )
    ax.axvline(
        log_time[cutoff],
        color="#333333",
        linestyle="--",
        label="Only later raw values changed",
    )
    ax.axvspan(log_time[cutoff], log_time[around][-1], color="#dddddd", alpha=0.6)
    ax.set(
        xlabel="Source log elapsed time (h)",
        ylabel="Change (source-reported mm)",
        title="Future-only +0.5 mm edit changes earlier estimates",
    )
    ax.legend(fontsize=8, loc="upper left")
    for ax in axes[1]:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Public MAPbBr₃ run: source-clock association and forecasting-input audit",
        fontsize=15,
    )
    fig.supxlabel(
        "Automatic contours are unreviewed proxies. Scale and hardware timing are unverified. All examples are development only.",
        fontsize=10,
    )
    fig.savefig(output / "process_audit.png", dpi=160)
    fig.savefig(output / "process_audit.svg")
    plt.close(fig)
    shutil.copyfile(Path(__file__), output / "plotter_snapshot.py")
    (output / "source.json").write_text(
        json.dumps(
            dict(
                result_manifest_sha256=sha(result / "manifest.json"),
                protocol_sha256=sha(result / "protocol.json"),
                plotter_sha256=sha(Path(__file__)),
                model_accuracy_claim=False,
                action_effect_claim=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(output / "process_audit.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--index", type=Path, default=ROOT / "data/real_vision_crystalcv_v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.result, args.index, args.output)
