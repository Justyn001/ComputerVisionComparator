import matplotlib.pyplot as plt
import pathlib
from typing import Dict, Any


def generate_plots(results: Dict[str, Dict[str, Any]], output_dir: pathlib.Path):
    print("Generating plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    models = []
    fps_values = []
    map_values = []

    for model_name, data in results.items():
        fps_list = []
        map50 = 0.0

        for key, val in data.items():
            if key == "accuracy_benchmark":
                if isinstance(val, dict):
                    map50 = val.get('mAP50', 0.0)
                continue

            if isinstance(val, dict):
                fps = val.get('system_fps', 0)
                fps_list.append(fps)

        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0

        models.append(model_name)
        fps_values.append(avg_fps)
        map_values.append(map50)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, fps_values, color='skyblue', edgecolor='black')

    plt.title('Average System FPS per Model', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('FPS (Frames Per Second)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.01), f'{yval:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "chart_fps_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, map_values, color='salmon', edgecolor='black')

    plt.title('Model Accuracy (mAP @ IoU=0.50)', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('mAP50', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "chart_accuracy_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(fps_values, map_values, color='purple', s=150, alpha=0.7, edgecolors='black')

    plt.title('Speed vs. Accuracy Trade-off', fontsize=16)
    plt.xlabel('Speed (FPS) -> Faster', fontsize=12)
    plt.ylabel('Accuracy (mAP50) -> Better', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    if fps_values:
        plt.xlim(0, max(fps_values) * 1.15)
    plt.ylim(0, 1.1)

    for i, txt in enumerate(models):
        plt.annotate(txt, (fps_values[i], map_values[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "chart_tradeoff.png")
    plt.close()

    print(f"Chart saved in: {output_dir.absolute()}")