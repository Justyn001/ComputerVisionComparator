import matplotlib.pyplot as plt
import pathlib
from typing import Dict, Any


def generate_plots(results: Dict[str, Dict[str, Any]], output_dir: pathlib.Path):
    print("Generating plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    models = []
    fps_averages = []
    map_values = []
    fps_distributions = []
    latency_values = []

    for model_name, data in results.items():
        fps_list_for_model = []
        map50 = 0.0

        for key, val in data.items():
            if key == "accuracy_benchmark":
                if isinstance(val, dict):
                    map50 = val.get('mAP50', 0.0)
                continue

            if isinstance(val, dict):
                fps = val.get('system_fps', 0)
                if fps > 0:
                    fps_list_for_model.append(fps)

        if fps_list_for_model:
            avg_fps = sum(fps_list_for_model) / len(fps_list_for_model)
            avg_latency = 1000.0 / avg_fps
        else:
            avg_fps = 0.0
            avg_latency = 0.0

        models.append(model_name)
        fps_averages.append(avg_fps)
        map_values.append(map50)
        fps_distributions.append(fps_list_for_model)
        latency_values.append(avg_latency)

    plt.style.use('bmh')

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, fps_averages, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title('Average System FPS per Model', fontsize=14)
    plt.ylabel('FPS (Higher is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "chart_01_fps_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, map_values, color='salmon', edgecolor='black', alpha=0.8)
    plt.title('Model Accuracy (mAP @ IoU=0.50)', fontsize=14)
    plt.ylabel('mAP50 (Higher is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "chart_02_accuracy_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(fps_averages, map_values, color='purple', s=150, alpha=0.7, edgecolors='black', zorder=2)
    plt.title('Speed vs. Accuracy Trade-off', fontsize=16)
    plt.xlabel('Speed (FPS) -> Faster', fontsize=12)
    plt.ylabel('Accuracy (mAP50) -> Better', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    for i, txt in enumerate(models):
        plt.annotate(txt, (fps_averages[i], map_values[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

    if fps_averages:
        plt.xlim(0, max(fps_averages) * 1.15)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "chart_03_tradeoff.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, latency_values, color='#ffcc5c', edgecolor='black', alpha=0.9)
    plt.title('Average Inference Latency', fontsize=14)
    plt.ylabel('Time per Frame (ms) (Lower is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f} ms', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "chart_04_latency_ms.png")
    plt.close()

    if any(fps_distributions):
        plt.figure(figsize=(10, 6))
        valid_distributions = [dist for dist in fps_distributions if dist]
        valid_labels = [models[i] for i, dist in enumerate(fps_distributions) if dist]
        
        plt.boxplot(valid_distributions, labels=valid_labels, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="black"),
                    medianprops=dict(color="red", linewidth=1.5))
        
        plt.title('FPS Stability Distribution (Boxplot)', fontsize=14)
        plt.ylabel('FPS', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / "chart_05_fps_stability_boxplot.png")
        plt.close()

    print(f"Charts saved in: {output_dir.absolute()}")