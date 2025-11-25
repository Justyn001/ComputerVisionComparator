import matplotlib.pyplot as plt
import pathlib
from typing import Dict, Any


def generate_plots(results: Dict[str, Dict[str, Any]], output_dir: pathlib.Path):
    """
    Generuje wykresy porównawcze. Wersja poprawiona dla płaskiej struktury danych.
    """
    print("📊 Generowanie wykresów...")

    # Upewnij się, że folder istnieje
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Przygotowanie danych
    models = []
    fps_values = []
    map_values = []

    for model_name, data in results.items():
        # === POPRAWIONA LOGIKA POBIERANIA FPS ===
        fps_list = []
        map50 = 0.0

        for key, val in data.items():
            # Jeśli to klucz z dokładnością, zapisz mAP i idź dalej
            if key == "accuracy_benchmark":
                if isinstance(val, dict):
                    map50 = val.get('mAP50', 0.0)
                continue

            # Jeśli to nie accuracy, to zakładamy, że to wynik wideo (słownik metryk)
            if isinstance(val, dict):
                fps = val.get('system_fps', 0)
                fps_list.append(fps)

        # Oblicz średni FPS dla modelu
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0

        models.append(model_name)
        fps_values.append(avg_fps)
        map_values.append(map50)

    # === WYKRES 1: Porównanie Szybkości (FPS) ===
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, fps_values, color='skyblue', edgecolor='black')

    plt.title('Average System FPS per Model', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('FPS (Frames Per Second)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Dodaj wartości nad słupkami
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.01), f'{yval:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "chart_fps_comparison.png")
    plt.close()

    # === WYKRES 2: Porównanie Dokładności (mAP50) ===
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, map_values, color='salmon', edgecolor='black')

    plt.title('Model Accuracy (mAP @ IoU=0.50)', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('mAP50', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)  # mAP jest zawsze 0-1 (dajemy zapas na etykiety)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "chart_accuracy_comparison.png")
    plt.close()

    # === WYKRES 3: Trade-off (Szybkość vs Dokładność) ===
    plt.figure(figsize=(10, 8))
    plt.scatter(fps_values, map_values, color='purple', s=150, alpha=0.7, edgecolors='black')

    plt.title('Speed vs. Accuracy Trade-off', fontsize=16)
    plt.xlabel('Speed (FPS) -> Faster', fontsize=12)
    plt.ylabel('Accuracy (mAP50) -> Better', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Ustawienie limitów osi z lekkim marginesem
    if fps_values:
        plt.xlim(0, max(fps_values) * 1.15)
    plt.ylim(0, 1.1)

    # Podpisz punkty nazwami modeli
    for i, txt in enumerate(models):
        plt.annotate(txt, (fps_values[i], map_values[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "chart_tradeoff.png")
    plt.close()

    print(f"✅ Wykresy zapisane w: {output_dir.absolute()}")