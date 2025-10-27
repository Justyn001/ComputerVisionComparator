# ComputerVisionComparator/
# |
# |-- benchmark.py         # Główny skrypt (Menedżer)
# |-- data/                # Folder z filmami użytkownika
# |-- report.xml           # Wygenerowany raport
# |
# `-- runners/             # <-- TU JEST MAGIA
#     |-- __init__.py
#     |-- base_runner.py     # Definicja "interfejsu" dla każdego modelu
#     |-- ultralytics_runner.py  # "Adapter" dla modeli YOLO (v5, v8, v11)
#     |-- torchvision_runner.py  # "Adapter" dla modeli z Torchvision (SSD, Faster R-CNN)
#     |-- (inny_runner.py)   # W przyszłości adapter dla modeli z TensorFlow Hub...

import argparse
import os
import sys
import pathlib
from typing import List
from runners.ultralytics_runner import predict_on_yolo
from utils.download_data import download_data
from utils.report_generator import generate_report

def main(args: argparse.Namespace) -> None:
    path_to_videos = pathlib.Path.cwd() / "data"

    if not path_to_videos.is_dir():
        os.mkdir(path_to_videos)
        download_data(path_to_videos)

    video_list: List[pathlib] = [video for video in list(path_to_videos.glob("*.mp4"))]

    predicted_time = predict_on_yolo(video_list, args.model)
    print(predicted_time)
    generate_report(predicted_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description= "CLI tool to benchmark object detection models' performance."
    )
    parser.add_argument("-m",
                        "--model",
                        nargs="+",
                        type=str,
                        required=True,
                        help="One or more model names to test (e.g., yolov8n.pt ssd_mobilenet_v2)")
    try:
        args = parser.parse_args()
        main(args)
    except SystemExit as e:
        print(f"ERROR: You must provide at least one model to test.")
        print("Usage: python benchmark.py --model yolov8n.pt")
        sys.exit(-1)