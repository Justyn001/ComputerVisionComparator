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

# Rodzinę YOLO (super balans)
#
# FasterRCNN (skrajna dokładność)
#
# SSDLite (skrajna lekkość)
#
# SSD300 (stary standard)
#
# RetinaNet (nowoczesny "złoty środek")


import argparse
import os
import sys
import pathlib
from typing import List, Dict
from runners.ultralytics_runner import predict_on_yolo
from runners.torchvision_runner import predict_on_pytorch
from utils.download_data import download_data
from utils.report_generator import generate_report

def main(args: argparse.Namespace) -> None:
    path_to_videos = pathlib.Path.cwd() / "data"

    if not path_to_videos.is_dir():
        os.mkdir(path_to_videos)
        download_data(path_to_videos)

    video_list: List[pathlib] = [video for video in list(path_to_videos.glob("*.mp4"))]

    results: Dict[str, Dict[str, Dict]] = {}

    yolo_models = [
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
        "yolo11n",
        "yolo11s",
        "yolo11m",
        "yolo11l",
        "yolo11x",
        "yolov5n",
        "yolov5s",
        "yolov5m",
        ]

    pytorch_models = [
        "fcos_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_resnet50_fpn",
        "retinanet_resnet50_fpn_v2",
        "retinanet_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
    ]

    for model in args.model:
        if model.lower() in yolo_models:
            results[model] = predict_on_yolo(video_list, model.lower())
        elif model.lower() in pytorch_models:
            results[model] = predict_on_pytorch(video_list, model.lower())
    #results = predict_on_pytorch(video_list, args.model, results)
    #predicted_time = predict_on_yolo(video_list, args.model, results)

    hardware 
    generate_report(results)

    print(results)

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