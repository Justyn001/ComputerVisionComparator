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

# test gita na nowej fedorze

import argparse
import os
import sys
import pathlib
from typing import List, Dict, Any
from runners.ultralytics_runner import predict_on_yolo, validate_yolo
from runners.torchvision_runner import predict_on_pytorch, validate_pytorch
from utils.download_data import download_data
from utils.hardware_info import get_hardware_info
from utils.report_generator import generate_report
from utils.plot_generator import generate_plots
from utils.live_demo import  run_live_demo

def main(args: argparse.Namespace) -> None:
    path_to_videos = pathlib.Path.cwd() / "data"

    if not path_to_videos.is_dir():
        os.mkdir(path_to_videos)
        download_data(path_to_videos)

    video_list: List[pathlib] = [video for video in list(path_to_videos.glob("*.mp4"))]

    path_coco = pathlib.Path.cwd() / "data" / "coco-128.v1i.coco" / "test"

    results: Dict[str, Dict[str, Any]] = {}

    yolo_models = [
        "yolov5nu",
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

    if args.demo:
        if args.demo in yolo_models:
            run_live_demo(args.demo,args.source)
            return
        print(f"❌ BŁĄD: Model '{args.demo}' nie jest wspierany w trybie demo.")
    if args.all:
        args.model = yolo_models + pytorch_models

    for model in args.model:
        if model.lower() in yolo_models:
            results[model] = predict_on_yolo(video_list, model.lower())
        elif model.lower() in pytorch_models:
            results[model] = predict_on_pytorch(video_list, model.lower())

        if path_coco:
            if model in yolo_models:
                accuracy_metrics = validate_yolo(model)
                results[model]["accuracy_benchmark"] = accuracy_metrics

            elif model in pytorch_models:
                accuracy_metrics = validate_pytorch(model, path_coco)
                results[model]["accuracy_benchmark"] = accuracy_metrics
    #results = predict_on_pytorch(video_list, args.model, results)
    #predicted_time = predict_on_yolo(video_list, args.model, results)

    reports_dir = pathlib.Path.cwd() / "Reports"

    hardware = get_hardware_info()
    generate_report(results, hardware, reports_dir)
    generate_plots(results, reports_dir)

    for key, value in results.items():
        for klucz, wartosc in value.items():
            print(f"{key} : {klucz} : {wartosc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description= "CLI tool to benchmark object detection models' performance."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-m",
                        "--model",
                        nargs="+",
                        type=str,
                        help="One or more model names to test (e.g., yolov8n.pt ssd_mobilenet_v2)")

    group.add_argument("--all",
                       action="store_true",
                       help="Test ALL supported models automatically")

    group.add_argument("--demo",
                       type=str,
                       metavar="MODEL",
                       help="Visual demo mode (YOLO only)")

    parser.add_argument("--source",
                       type=str,
                       default="0",
                       help="Source for demo: '0' for webcam OR filename in data/ folder")
    try:
        args = parser.parse_args()
        main(args)
    except SystemExit as e:
        print(f"ERROR: You must provide at least one model to test.")
        print("Usage: python benchmark.py --model yolov8n.pt")
        sys.exit(-1)