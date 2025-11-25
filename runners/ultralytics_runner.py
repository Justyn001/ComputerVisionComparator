import os
import pathlib
from typing import List, Dict
from ultralytics import YOLO, settings
import torch
from timeit import default_timer as timer, default_timer
import cv2
from utils.device_selection import select_device


def predict_on_yolo(video_list: List[pathlib],
                    model: str,
                    ) -> Dict[str, Dict[str, Dict]]:

    projects_root = pathlib.Path(__file__).parent.parent
    models_path = projects_root/"models"

    models_path.mkdir(exist_ok=True)
    print(models_path)
    settings.update({"weights_dir" : str(models_path)})


    if not model.endswith(".pt"):
        model += ".pt"

    model_path = models_path / model
    if model_path.exists():
        print(f"Model path exist {model_path}")
        yolo_model = YOLO(str(model_path))
    else:
        print(f"Model path do not exist {model}")
        yolo_model = YOLO(model)

    video_dict: Dict[str, Dict] = {}
    for video in video_list:
        cap = cv2.VideoCapture(str(video))

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_length = total_frames / video_fps
        cap.release()
        start_time = default_timer()
        yolo_model.predict(source=video,
                           show=False,
                           show_labels=False,
                           device=select_device())
        predict_time = default_timer() - start_time
        real_fps = total_frames / predict_time

        video_dict[video.stem] = {
            "video_length": round(video_length, 2),
            "processing_time" : round(predict_time, 2),
            "total_frames" : total_frames,
            "video_fps": round(video_fps, 2),
            "system_fps" : round(real_fps, 2)}
    #model_dict[model] = video_dict
    return video_dict


def validate_yolo(model_name: str) -> Dict[str, float]:
    """
    Uruchamia walidację modelu YOLO na podanym datasecie.
    Zwraca słownik z metrykami mAP.
    """
    print(f"--- 🎯 Walidacja dokładności (YOLO): {model_name} ---")

    project_root = pathlib.Path(__file__).parent.parent
    models_dir = project_root / "models"


    # Upewnij się, że nazwa ma .pt
    filename = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
    model_path = models_dir / filename

    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        model = YOLO(filename)

    try:
        # Uruchom walidację
        # split='test' używa zbioru testowego z yaml.
        # Jeśli go nie ma, YOLO automatycznie użyje 'val'.
        # project/name ustawiamy na 'runs/val', żeby nie śmiecić
        metrics = model.val(
            data='coco128.yaml',  # <--- MAGICZNA ZMIANA
            split='val',  # coco128 ma tylko 'train' i 'val', nie ma 'test'
            device=select_device(),
            verbose=False,
            plots=False
        )

        # Wyciągnij kluczowe metryki
        # map50 = mAP przy progu IoU 0.5
        # map50-95 = mAP uśrednione dla progów 0.5-0.95 (najważniejsza metryka COCO)
        results = {
            "mAP50": round(metrics.box.map50, 4),
            "mAP50-95": round(metrics.box.map, 4),
            "Precision": round(metrics.box.mp, 4),
            "Recall": round(metrics.box.mr, 4)
        }

        print(f"Wyniki: mAP50={results['mAP50']}, mAP50-95={results['mAP50-95']}")
        return results

    except Exception as e:
        print(f"    ❌ BŁĄD podczas walidacji: {e}")
        return {}