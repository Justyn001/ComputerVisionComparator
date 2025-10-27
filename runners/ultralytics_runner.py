import os
import pathlib
from typing import List, Dict
from ultralytics import YOLO, settings
import torch
from timeit import default_timer as timer, default_timer
import cv2


def predict_on_yolo(video_list: List[pathlib], model_list: List[str]):

    projects_root = pathlib.Path(__file__).parent.parent
    models_path = projects_root/"models"

    models_path.mkdir(exist_ok=True)
    print(models_path)
    settings.update({"weights_dir" : str(models_path)})

    model_dict: Dict[str, Dict[str, Dict]] = {}

    for model in model_list:

        if not model.endswith(".pt"):
            model += ".pt"

        model_path = models_path / model

        yolo_model = YOLO(str(model_path))

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
                               device="cuda" if torch.cuda.is_available() else "cpu")
            predict_time = default_timer() - start_time
            real_fps = total_frames / predict_time

            video_dict[video.stem] = {
                "video_length": round(video_length, 2),
                "processing_time" : round(predict_time, 2),
                "total_frames" : total_frames,
                "video_fps": round(video_fps, 2),
                "system_fps" : round(real_fps, 2)}
        model_dict[model] = video_dict
    return model_dict