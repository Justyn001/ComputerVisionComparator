import pathlib
from typing import List, Dict
from ultralytics import YOLO
import torch
from timeit import default_timer as timer, default_timer


def predict_on_yolo(video_list: List[pathlib], model_list: List[str]):

    model_dict: Dict[str, Dict[str, float]] = {}

    for model in model_list:
        yolo_model = YOLO(model)

        video_dict: Dict[str, float] = {}
        for video in video_list:

            start_time = default_timer()
            yolo_model.predict(source=video,
                               show=False,
                               show_labels=False,
                               device="cuda" if torch.cuda.is_available() else "cpu")
            predict_time = default_timer() - start_time

            video_dict[video.stem] = round(predict_time, 2)
        model_dict[model] = video_dict
    return model_dict