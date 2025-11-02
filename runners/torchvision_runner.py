import torchvision.models.detection as detection
from typing import List, Dict
import torch
import cv2
import pathlib
import time
import torchvision.transforms.v2 as transforms
from tqdm import tqdm

torchvision_models = ["fasterrcnn_resnet50_fpn_v2"]

def predict_on_pytorch(model_list: List[str], video_list: List[pathlib]) -> Dict[str, Dict[str, Dict]]:

    transformer = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((640, 640),antialias=True),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results: Dict[str, Dict[str, Dict]] = {}

    for model in model_list:
        model_name = getattr(detection, model)
        model_obj = model_name(weights="DEFAULT")
        model_obj.to(device)
        model_obj.eval()

        video_results: Dict[str, Dict] = {}
        for video in tqdm(video_list):
            cap = cv2.VideoCapture(str(video))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_length = total_frames / video_fps

            frame_count: int  = 0

            start_time = time.perf_counter()
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        break
                    converted_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    transformed_image = transformer(converted_to_rgb)
                    image_batch = [transformed_image.to(device)]

                    model_obj(image_batch)

                    frame_count += 1
            processing_time = time.perf_counter() - start_time
            cap.release()
            real_fps = frame_count/processing_time

            video_results[video.stem]= {
                "video_length" : round(video_length, 2),
                "processing_time" : round(processing_time, 2),
                "total_frames" : frame_count,
                "video_fps" : round(video_fps, 2),
                "system_fps" : round(real_fps, 2)
            }
        results[model] = video_results
    return results