import torchvision.models.detection as detection
from typing import List, Dict
import torch
import cv2
import pathlib
import time
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.device_selection import select_device


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def predict_on_pytorch(video_list: List[pathlib],
                       model: str) -> Dict[str, Dict[str, Dict]]:

    transformer = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((640, 640),antialias=True),
    ])

    device = select_device()

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

        with torch.no_grad():
            start_time = time.perf_counter()
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

    return video_results



def get_coco_ground_truth(json_path: pathlib.Path) -> Dict[str, Dict[str, torch.Tensor]]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    roboflow_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    name_to_pytorch_id = {}
    for name in roboflow_id_to_name.values():
        try:
            if name == "aeroplane": name = "airplane"
            if name == "motorbike": name = "motorcycle"

            pytorch_id = COCO_INSTANCE_CATEGORY_NAMES.index(name)
            name_to_pytorch_id[name] = pytorch_id
        except ValueError:
            print(f"Class '{name}' from dataset do not exist in PyTorch COCO. Skipping.")
            name_to_pytorch_id[name] = -1

    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    ground_truth = {}

    for ann in data['annotations']:
        image_id = ann['image_id']
        filename = img_id_to_filename.get(image_id)
        if not filename: continue

        json_cat_id = ann['category_id']
        category_name = roboflow_id_to_name.get(json_cat_id)

        final_cat_id = name_to_pytorch_id.get(category_name, -1)

        if final_cat_id == -1:
            continue

        if filename not in ground_truth:
            ground_truth[filename] = {"boxes": [], "labels": []}

        x, y, w, h = ann['bbox']
        box = [x, y, x + w, y + h]  

        ground_truth[filename]["boxes"].append(box)
        ground_truth[filename]["labels"].append(final_cat_id)  

    final_gt = {}
    for fname, d in ground_truth.items():
        if len(d["boxes"]) > 0:
            final_gt[fname] = {
                "boxes": torch.tensor(d["boxes"], dtype=torch.float32),
                "labels": torch.tensor(d["labels"], dtype=torch.int64)
            }
        else:
            final_gt[fname] = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            }

    return final_gt


def validate_pytorch(model_name: str, dataset_root: pathlib.Path) -> Dict[str, float]:
    print(f" Validation PyTorch Accuracy: {model_name} ---")

    device = select_device()

    json_file = list(dataset_root.rglob("*_annotations.coco.json"))
    if not json_file:
        print(f"Error: No JSON file in {dataset_root}")
        return {}

    json_path = json_file[0]
    images_dir = json_path.parent

    ground_truth_map = get_coco_ground_truth(json_path)

    try:
        model_function = getattr(detection, model_name)
        model = model_function(weights="DEFAULT")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"    ❌ Error Model: {e}")
        return {}

    metric = MeanAveragePrecision(iou_type="bbox")

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    image_filenames = list(ground_truth_map.keys())

    debug_printed = False

    with torch.no_grad():
        for fname in tqdm(image_filenames, desc=f"Accuracy ({model_name})"):
            img_path = images_dir / fname
            if not img_path.exists(): continue

            img_original = cv2.imread(str(img_path))
            if img_original is None: continue

            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

            input_tensor = transform(img_rgb).to(device)

            preds = model([input_tensor])

            preds_cpu = [{k: v.cpu() for k, v in preds[0].items()}]

            target = ground_truth_map[fname]

            metric.update(preds_cpu, [target])

    print("Calculating mAP...")
    result = metric.compute()

    final_metrics = {
        "mAP50": round(result['map_50'].item(), 4),
        "mAP50-95": round(result['map'].item(), 4),
        "Recall": round(result['mar_100'].item(), 4),
    }

    print(f"Results: {final_metrics}")
    return final_metrics