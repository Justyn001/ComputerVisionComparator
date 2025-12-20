import requests
import zipfile
import os

def download_data(path):
    with open(path/"data.zip", "wb") as f:
        file = requests.get("https://github.com/Justyn001/ComputerVisionComparator/releases/download/v.1.0-data/data.zip")
        f.write(file.content)
    with zipfile.ZipFile(path/"data.zip", "r") as zip_ref:
        zip_ref.extractall(path)
        os.remove(path/"data.zip")

    with open(path/"coco.zip", "wb") as f:
        file = requests.get("https://github.com/Justyn001/ComputerVisionComparator/releases/download/v.1.0-data/coco-128.v1i.coco.zip")
        f.write(file.content)
    with zipfile.ZipFile(path/"coco.zip", "r") as zip_ref:
        zip_ref.extractall(path)
        os.remove(path/"coco.zip")