import xml.etree.ElementTree as ET
from typing import Dict
from pathlib import Path
import pathlib
import os

def generate_report(report_data: Dict[str,
                    Dict[str, Dict]]) -> None:
    print("\n[INFO] Generating XML report")

    root = ET.Element("BenchmarkReport")
    for model_name, data in report_data.items():
        model_name_elem = ET.SubElement(root, "Model", name=model_name.capitalize())
        for video_name, video_data in data.items():
            video_name_elem = ET.SubElement(model_name_elem, "Video_type", name=video_name)
            for title, value in video_data.items():
                data_type_elem = ET.SubElement(video_name_elem, "srata", name=title)
                data_type_elem.text = f"{value:.2f}."
    tree = ET.ElementTree(root)
    ET.indent(tree)

    path_to_raports = pathlib.Path.cwd() / "Reports"

    if not path_to_raports.is_dir():
        os.mkdir(path_to_raports)
    tree.write("Reports/BenchmarkReport")
    print(f"[INFO] Raport saved at {path_to_raports}/BenchmarkReport.xml")

