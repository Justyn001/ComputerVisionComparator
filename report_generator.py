import xml.etree.ElementTree as ET
from typing import Dict
from pathlib import Path

def generate_report(report_data: Dict[str,
                    Dict[str, float]]) -> None:
    print("\n[INFO] Generating XML report")

    root = ET.Element("BenchmarkReport")
    for model_name, data in report_data.items():
        model_elem = ET.SubElement(root, "Model", name=model_name)
        for video_name, time in data.items():
            video_elem = ET.SubElement(model_elem, "Video", name=video_name)
            time_elem = ET.SubElement(video_elem, "ProcessingTime")
            time_elem.text = f"{time:.2f} seconds"
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write("BenchmarkReport")
    print(f"[INFO] Raport saved at {Path.cwd()}/BenchmarkReport.xml")

