import xml.etree.ElementTree as ET
from datetime import datetime
import pathlib
from typing import Dict, Any


def generate_report(results: Dict[str, Dict[str, Any]], hardware_info: Dict[str, Any], output_dir: pathlib.Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_report.xml"

    print(f"📝 Generowanie raportu XML do {output_file}...")

    root = ET.Element("ComputerVisionBenchmark")

    meta = ET.SubElement(root, "Metadata")
    ET.SubElement(meta, "Timestamp").text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hw = ET.SubElement(root, "Hardware")
    # CPU
    cpu = ET.SubElement(hw, "CPU")
    cpu.set("name", str(hardware_info.get('cpu', {}).get('name', 'Unknown')))
    ET.SubElement(cpu, "Cores").text = str(hardware_info.get('cpu', {}).get('physical_cores', 'N/A'))
    ET.SubElement(cpu, "LogicCores").text = str(hardware_info.get('cpu', {}).get('logical_cores', 'N/A'))
    ET.SubElement(cpu, "FreqMHz").text = str(hardware_info.get('cpu', {}).get('max_frequency_mhz', 'N/A'))

    # GPU
    gpu = ET.SubElement(hw, "GPU")
    gpu.set("available", str(hardware_info.get('gpu', {}).get('cuda_available', 'False')))
    ET.SubElement(gpu, "Name").text = str(hardware_info.get('gpu', {}).get('name', 'N/A'))
    ET.SubElement(gpu, "VRAM_GB").text = str(hardware_info.get('gpu', {}).get('vram_total_gb', '0'))

    # RAM
    ET.SubElement(hw, "RAM_GB").text = str(hardware_info.get('ram', {}).get('total_gb', 'N/A'))

    models_elem = ET.SubElement(root, "Models")

    best_speed_model = ("None", 0.0)
    best_accuracy_model = ("None", 0.0)

    for model_name, data in results.items():
        m_elem = ET.SubElement(models_elem, "Model", name=model_name)

        acc_data = data.get("accuracy_benchmark")
        map50 = 0.0

        acc_elem = ET.SubElement(m_elem, "Accuracy")
        if isinstance(acc_data, dict):
            map50 = acc_data.get('mAP50', 0.0)
            for k, v in acc_data.items():
                val = v.item() if hasattr(v, 'item') else v
                ET.SubElement(acc_elem, k).text = str(val)
        else:
            ET.SubElement(acc_elem, "Status").text = str(acc_data)

        vid_elem = ET.SubElement(m_elem, "VideoBenchmark")

        avg_fps_sum = 0.0
        vid_count = 0

        for key, val in data.items():
            if key == "accuracy_benchmark":
                continue  

            vid_name = key
            metrics = val

            if not isinstance(metrics, dict): continue

            v_node = ET.SubElement(vid_elem, "Video", name=vid_name)

            for k, v in metrics.items():
                val = v.item() if hasattr(v, 'item') else v
                ET.SubElement(v_node, k).text = str(val)

            avg_fps_sum += metrics.get('system_fps', 0)
            vid_count += 1

        avg_model_fps = avg_fps_sum / vid_count if vid_count > 0 else 0.0
        ET.SubElement(m_elem, "AverageSystemFPS").text = f"{avg_model_fps:.2f}"

        if avg_model_fps > best_speed_model[1]:
            best_speed_model = (model_name, avg_model_fps)

        if map50 > best_accuracy_model[1]:
            best_accuracy_model = (model_name, map50)

    summary = ET.SubElement(root, "Summary")

    win_speed = ET.SubElement(summary, "Winner_Speed")
    if best_speed_model[1] > 0:
        win_speed.set("model", best_speed_model[0])
        win_speed.text = f"{best_speed_model[1]:.2f} FPS"
    else:
        win_speed.text = "No video data"

    win_acc = ET.SubElement(summary, "Winner_Accuracy")
    if best_accuracy_model[1] > 0:
        win_acc.set("model", best_accuracy_model[0])
        win_acc.text = f"{best_accuracy_model[1]:.4f} mAP50"
    else:
        win_acc.text = "No accuracy data"

    # Zapis
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Report saved succesfully")