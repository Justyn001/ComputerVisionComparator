import pathlib
from ultralytics import YOLO, settings


def run_live_demo(model_name: str, source_input: str):

    if source_input == "0":
        print("Webcam Mode: Starting cam...")
        source = 0 
    else:
        data_dir = pathlib.Path(__file__).parent.parent / "data"
        file_path = data_dir / source_input

        if not file_path.exists():
            print(f"Error: Can not finde: '{source_input}' in folder data/")
            return

        print(f"File Mode: Playing: {file_path.name}...")
        source = str(file_path)

    print(f"--- LIVE DEMO: {model_name} ---")
    print("({Press Q to exit})")

    project_root = pathlib.Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    settings.update({'weights_dir': str(models_dir)})

    filename = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
    model_path = models_dir / filename

    try:
        if model_path.exists():
            model = YOLO(str(model_path))
        else:
            print(f"Downloading Model: {filename}...")
            model = YOLO(filename)

        model.predict(source=source, show=True, conf=0.5)

    except Exception as e:
        print(f"Demo error: {e}")