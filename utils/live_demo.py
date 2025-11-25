import pathlib
from ultralytics import YOLO, settings


def run_live_demo(model_name: str, source_input: str):
    """
    Uruchamia wizualne demo dla modelu YOLO.
    source_input: '0' (kamera) lub nazwa pliku w folderze data (np. 'traffic.mp4')
    """

    if source_input == "0":
        print("📷 Tryb KAMERY: Uruchamiam webcam...")
        source = 0  # YOLO woli int 0 dla kamery
    else:
        # Szukamy pliku w folderze data/
        data_dir = pathlib.Path(__file__).parent.parent / "data"
        file_path = data_dir / source_input

        if not file_path.exists():
            print(f"❌ BŁĄD: Nie znaleziono pliku '{source_input}' w folderze data/")
            return

        print(f"Tryb PLIKU: Odtwarzam {file_path.name}...")
        source = str(file_path)

    print(f"--- LIVE DEMO: {model_name} ---")
    print("(Naciśnij 'Q' na oknie wideo, aby zakończyć)")

    # 3. Konfiguracja modeli (żeby brał z models/)
    project_root = pathlib.Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    settings.update({'weights_dir': str(models_dir)})

    # 4. Ładowanie modelu
    filename = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
    model_path = models_dir / filename

    try:
        if model_path.exists():
            model = YOLO(str(model_path))
        else:
            print(f"⬇️ Pobieranie modelu {filename}...")
            model = YOLO(filename)

        # 5. START
        # show=True -> Okno, loop=True -> Zapętlanie, conf=0.5 -> Tylko pewne detekcje
        model.predict(source=source, show=True, conf=0.5)

    except Exception as e:
        print(f"❌ Błąd krytyczny demo: {e}")