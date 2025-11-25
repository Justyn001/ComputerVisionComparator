import torch

def select_device():
    """
    Inteligentnie wybiera najlepsze dostępne urządzenie obliczeniowe:
    - CUDA (Nvidia)
    - MPS (Apple Silicon M1/M2/M3)
    - CPU (Fallback)
    """
    if torch.cuda.is_available():
        print("🚀 Wykryto GPU NVIDIA (CUDA).")
        return torch.device("cuda")

    # Sprawdzenie dla Maców (Apple Silicon)
    elif torch.backends.mps.is_available():
        print("🍎 Wykryto Apple Silicon (MPS Metal).")
        return torch.device("mps")

    else:
        print("🐢 Nie wykryto akceleratora. Używam CPU.")
        return torch.device("cpu")