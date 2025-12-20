import torch

def select_device():
    if torch.cuda.is_available():
        print("Detected NVIDIA GPU (CUDA).")
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        print("Detected Apple Silicon (MPS Metal).")
        return torch.device("mps")

    else:
        print("No accelerator detected. Using CPU")
        return torch.device("cpu")