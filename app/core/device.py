import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def log_device():
    print(f"Using device: {DEVICE}")
