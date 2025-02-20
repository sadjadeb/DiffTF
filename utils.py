import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def preprocess_tensor(tensor, target_channels=32, feature_length=600):
    # Determine how many samples should remain (nearest multiple of target_channels)
    n_samples = tensor.shape[0]
    truncate_size = (n_samples // target_channels) * target_channels  # Nearest multiple of target_channels

    # Truncate the tensor to make the number of samples divisible by target_channels
    tensor_truncated = tensor[:truncate_size]

    # Reshape the tensor to (x, target_channels, feature_length)
    reshaped_tensor = tensor_truncated.view(-1, target_channels, feature_length)
    
    return reshaped_tensor