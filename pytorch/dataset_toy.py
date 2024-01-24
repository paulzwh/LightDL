import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._length = 4  # toy
        
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        image = torch.rand(1, 28, 28)
        label = (image.sum() / (28 * 28) > 0.5).long()

        return {"image": image, "label": label}

