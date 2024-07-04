import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple

class Potsdam(Dataset):
    CLASSES = ['boundary', 'impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']

    PALETTE = torch.tensor([
        [0, 0, 0],        # boundary (black)
        [255, 255, 255],  # impervious surfaces (white)
        [0, 0, 255],      # building (blue)
        [0, 255, 255],    # low vegetation (light blue)
        [0, 255, 0],      # tree (green)
        [255, 255, 0],    # car (yellow)
        [255, 0, 0],      # clutter/background (red)
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        img_path = Path(root) / f'{split}_set' / 'image'
        self.files = list(img_path.glob('*.png'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = img_path.replace('image', 'anno').replace('rgb_', 'gt_')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, self.encode_label(label)

    def encode_label(self, label: Tensor) -> Tensor:
        label = label.permute(1, 2, 0).numpy()
        encoded = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        
        for i, color in enumerate(self.PALETTE):
            mask = np.all(label == color.numpy(), axis=-1)
            encoded[mask] = i
        
        return torch.from_numpy(encoded).long()

if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(Potsdam, '/home/s/test/semantic_segmentation/data/potsdam')