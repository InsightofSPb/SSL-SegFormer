import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
from semseg.metrics import Metrics
from semseg.augmentations import get_val_augmentation
from rich.console import Console
import numpy as np

console = Console()

class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = torch.tensor([
            [0, 0, 0],        # boundary (black)
            [255, 255, 255],  # impervious surfaces (white)
            [0, 0, 255],      # building (blue)
            [0, 255, 255],    # low vegetation (light blue)
            [0, 255, 0],      # tree (green)
            [255, 255, 0],    # car (yellow)
            [255, 0, 0],      # clutter/background (red)
        ]).to(self.device)
        self.labels = ['boundary', 'impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))

        weights_path = cfg['TEST']['MODEL_PATH']
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = get_val_augmentation(self.size)

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # apply the same transformations as in validation
        image, _ = self.tf_pipeline(image, image)  # Pass the image twice to satisfy the mask requirement
        image = image.to(self.device)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = draw_text(seg_image, seg_map, self.labels)
        return image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = seg_map.squeeze(0)  # Remove batch dimension
        return seg_map

    def convert_annotation(self, annotation: Tensor) -> Tensor:
        # Convert 3-channel annotation to 7-channel annotation
        H, W = annotation.shape[1:]
        new_annotation = torch.zeros((7, H, W), dtype=torch.uint8, device=self.device)
        for i, color in enumerate(self.palette):
            mask = (annotation == color.view(3, 1, 1)).all(dim=0)
            new_annotation[i][mask] = 1
        return new_annotation

    def compute_iou(self, pred: Tensor, target: Tensor, num_classes: int) -> np.ndarray:
        ious = []
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / max(union, 1))
        return np.array(ious)

    def evaluate(self, img_dir: Path, anno_dir: Path) -> None:
        ious = []
        num_classes = len(self.palette)
        metrics = Metrics(num_classes, ignore_label=-1, device=self.device)
        
        for img_file in img_dir.glob('*.png'):
            img_id = img_file.stem.split('_')[-1]
            anno_file = anno_dir / f'gt_{img_id}.png'
            if not anno_file.exists():
                console.print(f"Annotation file {anno_file} not found, skipping.")
                continue
            pred = self.predict(str(img_file), overlay=False)
            target = io.read_image(str(anno_file)).squeeze(0).to(self.device)
            
            # Resize target to match prediction size
            target = F.interpolate(target.unsqueeze(0).float(), size=pred.shape[1:], mode='nearest').squeeze(0).to(int)
            
            # Convert target to 7-channel annotation
            target = self.convert_annotation(target)
            
            # Convert pred and target to single-channel tensors with class indices
            pred = pred.argmax(dim=0)
            target = target.argmax(dim=0)
            
            # Debugging statements
            console.print(f"Prediction shape: {pred.shape}")
            console.print(f"Target shape: {target.shape}")
            console.print(f"Unique values in prediction: {torch.unique(pred)}")
            console.print(f"Unique values in target: {torch.unique(target)}")
            
            # Flatten pred and target to 1D
            pred = pred.view(-1)
            target = target.view(-1)
            
            metrics.update(pred, target)
        
        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()
        
        console.print(f"Mean IoU: {miou}")
        for cls, iou in enumerate(ious):
            console.print(f"Class {self.labels[cls]} IoU: {iou}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)
    
    semseg = SemSeg(cfg)

    with console.status("[bright_green]Processing..."):
        if test_file.is_file():
            console.rule(f'[green]{test_file}')
            segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
            segmap.save(save_dir / f"{str(test_file.stem)}.png")
        else:
            img_dir = test_file / 'image'
            anno_dir = test_file / 'anno'
            semseg.evaluate(img_dir, anno_dir)

    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")




