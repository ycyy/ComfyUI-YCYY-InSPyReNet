from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm
import os
import folder_paths
from folder_paths import models_dir

# Define the directory for Inspyrenet models
models_dir = os.path.join(folder_paths.models_dir, "transparent-background")
ckpt_name = os.path.join(models_dir, "ckpt_base.pth")

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class InspyrenetRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "YCYY/image"

    def remove_background(self, image, torchscript_jit):
        if (torchscript_jit == "default"):
            remover = Remover(ckpt=ckpt_name)
        else:
            remover = Remover(jit=True,ckpt=ckpt_name)
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type='rgba')
            out =  pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)
        
class InspyrenetRembgAdvanced:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (
                    [ 
                        'ckpt_base.pth',
                        'ckpt_fast.pth',
                        'ckpt_base_nightly.pth'
                    ],
                    {
                    "default": 'ckpt_base.pth'
                    }
                ),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background_advanced"
    CATEGORY = "YCYY/image"

    def remove_background_advanced(self, image,model_name, torchscript_jit, threshold):
        ckpt_name = os.path.join(models_dir, model_name)
        if os.path.exists(ckpt_name):
            if (torchscript_jit == "default"):
                remover = Remover(ckpt=ckpt_name)
            else:
                remover = Remover(jit=True,ckpt=ckpt_name)
            img_list = []
            for img in tqdm(image, "Inspyrenet Rembg"):
                mid = remover.process(tensor2pil(img), type='rgba', threshold=threshold)
                out =  pil2tensor(mid)
                img_list.append(out)
            img_stack = torch.cat(img_list, dim=0)
            mask = img_stack[:, :, :, 3]
            return (img_stack, mask)
        else:
            print("ckpt not found")            
        