import torch
import numpy as np
from gymnasium import ObservationWrapper
from torchvision import transforms
from PIL import Image
from dinov2.models.vision_transformer import vit_base
from dinov2.models.utils import build_model_and_transform

# Load DINOv2 model and transform
dinov2, dinov2_transform = build_model_and_transform('dinov2_vitb14')
dinov2.eval().cuda()  # or .cpu()

class DinoWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dino_model = dinov2
        self.transform = dinov2_transform

    def observation(self, observation):
        # Convert to PIL and preprocess
        image = Image.fromarray(observation)
        image = self.transform(image).unsqueeze(0).cuda()  # [1, 3, H, W]

        # Extract features
        with torch.no_grad():
            features = self.dino_model(image)

        return features.cpu().numpy().flatten()  # return as flat array
