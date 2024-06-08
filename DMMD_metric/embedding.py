import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel

DINO_MODEL_NAME = 'facebook/dinov2-base'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dino_preprocess(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images

class DinoEmbedding:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        self.model = AutoModel.from_pretrained(DINO_MODEL_NAME).eval().to(DEVICE)
        self.input_image_size = self.processor.crop_size["height"]

    @torch.no_grad()
    def get_embedding(self, images):
        images = dino_preprocess(images, self.input_image_size)
        inputs = self.processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        image_embs = self.model(**inputs)['pooler_output'].cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs