import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel

_CUDA_AVAILABLE = torch.cuda.is_available()
_DINO_MODEL_NAME = 'facebook/dinov2-base'

def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images

class DinoEmbeddingModel:
    
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained(_DINO_MODEL_NAME)
        self._model = AutoModel.from_pretrained(_DINO_MODEL_NAME).eval()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images):
        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        image_embs = self._model(**inputs)['pooler_output'].cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs
