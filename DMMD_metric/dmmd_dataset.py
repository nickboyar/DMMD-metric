import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class DMMDDataset(Dataset):
    def __init__(self, path, reshape_to):
        self.path = path
        self.reshape_to = reshape_to
        self.img_path_list = self._get_image_list()

    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["np", "png", "jpg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob.glob(f"{self.path}/*{ext}"))
            image_list.extend(glob.glob(f"{self.path}/*.{ext.upper()}"))
        image_list.sort()
        return image_list

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        return im.resize((size, size), resample=Image.BICUBIC)

    def read_image_to_np(self, path, size):
        im = Image.open(path)
        im = self._center_crop_and_resize(im, size)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        x = self.read_image_to_np(img_path, self.reshape_to)
        return x