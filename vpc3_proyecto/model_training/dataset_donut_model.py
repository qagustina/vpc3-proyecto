import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

from vpc3_proyecto.data.utils import load_annotations


class DonutTextDatasetFromCocoTextV2Raw(Dataset):
    def __init__(self, image_dir, ann_zip_file, processor, max_length=512):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        # exists = os.path.exists(ann_zip_file)
        # if not exists:
        #     raise ValueError(f"El archivo de anotaciones {ann_zip_file} no existe.")
        coco_data = load_annotations(ann_zip_file)

        self.anns = coco_data["anns"]
        self.imgs = coco_data["imgs"]

        # Indexar anotaciones por imagen_id
        self.ann_by_image = {}
        for ann in self.anns.values():
            img_id = ann["image_id"]
            if img_id not in self.ann_by_image:
                self.ann_by_image[img_id] = []
            self.ann_by_image[img_id].append(ann)

        # Filtrar imágenes existentes en image_dir
        image_files = set(os.listdir(image_dir))
        self.valid_img_ids = [
            int(img_id)
            for img_id, img_data in self.imgs.items()
            if img_data["file_name"] in image_files
        ]

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        img_id = self.valid_img_ids[idx]
        img_info = self.imgs[str(img_id)]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        # Obtener texto legible
        anns = self.ann_by_image.get(img_id, [])
        texts = [
            ann["utf8_string"] for ann in anns
            if ann.get("legibility", "") == "legible" and ann.get("utf8_string", "").strip() != ""
        ]
        text = " ".join(texts).strip()

        # 1. Imagen
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"].squeeze()

        # 2. Texto (tokenizado manualmente)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.squeeze()

        # 3. Reemplazo de tokens pad
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "img_path": img_path
        }