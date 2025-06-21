import json
from torchvision.transforms import Resize
from functools import lru_cache
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class OCRDataset(Dataset):
    def get_df(self):
        return self.df
    def __init__(self, root_dir, processor, max_length=16, annotation_file=None):
        self.root_dir = root_dir
        self.processor = processor
        self.max_length = max_length  # Adjust based on your text length

        # Load and clean the dataset
        if not os.path.exists(os.path.join(root_dir, 'labels.csv')):
            # initialize df with empty data with coluymns file_name and text
            self.df = pd.DataFrame(columns=['file_name', 'text'])
        else:
            self.df = pd.read_csv(
                os.path.join(root_dir, 'labels.csv'),
                header=None,
                names=['file_name', 'text']
            )
        self.df.dropna(inplace=True)  # Remove NaN
        self.df = self.df[self.df['text'].str.strip().str.len() > 0]  # Remove empty strings
        self.df.reset_index(drop=True, inplace=True)
        # If annotation_file is provided → process it
        if annotation_file:
            print(f"🔍 Loading annotations from {annotation_file} ...")
            with open(annotation_file, "r") as f:
                annotations = json.load(f)

            # Build image_id → best ann mapping
            self.img_id_to_ann = {}
            for ann_id, ann in annotations["anns"].items():
                img_id = ann["image_id"]
                legible = ann.get("legibility", "legible") == "legible"
                bbox = ann["bbox"]
                area = bbox[2] * bbox[3]

                if not legible or area == 0:
                    continue  # skip non-legible or empty boxes

                if img_id not in self.img_id_to_ann:
                    self.img_id_to_ann[img_id] = ann
                else:
                    prev_ann = self.img_id_to_ann[img_id]
                    prev_area = prev_ann["bbox"][2] * prev_ann["bbox"][3]
                    if area > prev_area:
                        self.img_id_to_ann[img_id] = ann

            print(f"✅ Found {len(self.img_id_to_ann)} images with legible annotations.")
            # Map image_id → file_name using your function
            # Filter df to keep only file_names that match the best-annotated image_ids
            # Assuming COCO file_name matches
            img_id_to_filename = {img["id"]: img["file_name"] for img in annotations["imgs"].values()}
            # Build dataframe: only keep files that exist in root_dir!
            rows = []
            for img_id, ann in self.img_id_to_ann.items():
                fname = img_id_to_filename.get(img_id)
                text = ann.get("utf8_string", "").strip()
                full_path = os.path.join(root_dir, fname)
                if fname and text and os.path.exists(full_path):
                    rows.append({"file_name": fname, "text": text})

            self.df = pd.DataFrame(rows)
            original_count = len(self.df)
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            print(f"📄 Filtered dataset from {original_count} to {len(self.df)} images.")

        else:
            self.img_id_to_ann = None  # No annotations provided

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=200)  # Cache up to 1000 images in RAM
    def __getitem__(self, idx):
        # Load image and text
        file_name = self.df.iloc[idx]['file_name']
        text = str(self.df.iloc[idx]['text'])
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        # Process image and text
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)  # Shape: [1, C, H, W] -> [C, H, W]

        # Tokenize text (Donut uses a BART tokenizer)
        labels = self.processor.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()  # Shape: [1, max_length] -> [max_length]

        # Replace padding tokens with -100 (ignored by loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "img_path": os.path.join(self.root_dir, file_name)
        }