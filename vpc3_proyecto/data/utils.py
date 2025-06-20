import os
from dotenv import load_dotenv

import requests
from tqdm import tqdm
def get_data(data_folder):
    os.makedirs(data_folder, exist_ok=True)

    urls = {
        "cocotext.v2.zip": "https://bgshih.github.io/cocotext/cocotext.v2.zip",
        "train2014.zip": "http://images.cocodataset.org/zips/train2014.zip"
    }

    for name_file, url in urls.items():
        ruta_destino = os.path.join(data_folder, name_file)
        if os.path.exists(ruta_destino):
            print(f"{name_file} ya existe. Omitiendo descarga.")
            continue

        print(f"Descargando {name_file}...")
        respuesta = requests.get(url, stream=True)
        total = int(respuesta.headers.get('content-length', 0))

        with open(ruta_destino, 'wb') as archivo, tqdm(
            desc=name_file,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as barra:
            for datos in respuesta.iter_content(chunk_size=1024):
                archivo.write(datos)
                barra.update(len(datos))

    print("Descarga finalizada.")


def count_images(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        return len([f for f in zipf.namelist() if f.lower().endswith('.jpg')])

import zipfile
import json
def extract_anns(zip_path, json_filename='cocotext.v2.json'):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(json_filename) as json_file:
            return json.load(json_file)


def contar_labeled(coco_text_data):
    etiquetadas = set()
    for ann_id, ann in coco_text_data['anns'].items():
        img_id = ann['image_id']
        if ann['legibility'] in ['legible', 'illegible']:
            etiquetadas.add(img_id)
    return len(etiquetadas)

def image_id_to_train_filename(image_id):
    return f'COCO_train2014_{image_id:012d}.jpg'


def load_annotations(zip_path, json_filename='cocotext.v2.json'):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(json_filename) as file:
            return json.load(file)


def get_annotated_image_ids(coco_data):
    annotated_ids = set()
    for ann in coco_data['anns'].values():
        annotated_ids.add(ann['image_id'])
    return annotated_ids

from collections import defaultdict

def get_annotations_by_image_id(coco_text_data):
    """Groups annotations by image_id for easy lookup."""
    img_id_to_anns = defaultdict(list)
    for ann_id, ann in coco_text_data['anns'].items():
        # We only want to train on legible text
        if ann['legibility'] == 'legible':
            img_id_to_anns[ann['image_id']].append(ann)
    return img_id_to_anns

def extract_annotated_images(zip_path, output_dir, target_filenames):
    # Clean up target directory if it exists
    if os.path.exists(output_dir):
        print(f"🧹 Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zip_files = zipf.namelist()
        matched_files = [f for f in zip_files if os.path.basename(f) in target_filenames]

        if not matched_files:
            print(f'No matching files found in {zip_path}')
            return

        for file in tqdm(matched_files, desc=f'Extracting from {os.path.basename(zip_path)}'):
            # Extract only the filename without directory
            filename = os.path.basename(file)
            target_path = os.path.join(output_dir, filename)

            # Extract file directly to output_dir
            with zipf.open(file) as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)

import shutil
import random

def split_subset(source_dir, target_dir, val_count=7829, seed=42, move=False):
    # Clean up target directory if it exists
    if os.path.exists(target_dir):
        print(f"🧹 Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]

    if val_count > len(all_images):
        raise ValueError(f"Se solicitaron {val_count} imágenes, pero solo hay {len(all_images)} disponibles en '{source_dir}'.")

    random.seed(seed)
    selected_images = random.sample(all_images, val_count)

    for filename in selected_images:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        if move:
            shutil.move(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    print(f"{val_count} imágenes {'movidas' if move else 'copiadas'} a '{target_dir}'.")

def train_filename_to_image_id(filename):
    """Converts 'COCO_train2014_000000123456.jpg' to 123456."""
    return int(filename.split('_')[-1].split('.')[0])

import cv2
from PIL import Image
import csv

def create_cropped_dataset(image_dir, output_dir, img_id_to_anns):
    """
    Crops text regions from images and saves them for training.
    This version uses the `csv` module to correctly handle labels
    that contain commas.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing images in '{image_dir}' and saving crops to '{output_dir}'...")

    labels_file_path = os.path.join(output_dir, 'labels.csv')

    with open(labels_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        image_files = os.listdir(image_dir)

        for filename in tqdm(image_files, desc="Cropping text regions"):
            image_id = train_filename_to_image_id(filename)

            if image_id not in img_id_to_anns:
                continue

            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            annotations = img_id_to_anns[image_id]

            for i, ann in enumerate(annotations):
                bbox = [int(p) for p in ann['bbox']]
                text_label = ann['utf8_string']

                if not text_label or len(text_label) < 2:
                    continue

                x, y, w, h = bbox

                if w <= 0 or h <= 0 or x < 0 or y < 0 or (x+w) > image.shape[1] or (y+h) > image.shape[0]:
                    continue

                cropped_image = image[y:y+h, x:x+w]
                crop_filename = f"{image_id}_{i}.png"


                writer.writerow([crop_filename, text_label])

                cv2.imwrite(os.path.join(output_dir, crop_filename), cropped_image)


import os
import json
import pandas as pd


def create_labels_csv_from_annotations(root_dir, annotation_file, output_csv_name='labels.csv'):
    """
    Create a labels.csv file from COCO-style annotations JSON file.

    Args:
        root_dir (str): Directory containing the images
        annotation_file (str): Path to the COCO-style annotations JSON file
        output_csv_name (str): Name of the output CSV file (default: 'labels.csv')

    Returns:
        None (writes a CSV file to root_dir)
    """
    print(f"🔍 Loading annotations from {annotation_file} ...")
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    # Build image_id → best ann mapping (largest legible annotation per image)
    img_id_to_ann = {}
    for ann_id, ann in annotations["anns"].items():
        img_id = ann["image_id"]
        legible = ann.get("legibility", "legible") == "legible"
        bbox = ann["bbox"]
        area = bbox[2] * bbox[3]

        if not legible or area == 0:
            continue  # skip non-legible or empty boxes

        if img_id not in img_id_to_ann:
            img_id_to_ann[img_id] = ann
        else:
            prev_ann = img_id_to_ann[img_id]
            prev_area = prev_ann["bbox"][2] * prev_ann["bbox"][3]
            if area > prev_area:
                img_id_to_ann[img_id] = ann

    print(f"✅ Found {len(img_id_to_ann)} images with legible annotations.")

    # Map image_id → file_name
    img_id_to_filename = {img["id"]: img["file_name"] for img in annotations["imgs"].values()}

    # Build dataframe: only keep files that exist in root_dir
    rows = []
    for img_id, ann in img_id_to_ann.items():
        fname = img_id_to_filename.get(img_id)
        text = ann.get("utf8_string", "").strip()
        full_path = os.path.join(root_dir, fname)
        if fname and text and os.path.exists(full_path):
            rows.append({"file_name": fname, "text": text})

    df = pd.DataFrame(rows)
    original_count = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    output_path = os.path.join(root_dir, output_csv_name)
    df.to_csv(output_path, index=False, header=False)

    print(f"📄 Created {output_path} with {len(df)} entries (filtered from {original_count}).")
    print(f"Columns: file_name, text")