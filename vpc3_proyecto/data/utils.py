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