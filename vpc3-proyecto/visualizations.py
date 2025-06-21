import os
import json
import zipfile
from collections import defaultdict
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from eda import load_anns, group_anns, get_imgs, get_id


def get_imgs(id_to_filename: dict, img_folder: str, cant: int = 10):
    img_list = []
    for img_id, filename in id_to_filename.items():
        ruta = os.path.join(img_folder, filename.split('/')[-1])
        if os.path.exists(ruta):
            img_list.append((img_id, ruta))
        if len(img_list) == cant:
            break
    return img_list


def plot_examples(img_list: list, anotaciones_por_imagen: dict, nombre_subset: str, output_folder: str):
    fig, axs = plt.subplots(2, 5, figsize=(20, 12))
    axs = axs.flatten()

    for i, (img_id, ruta_img) in enumerate(img_list):
        img = Image.open(ruta_img)
        axs[i].imshow(img)
        axs[i].axis('off')

        textos = [
            ann['utf8_string']
            for ann in anotaciones_por_imagen[img_id]
            if ann['legibility'] == 'legible' and 'utf8_string' in ann
        ]
        axs[i].set_title('\n'.join(textos[:2]), fontsize=15)

    
    fig.suptitle(f"Images and annotations from the set {nombre_subset} examples.", fontsize=12)
    plt.tight_layout() 

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'visualization_{nombre_subset}.png')
    plt.savefig(output_path)

def plot_class_examples(anns_imgs, id_to_filename, img_folder, img_class, output_path):
    img_ok = False
    for img_id, anns in anns_imgs.items():
        if any(ann['class'] == img_class for ann in anns):
            img_ok = True
            filename = id_to_filename.get(img_id)
            if not filename:
                continue

            ruta = os.path.join(img_folder, filename.split('/')[-1])
            if not os.path.exists(ruta):
                continue

            image = Image.open(ruta).convert('RGB')
            draw = ImageDraw.Draw(image)

            # boxes
            for ann in anns:
                if ann['class'] == img_class: 
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x + w, y + h], outline='red', width=2)

            output_file = os.path.join(output_path, f'example_{img_class}.jpg')
            image.save(output_file)
            break  
    if not img_ok:
        print(f"No se encontró ninguna imagen con anotaciones {img_class}.")

def main(subset_folder: str):
    zip_path = '../data/raw/cocotext.v2.zip'
    json_file = 'cocotext.v2.json'
    img_folder = os.path.join('../data/raw/subset', subset_folder)
    output_path = '../reports/figures'

    coco_text = load_anns(zip_path, json_file)
    anns_imgs = group_anns(coco_text)
    id_to_filename = get_id(coco_text, subset='train')

    img_list = get_imgs(id_to_filename, img_folder, cant=10)
    plot_examples(img_list, anns_imgs, nombre_subset=subset_folder, output_folder=output_path)
    
    # classes 
    clases_objetivo = ['handwritten', 'machine printed']
    for clase in clases_objetivo:
        plot_class_examples(anns_imgs, id_to_filename, img_folder, clase, output_path)


if __name__ == "__main__":
    main("train2014")
    main("val2014")