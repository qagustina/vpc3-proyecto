import os
import json
import zipfile
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt


def load_anns(zip_path: str, json_file: str):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(json_file) as file:
            return json.load(file)


def get_id(coco_dict: dict, subset: str = 'train'):
    return {
        int(k): v['file_name']
        for k, v in coco_dict['imgs'].items()
        if v['set'] == subset
    }


def group_anns(coco_dict: dict):
    anns = defaultdict(list)
    for ann in coco_dict['anns'].values():
        anns[ann['image_id']].append(ann)
    return anns


def get_imgs(id_to_filename: dict, img_folder: str, cant: int = 10):
    img_list = []
    for img_id, filename in id_to_filename.items():
        ruta = os.path.join(img_folder, filename.split('/')[-1])
        if os.path.exists(ruta):
            img_list.append((img_id, ruta))
        if len(img_list) == cant:
            break
    return img_list


def plot(img_list: list, anotaciones_por_imagen: dict, nombre_subset: str):
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

    # make dir 
    output_folder = os.path.join('../reports/figures')
    os.makedirs(output_folder, exist_ok=True)

    # output
    output_path = os.path.join(output_folder, f'visualization_{nombre_subset}.png')
    plt.savefig(output_path)


def main(subset_folder: str):
    zip_path = '../data/raw/cocotext.v2.zip'
    json_file = 'cocotext.v2.json'
    img_folder = os.path.join('../data/raw/subset', subset_folder)

    coco_text = load_anns(zip_path, json_file)
    anns_imgs = group_anns(coco_text)
    id_to_filename = get_id(coco_text, subset='train')

    img_list = get_imgs(id_to_filename, img_folder, cant=10)
    plot(img_list, anns_imgs, nombre_subset=subset_folder)


if __name__ == "__main__":
    main("train2014")
    main("val2014")