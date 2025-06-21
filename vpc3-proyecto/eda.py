from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os
import zipfile
import json
from collections import defaultdict, Counter
from PIL import Image
import matplotlib.pyplot as plt



def load_anns(zip_path: str, json_file: str):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(json_file) as file:
            return json.load(file)


def group_anns(coco_dict: dict):
    anns = defaultdict(list)
    for ann in coco_dict['anns'].values():
        anns[ann['image_id']].append(ann)
    return anns


def get_imgs(id_to_filename: dict, img_folder: str):
    img_list = []
    for img_id, filename in id_to_filename.items():
        ruta = os.path.join(img_folder, filename.split('/')[-1])
        if os.path.exists(ruta):
            img_list.append((img_id, ruta))
    return img_list


def get_id(coco_dict: dict, subset: str = 'train'):
    return {
        int(k): v['file_name']
        for k, v in coco_dict['imgs'].items()
        if v['set'] == subset
    }


def eda_cocotext_pdf(
    anotaciones_por_imagen,
    imagenes_train,
    imagenes_val,
    salida_pdf='resumen_cocotext.pdf',
    num_transcripciones=5
):
    doc = SimpleDocTemplate(salida_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    legibles = sum(1 for anns in anotaciones_por_imagen.values() for ann in anns if ann['legibility'] == 'legible')
    ilegibles = sum(1 for anns in anotaciones_por_imagen.values() for ann in anns if ann['legibility'] == 'illegible')
    idiomas = Counter(ann['language'] for anns in anotaciones_por_imagen.values() for ann in anns)

    story.append(Paragraph("<b>Análisis Exploratorio Inicial del Dataset COCO-Text</b>", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Dataset original</b>", styles['Heading2']))
    story.append(Paragraph("- Imágenes en train2014.zip: 82783", styles['Normal']))
    story.append(Paragraph(f"- Imágenes etiquetadas (con texto): {len(anotaciones_por_imagen)}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Subconjuntos tomados</b>", styles['Heading2']))
    story.append(Paragraph(f"- Imágenes en subset train: 15656", styles['Normal']))
    story.append(Paragraph(f"- Imágenes en subset val: 7829", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Textos legibles:</b> {legibles}", styles['Normal']))
    story.append(Paragraph(f"<b>Textos ilegibles:</b> {ilegibles}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Distribución de idiomas</b>", styles['Heading2']))
    total_idiomas = sum(idiomas.values())
    for idioma, count in idiomas.items():
        porcentaje = (count / total_idiomas) * 100
        story.append(Paragraph(f"- {idioma}: {count} ({porcentaje:.2f}%)", styles['Normal']))
    story.append(Spacer(1, 12))

    # count img by class
    clases_por_imagen = defaultdict(set)
    for img_id, anns in anotaciones_por_imagen.items():
        for ann in anns:
            clases_por_imagen[img_id].add(ann['class'])

    conteo_clases = Counter()
    for clases in clases_por_imagen.values():
        for clase in clases:
            conteo_clases[clase] += 1

    story.append(Paragraph("<b>Conteo de imágenes por clase de texto</b>", styles['Heading2']))
    for clase, cant in conteo_clases.most_common():
        story.append(Paragraph(f"- {clase}: {cant} imágenes", styles['Normal']))
    story.append(Spacer(1, 12))

    doc.build(story)


def main():
    # paths
    zip_path = '../data/raw/cocotext.v2.zip'
    json_file = 'cocotext.v2.json'
    folder_train = '../data/raw/subset/train2014/train2014'
    folder_val = '../data/raw/subset/val2014'
    salida_pdf = '../reports/eda_cocotext.pdf'

    coco_dict = load_anns(zip_path, json_file)

    ids_train = get_id(coco_dict, subset='train')
    ids_val = get_id(coco_dict, subset='val')

    grouped_anns = group_anns(coco_dict)

    imgs_train = get_imgs(ids_train, folder_train)
    imgs_val = get_imgs(ids_val, folder_val)

    eda_cocotext_pdf(grouped_anns, imgs_train, imgs_val, salida_pdf)


if __name__ == "__main__":
    main()

