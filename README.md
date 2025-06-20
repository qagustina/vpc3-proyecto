#  Trabajo Final - Visión por Computadora III (CEIA)

###  Integrantes

- Florentino Arias  
- Juan Cruz Piñero  
- Agustina Quiros  
- Agustín de la Vega  

### Traducción de Imágenes a Texto 
Aplicación de modelos de OCR sobre el dataset COCO-Text.
Este proyecto explora la capacidad de modelos de OCR basados en transformers para transcribir texto presente en imágenes naturales. Se comparan dos enfoques:  
- **TrOCR**, orientado a la transcripción directa de texto en regiones específicas.  
- **Donut**, diseñado para el entendimiento estructurado de documentos completos.

Se trabajó con el dataset **COCO-Text**, realizando un proceso de *fine-tuning* y evaluación basado en métricas como **Mean Character Accuracy** y **Character Error Rate**.

#### 🖼️Dataset: [Coco-Text dataset](https://bgshih.github.io/cocotext/)

#### 🤗 Modelos Utilizados

- [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)

- [Donut](https://huggingface.co/naver-clova-ix/donut-base)



<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>




## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         vpc3-proyecto and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── vpc3-proyecto   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes vpc3-proyecto a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Data preprocessing

### 🖥️ Método 1: Notebook `aq-data-processing.ipynb`

#### Pasos de Ejecución
1. Abrir el notebook en Jupyter:
2. Modificar la variable requerida para leer la ubicación de los archivos zip del dataset cocotext.
3. ejecutar la notebook
4. se generaran carpetas en la carpeta `data` (relativa a la ubicación del notebook) con los archivos procesados.

### 🖥️ Método 2: Desde submódulo vp3_proyecto ejecutar notebook `data/dataset_split.ipynb`

#### Pasos de Ejecución
1. Setear variables de entorno en archivo .env en la raiz del submódulo (vpc3-proyecto)
2. ejecutar notebook `data/dataset_split.ipynb`
3. se generaran carpetas en la carpeta indicada en la variable de entorno `PROCESSED_DATA_DIR` con las imagenes divididas en subdirectorios segun la proporción indicada.
--------

## Fine tuning

### 1. Setup archivo `.env`
Modificar un archivo `.env` en el directorio raíz del proyecto con este contenido:

```ini
PROCESSED_DATA_DIR=/ruta/absoluta/a/data/processed ==> ruta donde se quiere generar el split del dataset
RAW_DATA_DIR=/ruta/absoluta/a/data/raw ==> ruta a .zip donde estan las imagenes y las anottations
CHECKPOINT_DIR=/ruta/absoluta/a/checkpoints ==> carpeta de guardado de modelos (ej: /home/juan/CEIA/vpc3_proyecto/models)
```
### Modificar configuracion de parámetros de entrenamiento según necesidad.

### Ejecutar notebook correspondiente al modelo deseado

Por ejemplo, para donut, se debe ejecutar la notebook 'vpc3_proyecto/model_training/donut_fine_tuning.ipynb'

El modelo resultante será guardado en el directorio indicado.

## Evaluación de modelo

Para esto se pueden utilizar las notebooks dentro de 'vpc3_proyecto/model_evaluation/'

Es necesario tener las variables de entorno configuradas en el archivo `.env` correspondiente.

### Ejecutar notebook correspondiente al modelo deseado

Por ejemplo, para donut, se debe ejecutar la notebook 'vpc3_proyecto/model_evaluation/donut_evaluation.ipynb'

Es posible que sea necesario modificar la notebook para indicar el directorio especifico en el cual se encuentra el modelo guardado.

### Visualización de atención

