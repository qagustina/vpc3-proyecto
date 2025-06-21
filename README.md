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

--------

