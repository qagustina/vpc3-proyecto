import json
import matplotlib.pyplot as plt
import os
import re  # Para manejar la extracción de números de los nombres de carpeta si es necesario


def plot_comparison_metrics(json_paths, metric_names=["cer", "wer"], title_prefix="Comparación de Métricas",
                            save_dir="metric_plots"):
    """
    Carga métricas de múltiples archivos JSON y genera gráficos comparativos.

    Args:
        json_paths (list): Una lista de rutas a los archivos 'metrics.json' (o directorios
                           que contengan 'metrics.json' anidados, como carpetas de checkpoint).
                           Puede ser una lista de strings o una lista de listas de strings si
                           quieres agrupar rutas para una sola serie en el gráfico.
        metric_names (list): Lista de nombres de métricas a graficar (ej., ["cer", "wer"]).
        title_prefix (str): Prefijo para el título de los gráficos.
        save_dir (str): Directorio donde se guardarán los gráficos generados.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_metrics_data = {}

    for i, path_group in enumerate(json_paths):
        if isinstance(path_group, str):
            path_group = [path_group]  # Asegura que siempre sea una lista para iterar

        run_name = f"Run {i + 1}"
        run_metrics = {}

        for path in path_group:
            metrics_file = path
            if os.path.isdir(path):
                # Si es un directorio, busca 'metrics.json' dentro
                if "checkpoint-" in os.path.basename(path):
                    # Asume que es una carpeta de checkpoint de Trainer
                    # Intentaremos extraer el paso para un nombre más descriptivo
                    match = re.search(r"checkpoint-(\d+)", os.path.basename(path))
                    if match:
                        step = match.group(1)
                        run_name = f"Checkpoint {step}"
                    else:
                        run_name = os.path.basename(path)

                metrics_file = os.path.join(path, "metrics.json")
                if not os.path.exists(metrics_file):
                    print(f"Advertencia: No se encontró 'metrics.json' en '{path}'. Saltando.")
                    continue

            if not os.path.exists(metrics_file):
                print(f"Advertencia: El archivo '{metrics_file}' no existe. Saltando.")
                continue

            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    # Si el JSON contiene un diccionario con la clave "metrics" (como en tu save_results)
                    if "metrics" in metrics and isinstance(metrics["metrics"], dict):
                        for metric_name in metric_names:
                            if metric_name in metrics["metrics"]:
                                run_metrics.setdefault(metric_name, []).append(metrics["metrics"][metric_name])
                    else:  # Si el JSON es directamente el diccionario de métricas
                        for metric_name in metric_names:
                            if metric_name in metrics:
                                run_metrics.setdefault(metric_name, []).append(metrics[metric_name])

            except json.JSONDecodeError:
                print(f"Error: No se pudo decodificar JSON de '{metrics_file}'.")
            except Exception as e:
                print(f"Error al procesar '{metrics_file}': {e}")

        # Solo añade a los datos si se encontraron métricas para este grupo
        if run_metrics:
            # Si hay múltiples valores para una métrica en un solo grupo (ej. varios checkpoints)
            # podemos promediarlos o tomar el último/primero. Por simplicidad, tomaremos el primero si hay múltiples.
            # Podrías modificar esto para mostrar la progresión si un 'path_group' representa checkpoints de una sola corrida.
            final_run_metrics = {k: v[0] if v else float('nan') for k, v in run_metrics.items()}
            all_metrics_data[run_name] = final_run_metrics

    if not all_metrics_data:
        print("No se encontraron datos de métricas para graficar.")
        return

    # Preparar datos para graficar
    labels = list(all_metrics_data.keys())

    # Crear un gráfico para cada métrica solicitada
    for metric_name in metric_names:
        values = [data.get(metric_name, float('nan')) for data in all_metrics_data.values()]

        # Filtra 'nan' para que no rompa el gráfico o muestre conexiones raras
        valid_labels = [label for label, value in zip(labels, values) if
                        not (isinstance(value, float) and value != value)]
        valid_values = [value for value in values if not (isinstance(value, float) and value != value)]

        if not valid_values:
            print(f"No hay datos válidos para la métrica '{metric_name}'. Saltando gráfico.")
            continue

        plt.figure(figsize=(10, 6))

        # Si hay solo un punto, plotearlo como un scatter, si hay más, como línea
        if len(valid_values) == 1:
            plt.scatter(valid_labels, valid_values, marker='o', s=100, color='blue')
        else:
            plt.plot(valid_labels, valid_values, marker='o', linestyle='-', markersize=8)

        plt.title(f"{title_prefix}: {metric_name.upper()}")
        plt.xlabel("Ejecución/Modelo")
        plt.ylabel(metric_name.upper())
        plt.grid(True)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric_name}_comparison.png"))
        plt.close()
        print(f"Gráfico para '{metric_name}' guardado en '{os.path.join(save_dir, f'{metric_name}_comparison.png')}'")

