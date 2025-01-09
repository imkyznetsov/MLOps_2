import os
import pandas as pd
from sklearn.datasets import load_iris

def download_data(output_path):
    # Получаем путь к корневой директории проекта
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Путь к скрипту
    root_dir = os.path.dirname(script_dir)  # Поднимаемся на уровень выше (к корню проекта)

    # Полный путь к output_path
    full_output_path = os.path.join(root_dir, output_path)

    # Убедимся, что директория для сохранения существует
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

    # Загружаем датасет iris
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Сохраняем датасет в файл
    df.to_csv(full_output_path, index=False)
    print(f"Файл сохранен по пути: {full_output_path}")


# Используем функцию для сохранения данных
download_data("data/iris_dataset.csv")
