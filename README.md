
# Автоматизация администрирования MLOps II
# Домашнее задание №2. Части 1, 2 и 3 (HW)

### Цель проекта:
```
Ознакомиться с основами управления данными с помощью DVC, управления экспериментами с 
использованием MLflow и автоматизации с ClearML. Основная задача — интегрировать все 
три инструмента для построения полного цикла ML-проекта.
```

### Структура проекта

```
.
├── ...
├── .dvc                        # data version control folder 
├── .github                     # Метаданные и конфигурация DVC 
│   └── workflows               # GitHub Actions Workflows для CI/CD
│       └── dvc-ci.yml      
├── data                        # Директория для хранения файлов данных
│   ├── iris_dataset.csv        # Исходный датасет
│   └── proc_iris_dataset.csv   # результат обработки
├── docs                        # Вспомогательный, скриншоты выполнения
├── scripts                     # Исходный код для обработки данных и обучения модели
│   ├── download.py             # загрузка датасета
│   ├── pipeline.py             # пайплайн ClearML
│   ├── process_data.py         # обработка датасета
│   └── train.py                # обучение модели, MLflow-эксперименты
.dvcignore                      # Игнорировать файлы, которые не нужно коммитить в DVC
dvc.lock                        # созданные DVC файлы
dvc.yaml                        # созданные DVC файлы
README.md                       # документация проекта
requirements.txt                # зависимости
```

### Часть 1: управление данными с DVC

> [!IMPORTANT] 
> *Задача: использовать DVC для управления данными и построения ML-пайплайнов. 
> Настроить удаленное хранилище и запустить пайплайн с использованием CI/CD.*
 

#### Этапы выполнения (самоконтроль)
`1. Добавление данных в DVC:`
- Добавьте набор данных в проект, используя DVC.
- Закоммитьте изменения и добавьте DVC файлы в Git.

`2. Настройка удаленного хранилища:`
- Настройте удаленное хранилище для DVC (например, Google Drive, AWS S3).
- Убедитесь, что данные могут синхронизироваться с удаленным хранилищем.

`3. Создание и запуск пайплайна:`
- Создайте пайплайн для обработки данных (например, очистка данных или подготовка признаков).
- Закоммитьте пайплайн в DVC.

`4. Интеграция DVC в CI/CD:`
- Настройте пайплайн в CI/CD, который будет автоматически запускать DVC-процесс.
- Убедитесь, что пайплайн может корректно запускать шаги обработки данных.

#### Результат выполнения
![Result](./docs/Screenshot_1.png)

### Часть 2: управление экспериментами с MLflow

> [!IMPORTANT] 
> *Задача: настроить MLflow для управления экспериментами и их сравнением*