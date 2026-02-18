import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve)
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filename, folder=None):
    """Загрузка данных из файла"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if folder:
        path = os.path.join(current_dir, folder, filename)
    else:
        path = os.path.join(current_dir, filename)
    return pd.read_csv(path)


def print_basic_info(df, name="датафрейм"):
    """Базовая информация о данных"""
    print(f"A. Строк в {name}: {df.shape[0]}, Столбцов: {df.shape[1]}")
    print(f"B. Память ({name}): {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nСтолбцы в датасете:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")


def analyze_numeric_variables(X, numeric_cols):
    """Анализ интервальных переменных"""
    print("\nC. Интервальные переменные:")
    for col in numeric_cols:
        col_data = X[col].dropna()
        q25, q75 = np.percentile(col_data, [25, 75])
        print(f"{col:25} мин={col_data.min():8.3f}, медиана={col_data.median():8.3f}, "
              f"среднее={col_data.mean():8.3f}, макс={col_data.max():8.3f}, "
              f"25%={q25:8.3f}, 75%={q75:8.3f}")


def handle_missing_values(X, numeric_cols):
    """Обработка пропусков"""
    print("A. Обработка пропусков:")
    train_stats = {}  # Словарь для сохранения параметров обработки
    
    for col in numeric_cols:
        # проверяем пропуски
        if X[col].isnull().any():
            median_val = X[col].median()
            # fillna() заменяет пропуски, inplace=True изменяет исходный DataFrame
            X[col].fillna(median_val, inplace=True)

            train_stats[f'{col}_median'] = median_val
            print(f"  {col}: заполнено медианой {median_val:.3f}")
    return train_stats


def handle_outliers(X, numeric_cols):
    """Обработка выбросов методом IQR (межквартильного размаха)"""
    print("\nB. Обработка выбросов:")
    train_stats = {}
    
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)  
        Q3 = X[col].quantile(0.75)  
        IQR = Q3 - Q1  
        
        # Границы для выбросов: Q1 - 1.5*IQR и Q3 + 1.5*IQR
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Находим выбросы (значения за границами)
        outliers = X[(X[col] < lower) | (X[col] > upper)]
        if len(outliers) > 0:
            # np.where(условие, значение_если_истина, значение_если_ложь)
            X[col] = np.where(X[col] < lower, lower, X[col])
            X[col] = np.where(X[col] > upper, upper, X[col])
            # Сохраняем границы для тестовых данных
            train_stats[f'{col}_lower'] = lower
            train_stats[f'{col}_upper'] = upper
            print(f"  {col}: заменено {len(outliers)} выбросов")
    return train_stats


def prepare_features(X, y, threshold=None, binary=True):
    """Подготовка признаков и бинаризация целевой переменной"""
    if binary:
        print(f"\nУникальные значения в целевой переменной: {sorted(y.unique())}")
        print(f"Распределение:\n{y.value_counts().sort_index()}")
        
        # Если порог не задан, используем медиану
        if threshold is None:
            threshold = y.median()
        # Преобразуем в бинарные значения: 1 если >= порога, иначе 0
        y_binary = (y >= threshold).astype(int)
        print(f"\nПреобразовано в бинарную классификацию (порог = {threshold:.2f})")
        print(f"Распределение классов:\n{y_binary.value_counts()}")
        return X, y_binary, threshold
    return X, y, None


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Разделение на train/test и масштабирование"""
    # train_test_split разделяет данные
    # test_size=0.2 означает 20% на тест, 80% на обучение
    # stratify=y сохраняет пропорции классов в обучающей и тестовой выборках
    # random_state для воспроизводимости результатов
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nD. Разделение: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    # стандартизирует признаки: (x - mean) / std

    scaler = StandardScaler()
    # вычисляет параметры (mean, std) на train и применяет их
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_models():
    """Инициализация моделей"""
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),  # 5 ближайших соседей
        'Логистическая регрессия': LogisticRegression(max_iter=1000, random_state=42),  # max_iter - макс. итераций для сходимости
        'SVM': SVC(probability=True, random_state=42)  # probability=True для получения вероятностей классов
    }


def train_models(models, X_train, y_train, X_test):
    """Обучение моделей и получение предсказаний"""
    results = {}
    for name, model in models.items():
        # подбор параметров
        model.fit(X_train, y_train)
        # (0 или 1)
        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {'pred': y_pred, 'proba': y_proba, 'model': model}
        print(f"  {name}: обучена")
    return results


def evaluate_models(results, y_test):
    """Оценка качества моделей"""
    metrics_data = []
    best_f1 = 0
    best_model_name = ""
    
    for name in results.keys():
        y_pred = results[name]['pred']
        y_proba = results[name]['proba']
        
        acc = accuracy_score(y_test, y_pred)  # Доля правильных ответов
        prec = precision_score(y_test, y_pred, zero_division=0)  # TP / (TP + FP)
        rec = recall_score(y_test, y_pred, zero_division=0)  # TP / (TP + FN)
        f1 = f1_score(y_test, y_pred, zero_division=0)  # Гармоническое среднее precision и recall
        roc_auc = roc_auc_score(y_test, y_proba)  # Площадь под ROC-кривой
        
        metrics_data.append({
            'Модель': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        })
        
        # Отслеживаем лучшую модель по F1-мере
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{name}:")
        print(f"  Точность (Accuracy): {acc:.4f}")
        print(f"  Точность (Precision): {prec:.4f}")
        print(f"  Полнота (Recall): {rec:.4f}")
        print(f"  F1-мера: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Матрица ошибок:")
        # Форматированный вывод матрицы 2x2
        print(f"  [{cm[0,0]:4d} {cm[0,1]:4d}]")
        print(f"  [{cm[1,0]:4d} {cm[1,1]:4d}]")
    
    return metrics_data, best_model_name, best_f1


def plot_roc_curves(results, y_test):
    """Построение ROC-кривых"""
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
    
    for name in results.keys():
        y_proba = results[name]['proba']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.xlabel('Доля ложных положительных срабатываний (False Positive Rate)', fontsize=11)
    plt.ylabel('Доля истинных положительных срабатываний (True Positive Rate)', fontsize=11)
    plt.title('ROC-кривые для задач классификации', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def process_test_data(X_test, numeric_cols, train_data, train_stats, scaler):
    """Обработка тестовых данных и предсказания"""
    X_submit = X_test.copy()
    
    for col in numeric_cols:
        if col in X_submit.columns:
            # Заполнение пропусков медианой из обучающей выборки
            if X_submit[col].isnull().any():
                median_val = train_stats.get(f'{col}_median', train_data[col].median())
                X_submit[col].fillna(median_val, inplace=True)
            
            # Обработка выбросов с границами из обучающей выборки
            lower = train_stats.get(f'{col}_lower', 
                                   train_data[col].quantile(0.25) - 1.5 * (train_data[col].quantile(0.75) - train_data[col].quantile(0.25)))
            upper = train_stats.get(f'{col}_upper',
                                   train_data[col].quantile(0.75) + 1.5 * (train_data[col].quantile(0.75) - train_data[col].quantile(0.25)))
            # Обрезаем значения за границами
            X_submit[col] = np.where(X_submit[col] < lower, lower, X_submit[col])
            X_submit[col] = np.where(X_submit[col] > upper, upper, X_submit[col])
    
    X_submit_scaled = scaler.transform(X_submit)
    return X_submit_scaled