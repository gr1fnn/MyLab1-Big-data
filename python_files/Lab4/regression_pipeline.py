import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             r2_score, mean_absolute_percentage_error)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filename, folder='csvfiles'):
    """Загрузка данных из папки csvfiles"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, folder, filename)
    return pd.read_csv(path)


def save_model(model, filename):
    """Сохранение модели с помощью joblib"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'models', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Модель сохранена: {path}")


def load_model(filename):
    """Загрузка модели с помощью joblib"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'models', filename)
    return joblib.load(path)


def print_basic_info(df, name="датафрейм"):
    """Базовая информация о данных"""
    print(f"a. Строк в {name}: {df.shape[0]}, Столбцов: {df.shape[1]}")
    print(f"b. Память ({name}): {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nСтолбцы в датасете:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")


def analyze_numeric_variables(X, numeric_cols):
    """Анализ интервальных переменных - мин, медиана, среднее, макс, персентили 25, 75"""
    print("\nc. Интервальные переменные:")
    for col in numeric_cols:
        col_data = X[col].dropna()
        q25, q75 = np.percentile(col_data, [25, 75])
        print(f"{col:30} мин={col_data.min():12.3f}, медиана={col_data.median():12.3f}, "
              f"среднее={col_data.mean():12.3f}, макс={col_data.max():12.3f}, "
              f"25%={q25:12.3f}, 75%={q75:12.3f}")


def analyze_categorical_variables(X, categorical_cols):
    """Анализ категориальных переменных - мода и частота"""
    print("\nd. Категориальные переменные:")
    if not categorical_cols:
        print("  Категориальные переменные отсутствуют")
        return
    
    for col in categorical_cols:
        mode_val = X[col].mode()[0]
        mode_count = (X[col] == mode_val).sum()
        print(f"{col:30} мода={mode_val:12}, встречается {mode_count:6} раз ({mode_count/len(X)*100:5.1f}%)")


def handle_missing_values(X, numeric_cols, categorical_cols):
    """Обработка пропусков - замена медианой/модой"""
    print("\na. Обработка пропусков:")
    train_stats = {}
    
    # Для числовых - медиана
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            train_stats[f'{col}_median'] = median_val
            print(f"  {col}: заполнено медианой {median_val:.3f}")
    
    # Для категориальных - мода
    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
            train_stats[f'{col}_mode'] = mode_val
            print(f"  {col}: заполнено модой {mode_val}")
    
    return train_stats


def handle_outliers_iqr(X, numeric_cols):
    """Обработка выбросов методом IQR - замена граничными значениями"""
    print("\nb. Обработка выбросов:")
    train_stats = {}
    
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = X[(X[col] < lower) | (X[col] > upper)]
        if len(outliers) > 0:
            X[col] = np.where(X[col] < lower, lower, X[col])
            X[col] = np.where(X[col] > upper, upper, X[col])
            train_stats[f'{col}_lower'] = lower
            train_stats[f'{col}_upper'] = upper
            print(f"  {col}: заменено {len(outliers)} выбросов ({len(outliers)/len(X)*100:5.1f}%)")
    
    return train_stats



def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Разделение на трейн/тест и масштабирование"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\ne. Разделение: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_regression_models():
    """Инициализация регрессионных моделей"""
    return {
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'LASSO': Lasso(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    }


def train_regression_models(models, X_train, y_train, X_test):
    """Обучение регрессионных моделей и получение предсказаний"""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {'pred': y_pred, 'model': model}
        print(f"  {name}: обучена")
    return results


def evaluate_regression_models(results, y_test):
    """Оценка качества регрессионных моделей - MAE, MSE, RMSE, MAPE, R2"""
    metrics_data = []
    best_r2 = -np.inf
    best_model_name = ""
    
    for name in results.keys():
        y_pred = results[name]['pred']
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)
        
        metrics_data.append({
            'Модель': name,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAPE(%)': round(mape, 2),
            'R2': round(r2, 4)
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R2: {r2:.4f}")
    
    return metrics_data, best_model_name, best_r2


def plot_models_comparison(results, y_test):
    """Построение отдельных графиков для линейных моделей и KNN"""
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Цвета для моделей
    colors = {'Linear Regression': 'green', 'Ridge': 'orange', 'KNN': 'purple'}
    
    # ГРАФИК 1: 
    ax1.axline([0, 0], [1, 1], color='red', linestyle='--', lw=2, 
               label='Идеальная линия', alpha=0.7)
    
    for model_name in ['Linear Regression', 'Ridge']:
        if model_name in results:
            y_pred = results[model_name]['pred']
            r2 = r2_score(y_test, y_pred)
            ax1.scatter(y_test, y_pred, alpha=0.5, s=20, 
                       color=colors[model_name], 
                       label=f'{model_name} (R²={r2:.3f})')
    
    ax1.set_xlabel('Фактические продажи (млн $)', fontsize=12)
    ax1.set_ylabel('Предсказанные продажи (млн $)', fontsize=12)
    ax1.set_title('Линейные модели: Linear Regression vs Ridge', fontsize=14, pad=15)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.2)
    
    # Добавляем текст с пояснением для линейных моделей
    textstr1 = 'Линейные модели предполагают\nлинейную зависимость между\nпризнаками и целевой переменной'
    props1 = dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props1)
    
    # ГРАФИК 2: KNN
    ax2.axline([0, 0], [1, 1], color='red', linestyle='--', lw=2, 
               label='Идеальная линия', alpha=0.7)
    
    if 'KNN' in results:
        y_pred_knn = results['KNN']['pred']
        r2_knn = r2_score(y_test, y_pred_knn)
        ax2.scatter(y_test, y_pred_knn, alpha=0.6, s=25, 
                   color=colors['KNN'], 
                   label=f'KNN (k=5, R²={r2_knn:.3f})')
    
    ax2.set_xlabel('Фактические продажи (млн $)', fontsize=12)
    ax2.set_ylabel('Предсказанные продажи (млн $)', fontsize=12)
    ax2.set_title('Метод ближайших соседей: KNN', fontsize=14, pad=15)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.2)
    
    # Добавляем текст с пояснением для KNN
    textstr2 = 'KNN ищет похожие объекты\nи усредняет их значения\n(нелинейный метод)'
    props2 = dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
    ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props2)
    
    plt.suptitle('Сравнение подходов к прогнозированию продаж', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_knn_vs_elasticnet(results, y_test):
    """Построение отдельных графиков для KNN и ElasticNet"""
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Цвета для моделей
    colors = {'KNN': 'purple', 'ElasticNet': 'brown'}
    
    # ГРАФИК 1: KNN
    ax1.axline([0, 0], [1, 1], color='red', linestyle='--', lw=2, 
               label='Идеальная линия', alpha=0.7)
    
    if 'KNN' in results:
        y_pred_knn = results['KNN']['pred']
        r2_knn = r2_score(y_test, y_pred_knn)
        ax1.scatter(y_test, y_pred_knn, alpha=0.6, s=25, 
                   color=colors['KNN'], 
                   label=f'KNN (k=5, R²={r2_knn:.3f})')
    
    ax1.set_xlabel('Фактическая цена ($)', fontsize=12)
    ax1.set_ylabel('Предсказанная цена ($)', fontsize=12)
    ax1.set_title('Метод ближайших соседей (KNN)', fontsize=14, pad=15)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.2)
    
    # Добавляем текст с пояснением для KNN
    textstr1 = 'KNN:\n• Ищет похожие автомобили\n• Усредняет их цены\n• Нелинейный метод\n• Чувствителен к масштабу'
    props1 = dict(boxstyle='round', facecolor='lavender', alpha=0.7)
    ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props1)
    
    # ГРАФИК 2: ElasticNet
    ax2.axline([0, 0], [1, 1], color='red', linestyle='--', lw=2, 
               label='Идеальная линия', alpha=0.7)
    
    if 'ElasticNet' in results:
        y_pred_en = results['ElasticNet']['pred']
        r2_en = r2_score(y_test, y_pred_en)
        ax2.scatter(y_test, y_pred_en, alpha=0.6, s=25, 
                   color=colors['ElasticNet'], 
                   label=f'ElasticNet (R²={r2_en:.3f})')
    
    ax2.set_xlabel('Фактическая цена ($)', fontsize=12)
    ax2.set_ylabel('Предсказанная цена ($)', fontsize=12)
    ax2.set_title('ElasticNet регрессия', fontsize=14, pad=15)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.2)
    
    # Добавляем текст с пояснением для ElasticNet
    textstr2 = 'ElasticNet:\n• Комбинация L1 и L2 регуляризации\n• Отбирает важные признаки\n• Устойчив к мультиколлинеарности\n• Линейная модель'
    props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props2)
    
    # Добавляем метрики качества на графики
    if 'KNN' in results:
        mae_knn = mean_absolute_error(y_test, results['KNN']['pred'])
        rmse_knn = np.sqrt(mean_squared_error(y_test, results['KNN']['pred']))
        mape_knn = mean_absolute_percentage_error(y_test, results['KNN']['pred']) * 100
        metrics_text1 = f'MAE: ${mae_knn:.2f}\nRMSE: ${rmse_knn:.2f}\nMAPE: {mape_knn:.2f}%'
        ax1.text(0.95, 0.05, metrics_text1, transform=ax1.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    if 'ElasticNet' in results:
        mae_en = mean_absolute_error(y_test, results['ElasticNet']['pred'])
        rmse_en = np.sqrt(mean_squared_error(y_test, results['ElasticNet']['pred']))
        mape_en = mean_absolute_percentage_error(y_test, results['ElasticNet']['pred']) * 100
        metrics_text2 = f'MAE: ${mae_en:.2f}\nRMSE: ${rmse_en:.2f}\nMAPE: {mape_en:.2f}%'
        ax2.text(0.95, 0.05, metrics_text2, transform=ax2.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle('Сравнение KNN и ElasticNet для прогнозирования цен автомобилей', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def process_test_data(X_test, numeric_cols, categorical_cols, train_data, train_stats, encoders, scaler):
    """Обработка тестовых данных для предсказаний"""
    X_submit = X_test.copy()
    
    # Обработка пропусков
    for col in numeric_cols:
        if col in X_submit.columns:
            if X_submit[col].isnull().any():
                median_val = train_stats.get(f'{col}_median', train_data[col].median())
                X_submit[col].fillna(median_val, inplace=True)
    
    for col in categorical_cols:
        if col in X_submit.columns:
            if X_submit[col].isnull().any():
                mode_val = train_stats.get(f'{col}_mode', train_data[col].mode()[0])
                X_submit[col].fillna(mode_val, inplace=True)
    
    # Обработка выбросов
    for col in numeric_cols:
        if col in X_submit.columns:
            lower = train_stats.get(f'{col}_lower', 
                                   train_data[col].quantile(0.25) - 1.5 * (train_data[col].quantile(0.75) - train_data[col].quantile(0.25)))
            upper = train_stats.get(f'{col}_upper',
                                   train_data[col].quantile(0.75) + 1.5 * (train_data[col].quantile(0.75) - train_data[col].quantile(0.25)))
            X_submit[col] = np.where(X_submit[col] < lower, lower, X_submit[col])
            X_submit[col] = np.where(X_submit[col] > upper, upper, X_submit[col])
    
    # Кодирование категориальных переменных
    for col in categorical_cols:
        if col in X_submit.columns and col in encoders:
            le = encoders[col]
            # Преобразуем только известные значения
            X_submit[col] = X_submit[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    
    # Масштабирование
    X_submit_scaled = scaler.transform(X_submit)
    
    return X_submit_scaled

def encode_categorical_variables(X, categorical_cols, method='onehot', y=None):
    """Кодирование категориальных переменных"""
    print(f"\nc. Категориальных переменных: {len(categorical_cols)}")
    
    if not categorical_cols:
        print("  Категориальные переменные отсутствуют")
        return {}
    
    encoders = {}
    
    if method == 'label':
        # Label Encoding
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            print(f"  {col}: закодирована (LabelEncoder)")
    
    elif method == 'onehot':
        # One-Hot Encoding
        print(f"  Применяем One-Hot Encoding для {len(categorical_cols)} переменных:")
        for col in categorical_cols:
            print(f"    - {col}: {X[col].nunique()} уникальных значений")
        
        # Получаем dummy переменные
        dummies = pd.get_dummies(X[categorical_cols], prefix=categorical_cols, drop_first=True)
        
        # Удаляем исходные категориальные колонки
        X.drop(columns=categorical_cols, inplace=True)
        
        # Добавляем dummy переменные
        for col in dummies.columns:
            X[col] = dummies[col]
            encoders[col] = 'onehot'
        
        print(f"  Создано {len(dummies.columns)} новых признаков")
    
    elif method == 'freq':
        # Frequency Encoding
        for col in categorical_cols:
            freq_encoding = X[col].value_counts().to_dict()
            X[col] = X[col].map(freq_encoding)
            encoders[col] = freq_encoding
            print(f"  {col}: закодирована частотой встречаемости (Frequency Encoding)")
    
    elif method == 'mean_target' and y is not None:
        # Mean Target Encoding
        for col in categorical_cols:
            mean_encoding = y.groupby(X[col]).mean().to_dict()
            X[col] = X[col].map(mean_encoding)
            encoders[col] = mean_encoding
            print(f"  {col}: закодирована средним значением целевой переменной (Mean Target Encoding)")
    
    return encoders


def process_test_data_categorical(X_test, categorical_cols, encoders, method='onehot', y=None):
    """Обработка тестовых данных для категориальных переменных"""
    X_submit = X_test.copy()
    
    if method == 'label':
        for col in categorical_cols:
            if col in X_submit.columns and col in encoders:
                le = encoders[col]
                X_submit[col] = X_submit[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    elif method == 'onehot':
        # Для One-Hot Encoding нужно создать те же dummy переменные
        for col in categorical_cols:
            if col in X_submit.columns:
                unique_values = X_submit[col].unique()
                for val in unique_values:
                    dummy_col = f"{col}_{val}"
                    if dummy_col in encoders:
                        X_submit[dummy_col] = (X_submit[col] == val).astype(int)
                X_submit.drop(columns=[col], inplace=True)
        
        # Добавляем недостающие колонки
        for col in encoders.keys():
            if col not in X_submit.columns:
                X_submit[col] = 0
    
    elif method == 'freq':
        for col in categorical_cols:
            if col in X_submit.columns and col in encoders:
                freq_dict = encoders[col]
                X_submit[col] = X_submit[col].map(freq_dict).fillna(0)
    
    elif method == 'mean_target':
        for col in categorical_cols:
            if col in X_submit.columns and col in encoders:
                mean_dict = encoders[col]
                # Используем среднее значение из словаря или глобальное среднее
                global_mean = y.mean() if y is not None else 0
                X_submit[col] = X_submit[col].map(mean_dict).fillna(global_mean)
    
    return X_submit