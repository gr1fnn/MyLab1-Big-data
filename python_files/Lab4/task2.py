import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from regression_pipeline import *

print("ЗАДАЧА 2: ПРОГНОЗИРОВАНИЕ ЦЕНЫ АВТОМОБИЛЯ (РЕГРЕССИЯ)")

# ЗАГРУЗКА ДАННЫХ
df = load_data('CarPrice_Assignment.csv')
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

# 1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ
print("\n1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
print_basic_info(df, "CarPrice")

# ЦЕЛЕВАЯ ПЕРЕМЕННАЯ - price
target_col = 'price'
print(f"\nЦелевая переменная: '{target_col}' - цена автомобиля ($)")

# Признаки - все колонки кроме car_ID и price
feature_cols = [col for col in df.columns if col not in ['car_ID', target_col]]
print(f"Всего признаков: {len(feature_cols)}")

# Отделяем признаки от целевой переменной
X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"\nСтатистика целевой переменной (цена):")
print(f"  мин: {y.min():.2f} $")
print(f"  макс: {y.max():.2f} $")
print(f"  среднее: {y.mean():.2f} $")
print(f"  медиана: {y.median():.2f} $")
print(f"  стандартное отклонение: {y.std():.2f} $")

# Определяем типы переменных
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nТипы переменных:")
print(f"  Интервальных (числовых): {len(numeric_cols)}")
print(f"  Категориальных: {len(categorical_cols)}")
print(f"  Категориальные переменные: {categorical_cols}")

# 2. АНАЛИЗ ПЕРЕМЕННЫХ
print("\nc. АНАЛИЗ ИНТЕРВАЛЬНЫХ ПЕРЕМЕННЫХ")
analyze_numeric_variables(X, numeric_cols[:10]) 

print("\nd. АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
analyze_categorical_variables(X, categorical_cols)

# 3. ПОДГОТОВКА ДАТАСЕТА
print("\n2. ПОДГОТОВКА ДАТАСЕТА")

# a. Обработка пропусков
print("\na. Обработка пропусков:")
train_stats = {}
train_stats.update(handle_missing_values(X, numeric_cols, categorical_cols))

# b. Обработка выбросов
print("\nb. Обработка выбросов:")
train_stats.update(handle_outliers_iqr(X, numeric_cols))

# c. Кодирование категориальных переменных
print("\nc. Кодирование категориальных переменных:")

encoding_method = 'onehot'  # One-Hot Encoding наиболее подходит для автомобильных характеристик
encoders = encode_categorical_variables(X, categorical_cols, method=encoding_method, y=y)

# d. ПРОВЕРКА ГИПОТЕЗ
print("\nd. ПРОВЕРКА ГИПОТЕЗ")

# Гипотеза 1: Мощность двигателя (horsepower) положительно коррелирует с ценой
if 'horsepower' in df.columns:
    corr_horsepower = np.corrcoef(df['horsepower'], df[target_col])[0, 1]
    print(f"\nГИПОТЕЗА 1: Мощность двигателя влияет на цену автомобиля")
    print(f"   Формулировка: Чем выше мощность двигателя, тем дороже автомобиль")
    print(f"   Коэффициент корреляции Пирсона: {corr_horsepower:.4f}")
    
    if corr_horsepower > 0.7:
        print(f"   ✓ Гипотеза ПОДТВЕРЖДЕНА: очень сильная положительная связь")
        print(f"     Мощность двигателя - ключевой фактор ценообразования")
    elif corr_horsepower > 0.5:
        print(f"   ✓ Гипотеза ПОДТВЕРЖДЕНА: сильная положительная связь")
        print(f"     Мощность значительно влияет на цену автомобиля")
    elif corr_horsepower > 0.3:
        print(f"   ~ Гипотеза ЧАСТИЧНО ПОДТВЕРЖДЕНА: умеренная положительная связь")
        print(f"     Мощность влияет на цену, но есть и другие важные факторы")
    elif corr_horsepower > 0:
        print(f"   ~ Гипотеза СЛАБО ПОДТВЕРЖДЕНА: слабая положительная связь")
        print(f"     Мощность не является определяющим фактором цены")
    else:
        print(f"   ✗ Гипотеза НЕ ПОДТВЕРЖДЕНА: отрицательная или нулевая связь")

# Гипотеза 2: Тип кузова (carbody) влияет на цену автомобиля
if 'carbody' in df.columns:
    carbody_mean = df.groupby('carbody')[target_col].mean().sort_values(ascending=False)
    print(f"\nГИПОТЕЗА 2: Тип кузова влияет на цену автомобиля")
    print(f"   Формулировка: Разные типы кузова имеют разную рыночную стоимость")
    print(f"\n   Средняя цена по типам кузова:")
    for body, price in carbody_mean.items():
        print(f"     {body:15}: {price:8.2f} $")
    
    # Находим максимальную разницу
    max_price = carbody_mean.max()
    min_price = carbody_mean.min()
    diff = max_price - min_price
    diff_percent = (diff / min_price) * 100
    
    print(f"\n   Разница между самым дорогим и дешевым: {diff:.2f} $ ({diff_percent:.1f}%)")
    
    if diff_percent > 50:
        print(f"   ✓ Гипотеза ПОДТВЕРЖДЕНА: тип кузова сильно влияет на цену")
        print(f"     Разница в цене между типами кузова составляет более 50%")
    elif diff_percent > 25:
        print(f"   ✓ Гипотеза ПОДТВЕРЖДЕНА: тип кузова умеренно влияет на цену")
        print(f"     Разница в цене между типами кузова составляет {diff_percent:.1f}%")
    elif diff_percent > 10:
        print(f"   ~ Гипотеза ЧАСТИЧНО ПОДТВЕРЖДЕНА: тип кузова слабо влияет на цену")
        print(f"     Разница в цене между типами кузова незначительна")
    else:
        print(f"   ✗ Гипотеза НЕ ПОДТВЕРЖДЕНА: тип кузова не влияет на цену")

# e. Разделение на трейн и тест
print("\ne. РАЗДЕЛЕНИЕ ДАННЫХ")
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

# 4. ОБУЧЕНИЕ МОДЕЛЕЙ
print("\n3. ОБУЧЕНИЕ РЕГРЕССИОННЫХ МОДЕЛЕЙ")

# Выбираем ТОЛЬКО ДВЕ модели для обучения: KNN и ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

models_to_train = {
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
}

results = train_regression_models(models_to_train, X_train, y_train, X_test)

# 5. ОЦЕНКА КАЧЕСТВА
print("\n4. ОЦЕНКА КАЧЕСТВА РЕГРЕССИОННЫХ МОДЕЛЕЙ")

metrics_data, best_model_name, best_r2 = evaluate_regression_models(results, y_test)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ РЕГРЕССИИ")

metrics_df = pd.DataFrame(metrics_data)
print(metrics_df.to_string(index=False))

print(f"\nЛучшая модель по R2: {best_model_name} (R2 = {best_r2:.4f})")

# Обоснование выбора лучшей модели
print("\nОБОСНОВАНИЕ ВЫБОРА ЛУЧШЕЙ МОДЕЛИ")

best_model_metrics = metrics_df[metrics_df['Модель'] == best_model_name].iloc[0]

print(f"""
Лучшая модель: {best_model_name}

Обоснование выбора:
1. R2 Score = {best_r2:.4f} - {'отличная' if best_r2 > 0.85 else 'хорошая' if best_r2 > 0.7 else 'средняя' if best_r2 > 0.5 else 'низкая'} 
   точность, модель объясняет {best_r2*100:.1f}% вариации цены автомобиля

2. RMSE = {best_model_metrics['RMSE']:.2f} $ - 
   {'низкая' if best_model_metrics['RMSE'] < 2000 else 'средняя' if best_model_metrics['RMSE'] < 4000 else 'высокая'} 
   ошибка прогноза (в среднем {best_model_metrics['RMSE']/y.mean()*100:.1f}% от средней цены)

3. MAPE = {best_model_metrics['MAPE(%)']:.2f}% - 
   {'отличная' if best_model_metrics['MAPE(%)'] < 10 else 'хорошая' if best_model_metrics['MAPE(%)'] < 15 else 'удовлетворительная'} 
   точность прогнозирования

4. Обобщающая способность:
   - Модель {'отлично' if best_r2 > 0.8 else 'хорошо' if best_r2 > 0.7 else 'удовлетворительно'} 
     обобщается на тестовых данных
   - {'Низкий риск переобучения' if best_r2 > 0.7 else 'Требуется дополнительная настройка'}

Вывод: Модель {best_model_name} показывает {'наилучшие' if best_r2 == max([m['R2'] for m in metrics_data]) else 'конкурентоспособные'} 
результаты и рекомендуется для прогнозирования цен на автомобили.
""")
# 6. ВИЗУАЛИЗАЦИЯ
print("5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

# График предсказаний для лучшей модели
plot_knn_vs_elasticnet(results, y_test)

# График важности признаков для линейной модели
if best_model_name == 'ElasticNet':
    best_model = results[best_model_name]['model']
    if hasattr(best_model, 'coef_'):
        # Получаем имена признаков после One-Hot Encoding
        feature_names = X.columns.tolist()
        coefficients = best_model.coef_
        
        # Берем топ-10 признаков по абсолютному значению коэффициента
        top_indices = np.argsort(np.abs(coefficients))[-10:]
        top_features = [feature_names[i] for i in top_indices]
        top_coefs = coefficients[top_indices]
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if c > 0 else 'red' for c in top_coefs]
        plt.barh(range(len(top_coefs)), top_coefs, color=colors)
        plt.yticks(range(len(top_coefs)), top_features)
        plt.xlabel('Коэффициент регрессии', fontsize=12)
        plt.title('Топ-10 наиболее важных признаков (ElasticNet)', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# 7. СОХРАНЕНИЕ МОДЕЛИ
print("6. СОХРАНЕНИЕ МОДЕЛИ")
model_filename = f"carprice_{best_model_name.lower().replace(' ', '_')}_model.joblib"
save_model(results[best_model_name]['model'], model_filename)

# 8. ВЫВОДЫ
print("7. ВЫВОДЫ")
print(f"""
ИТОГОВЫЙ ОТЧЕТ ПО ЗАДАЧЕ 2 (РЕГРЕССИЯ):

1. ДАННЫЕ:
   - Датасет: CarPrice_Assignment
   - Количество записей: {df.shape[0]}
   - Количество признаков: {len(feature_cols)}
   - Целевая переменная: {target_col} (цена автомобиля)
   - Интервальных переменных: {len(numeric_cols)}
   - Категориальных переменных: {len(categorical_cols)}
   - Метод кодирования: {encoding_method}

2. ПРЕДОБРАБОТКА:
   - Пропуски: {'обнаружены и обработаны' if train_stats else 'не обнаружены'}
   - Выбросы: обработаны методом IQR
   - Кодирование: {encoding_method}
   - Масштабирование: StandardScaler

3. МОДЕЛИ:
   {', '.join(models_to_train.keys())}

4. МЕТРИКИ КАЧЕСТВА:
{metrics_df.to_string(index=False).replace(chr(10), chr(10) + '   ')}

5. ЛУЧШАЯ МОДЕЛЬ:
   - {best_model_name}
   - R2 Score: {best_r2:.4f}
   - RMSE: {best_model_metrics['RMSE']:.2f} $
   - MAE: {best_model_metrics['MAE']:.2f} $
   - MAPE: {best_model_metrics['MAPE(%)']:.2f}%

6. ПРОВЕРКА ГИПОТЕЗ:
   
   Гипотеза 1 (мощность → цена):
   - Коэффициент корреляции: {corr_horsepower:.4f}
   - Статус: {'ПОДТВЕРЖДЕНА' if corr_horsepower > 0.5 else 'ЧАСТИЧНО ПОДТВЕРЖДЕНА' if corr_horsepower > 0.3 else 'НЕ ПОДТВЕРЖДЕНА'}
   - Вывод: {'Мощность двигателя - важный фактор ценообразования' if corr_horsepower > 0.5 else 'Мощность не является определяющим фактором'}
   
   Гипотеза 2 (тип кузова → цена):
   - Разница между max и min: {diff:.2f} $ ({diff_percent:.1f}%)
   - Статус: {'ПОДТВЕРЖДЕНА' if diff_percent > 50 else 'ЧАСТИЧНО ПОДТВЕРЖДЕНА' if diff_percent > 25 else 'НЕ ПОДТВЕРЖДЕНА'}
   - Вывод: {'Тип кузова существенно влияет на цену' if diff_percent > 50 else 'Влияние типа кузова умеренное'}

7. ПРОГНОЗ:
   - Модель обучена на {X_train.shape[0]} записях
   - Протестирована на {X_test.shape[0]} записях
   - Модель сохранена в файл: {model_filename}

8. РЕКОМЕНДАЦИИ:
   - {'Использовать ' + best_model_name + ' для прогнозирования цен' if best_r2 > 0.7 else 'Требуется улучшение модели'}
   - {'Добавить больше признаков для повышения точности' if best_r2 < 0.7 else 'Модель готова к использованию'}
   - {'Попробовать ансамблевые методы (Random Forest, XGBoost)' if best_r2 < 0.8 else ''}
""")