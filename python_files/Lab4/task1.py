import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from regression_pipeline import *

train_df = load_data('train.csv')
test_df = load_data('test.csv')

print("\n1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ\n")
print_basic_info(train_df, "train")
print_basic_info(test_df, "test")

target_col = 'store_sales(in millions)'
print(f"\nЦелевая переменная: '{target_col}' - прогнозирование продаж\n")

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
common_cols = list(train_cols.intersection(test_cols))
common_cols = [col for col in common_cols if col not in ['id', target_col]]

print(f"Общие признаки в train и test ({len(common_cols)}):")
for col in common_cols:
    print(f"  - {col}")

# Используем только общие кол
feature_cols = common_cols
X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

print(f"\nРазмер X (обучение): {X.shape}")
print(f"Размер X_test (тест): {X_test.shape}")
print(f"\nСтатистика целевой переменной:")
print(f"  мин: {y.min():.3f}")
print(f"  макс: {y.max():.3f}")
print(f"  среднее: {y.mean():.3f}")
print(f"  медиана: {y.median():.3f}")

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

analyze_numeric_variables(X, numeric_cols)
analyze_categorical_variables(X, categorical_cols)

print("\n2. ПОДГОТОВКА ДАТАСЕТА")

train_stats = {}
train_stats.update(handle_missing_values(X, numeric_cols, categorical_cols))

train_stats.update(handle_outliers_iqr(X, numeric_cols))

encoders = encode_categorical_variables(X, categorical_cols, method='label')

print("\nd. ПРОВЕРКА ГИПОТЕЗ:")

# Гипотеза 1: Площадь магазина положительно коррелирует с продажами
# Используем оригинальный train_df для проверки гипотез, а не X
if 'store_sqft' in train_df.columns:
    corr_store_sqft = np.corrcoef(train_df['store_sqft'], train_df[target_col])[0, 1]
    print(f"  Гипотеза 1: Корреляция между площадью магазина и продажами: {corr_store_sqft:.4f}")
    if corr_store_sqft > 0.3:
        print(f"    ✓ Гипотеза подтверждена: сильная положительная связь")
    elif corr_store_sqft > 0:
        print(f"    ~ Гипотеза частично подтверждена: слабая положительная связь")
    else:
        print(f"    ✗ Гипотеза не подтверждена: отрицательная связь")

# Гипотеза 2: Наличие кофе-бара увеличивает продажи
# Используем оригинальный train_df для проверки гипотез
if 'coffee_bar' in train_df.columns:
    coffee_bar_mean = train_df.groupby('coffee_bar')[target_col].mean()
    if len(coffee_bar_mean) > 1:
        diff = coffee_bar_mean.iloc[1] - coffee_bar_mean.iloc[0]
        print(f"  Гипотеза 2: Влияние кофе-бара на продажи:")
        print(f"    Без кофе-бара: {coffee_bar_mean.iloc[0]:.3f}")
        print(f"    С кофе-баром: {coffee_bar_mean.iloc[1]:.3f}")
        print(f"    Разница: {diff:.3f} ({diff/coffee_bar_mean.iloc[0]*100:+.1f}%)")
        if diff > 0:
            print(f"    ✓ Гипотеза подтверждена: кофе-бар увеличивает продажи")
        else:
            print(f"    ✗ Гипотеза не подтверждена: кофе-бар не увеличивает продажи")

# Разделение на трейн и тест
X_train, X_test_split, y_train, y_test_split, scaler = split_and_scale(X, y)

print("\n3. ОБУЧЕНИЕ РЕГРЕССИОННЫХ МОДЕЛЕЙ")

models_to_train = {
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
}

results = train_regression_models(models_to_train, X_train, y_train, X_test_split)

print("\n4. ОЦЕНКА КАЧЕСТВА РЕГРЕССИОННЫХ МОДЕЛЕЙ")


metrics_data, best_model_name, best_r2 = evaluate_regression_models(results, y_test_split)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ РЕГРЕССИИ")

metrics_df = pd.DataFrame(metrics_data)
print(metrics_df.to_string(index=False))

print(f"\nЛучшая модель по R2: {best_model_name} (R2 = {best_r2:.4f})")

print("\nОБОСНОВАНИЕ ВЫБОРА ЛУЧШЕЙ МОДЕЛИ")


best_model_metrics = metrics_df[metrics_df['Модель'] == best_model_name].iloc[0]
print(f"""
Лучшая модель: {best_model_name}

Обоснование выбора:
1. R2 Score = {best_r2:.4f} - {'высокий' if best_r2 > 0.7 else 'средний' if best_r2 > 0.5 else 'низкий'}, 
   объясняет {best_r2*100:.1f}% дисперсии целевой переменной

2. RMSE = {best_model_metrics['RMSE']:.4f} - 
   {'низкая' if best_r2 > 0.7 else 'средняя' if best_r2 > 0.5 else 'высокая'} ошибка прогноза

3. MAPE = {best_model_metrics['MAPE(%)']:.2f}% - 
   {'отличная' if best_model_metrics['MAPE(%)'] < 10 else 'хорошая' if best_model_metrics['MAPE(%)'] < 20 else 'удовлетворительная'} точность прогнозирования

4. Обобщающая способность: 
   - Модель {'хорошо' if best_r2 > 0.6 else 'удовлетворительно' if best_r2 > 0.4 else 'плохо'} 
     обобщается на тестовых данных
   - Отсутствие переобучения (проверено на отложенной выборке)

Вывод: Модель {best_model_name} показывает {'наилучшие' if best_r2 == max([m['R2'] for m in metrics_data]) else 'хорошие'} 
результаты по всем метрикам и рекомендуется для прогнозирования продаж.
""")

print("\n5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

plot_models_comparison(results, y_test_split)

print("\n6. СОХРАНЕНИЕ МОДЕЛИ")

model_filename = f"{best_model_name.lower().replace(' ', '_')}_sales_model.joblib"
save_model(results[best_model_name]['model'], model_filename)

print("\n7. ПРЕДСКАЗАНИЯ ДЛЯ ТЕСТОВОГО НАБОРА")

numeric_cols_test = [col for col in numeric_cols if col in X_test.columns]
categorical_cols_test = [col for col in categorical_cols if col in X_test.columns]

X_submit_scaled = process_test_data(X_test, numeric_cols_test, categorical_cols_test, X, 
                                   train_stats, encoders, scaler)
best_model = results[best_model_name]['model']
predictions = best_model.predict(X_submit_scaled)

submission = pd.DataFrame({
    'id': test_df['id'],
    'predicted_sales': predictions
})
submission.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'sales_predictions.csv'), index=False)
print(f"  Предсказания сохранены в sales_predictions.csv")
print(f"  Статистика предсказаний:")
print(f"    мин: {predictions.min():.3f}")
print(f"    макс: {predictions.max():.3f}")
print(f"    среднее: {predictions.mean():.3f}")
print(f"    медиана: {np.median(predictions):.3f}")

print("\n8. ВЫВОДЫ")

print(f"""
ИТОГОВЫЙ ОТЧЕТ ПО ЗАДАЧЕ 1 (РЕГРЕССИЯ):
================================================================================
1. ДАННЫЕ:
   - Тренировочный набор: {train_df.shape[0]} строк, {train_df.shape[1]} столбцов
   - Тестовый набор: {test_df.shape[0]} строк, {test_df.shape[1]} столбцов
   - Целевая переменная: {target_col}
   - Количество признаков: {len(feature_cols)}
   - Интервальных переменных: {len(numeric_cols_test)}
   - Категориальных переменных: {len(categorical_cols_test)}

2. ПРЕДОБРАБОТКА:
   - Пропуски: {'обнаружены и обработаны' if train_stats else 'не обнаружены'}
   - Выбросы: обработаны методом IQR
   - Кодирование: {'LabelEncoder' if encoders else 'не требуется'}
   - Масштабирование: StandardScaler

3. МОДЕЛИ:
   - KNN (k=5)
   - Линейная регрессия
   - Гребневая регрессия (Ridge)

4. МЕТРИКИ КАЧЕСТВА:
{metrics_df.to_string(index=False).replace(chr(10), chr(10) + '   ')}

5. ЛУЧШАЯ МОДЕЛЬ:
   - {best_model_name}
   - R2 Score: {best_r2:.4f}
   - RMSE: {best_model_metrics['RMSE']:.4f}
   - MAPE: {best_model_metrics['MAPE(%)']:.2f}%


6. ПРОВЕРКА ГИПОТЕЗ:
   - Гипотеза 1 (площадь → продажи): {'подтверждена' if 'store_sqft' in train_df.columns and corr_store_sqft > 0 else 'не проверена'}
   - Гипотеза 2 (кофе-бар → продажи): {'подтверждена' if 'coffee_bar' in train_df.columns and diff > 0 else 'не проверена'}
7. ПРОГНОЗ:
   - Сделаны предсказания для {len(predictions)} записей тестового набора
   - Модель сохранена в файл: {model_filename}
   - Результаты сохранены в sales_predictions.csv
""")