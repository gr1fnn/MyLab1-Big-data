import pandas as pd
import numpy as np
from ml_pipeline import *

df = load_data('WineQT.csv')

print("1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ\n")
print_basic_info(df)

print(f"\nЦелевая переменная: 'quality' \n")

feature_cols = [col for col in df.columns if col not in ['quality', 'Id', 'id']]
print(f"Признаки для обучения: {feature_cols}")

X = df[feature_cols].copy()
y = df['quality'].copy()
print(f"\nРазмер X: {X.shape}")

X, y, threshold = prepare_features(X, y, threshold=7, binary=True)

numeric_cols = feature_cols.copy()
analyze_numeric_variables(X, numeric_cols)
print("\nD. Категориальные переменные: отсутствуют")

print("\n2. ПОДГОТОВКА ДАТАСЕТА")

handle_missing_values(X, numeric_cols)
handle_outliers(X, numeric_cols)
print(f"\nC. Категориальных переменных: 0")

X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

print("\n3. ПОСТРОЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ\n")
models = get_models()
results = train_models(models, X_train, y_train, X_test)

print("\n4. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ КЛАССИФИКАЦИИ\n")
metrics_data, best_model_name, best_f1 = evaluate_models(results, y_test)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ\n")
metrics_df = pd.DataFrame(metrics_data)
print(metrics_df.to_string(index=False))
print(f"\nЛучшая модель по F1-мере: {best_model_name} (F1 = {best_f1:.4f})")

plot_roc_curves(results, y_test)

print("\n6. ВЫВОДЫ")
print(f"""
1. Проведен разведочный анализ данных
2. Целевая переменная: 'quality' 
3. Задача: БИНАРНАЯ КЛАССИФИКАЦИЯ (quality >= 7 - хорошее вино)
4. Выполнена предобработка: пропуски, выбросы, нормализация
5. Построены 3 модели классификации: KNN, Логистическая регрессия, SVM
6. Метрики качества на тестовой выборке:
   {metrics_df.to_string(index=False).replace(chr(10), chr(10) + '   ')}
7. Лучшая модель: {best_model_name} (F1 = {best_f1:.4f})
""")
