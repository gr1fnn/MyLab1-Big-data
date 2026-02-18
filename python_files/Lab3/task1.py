import pandas as pd
import numpy as np
from ml_pipeline import *

train_df = load_data('train.csv', 'playground-series-s3e12')
test_df = load_data('test.csv', 'playground-series-s3e12')

print("1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ\n")
print_basic_info(train_df, "train")
print_basic_info(test_df, "test")

target_col = 'gravity'
print(f"\nЦелевая переменная: '{target_col}' \n")

feature_cols = ['ph', 'osmo', 'cond', 'urea', 'calc']
X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

print(f"\nПризнаки для обучения: {feature_cols}")
print(f"Размер X: {X.shape}")
print(f"Размер X_test: {X_test.shape}")

X, y, threshold = prepare_features(X, y, binary=True)

numeric_cols = feature_cols.copy()
analyze_numeric_variables(X, numeric_cols)
print("\nD. Категориальные переменные: отсутствуют")

print("\n2. ПОДГОТОВКА ДАТАСЕТА")

train_stats = {}
train_stats.update(handle_missing_values(X, numeric_cols))
train_stats.update(handle_outliers(X, numeric_cols))
print(f"\nC. Категориальных переменных: 0")

X_train, X_val, y_train, y_val, scaler = split_and_scale(X, y)

print("\n3. ПОСТРОЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ\n")
models = get_models()
results = train_models(models, X_train, y_train, X_val)


print("\n4. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ КЛАССИФИКАЦИИ\n")
metrics_data, best_model_name, best_f1 = evaluate_models(results, y_val)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ\n")
metrics_df = pd.DataFrame(metrics_data)
print(metrics_df.to_string(index=False))
print(f"\nЛучшая модель по F1-мере: {best_model_name} (F1 = {best_f1:.4f})")

plot_roc_curves(results, y_val)

print("\n5. ПРЕДСКАЗАНИЯ ДЛЯ ТЕСТОВОГО НАБОРА\n")
X_submit_scaled = process_test_data(X_test, numeric_cols, X, train_stats, scaler)
best_model = results[best_model_name]['model']
predictions = best_model.predict(X_submit_scaled)

print("\n6. ВЫВОДЫ")
print(f"""
1. Проведен разведочный анализ данных
2. Целевая переменная: '{target_col}' 
3. Задача: БИНАРНАЯ КЛАССИФИКАЦИЯ (предсказание high/low gravity)
4. Выполнена предобработка: пропуски, выбросы, нормализация
5. Построены 3 модели классификации: KNN, Логистическая регрессия, SVM
6. Метрики качества на валидации:
   {metrics_df.to_string(index=False).replace(chr(10), chr(10) + '   ')}
7. Лучшая модель: {best_model_name} (F1 = {best_f1:.4f})
8. Сделаны предсказания для тестового набора ({len(predictions)} записей)
""")
