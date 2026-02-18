import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline    
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')


os.makedirs('results', exist_ok=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'testdat.csv')
df = pd.read_csv(file_path)

print("\n1. Первые 20 строк датасета:")
print(df.head(20).to_string())

print("\n2. Информация о датасете (типы данных, пропуски):")
print(df.info())

print("\n3. Базовая статистика по числовым признакам:")
print(df.describe())

print("\n4. Базовая статистика по категориальным признакам:")
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
else:
    print("Категориальные признаки отсутствуют")

print(f"1. Сколько наблюдений в датасете? {df.shape[0]}")
print(f"2. Сколько признаков (столбцов)? {df.shape[1]}")

target_col = df.columns[-1]
print(f"   Целевая переменная: '{target_col}'")

print(f"3. Есть ли пропущенные значения? {'Да' if df.isnull().any().any() else 'Нет'}")
if df.isnull().any().any():
    print(f"   Пропуски в столбцах: {df.columns[df.isnull().any()].tolist()}")
print(f"4. Все ли столбцы являются признаками? Нет, столбец '{target_col}' - целевая переменная")
print(f"5. Какие признаки нуждаются в преобразовании?")
print(f"   - Категориальные признаки: {list(categorical_cols) if len(categorical_cols)>0 else 'отсутствуют'} (нужно One-Hot Encoding)")
print(f"   - Числовые признаки: нуждаются в масштабировании")
print(f"   - Пропуски: нужно заполнить (медианой для чисел, модой для категорий)")


# График распределения целевого признака
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df[target_col], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Продажи (млн $)')
plt.ylabel('Частота')
plt.title('Распределение целевой переменной')

# Boxplot для целевой переменной
plt.subplot(1, 3, 2)
plt.boxplot(df[target_col])
plt.ylabel('Продажи (млн $)')
plt.title('Boxplot целевой переменной')
plt.grid(True, alpha=0.3)

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove(target_col)
key_feature = numeric_features[0] if numeric_features else None

if key_feature:
    plt.subplot(1, 3, 3)
    plt.scatter(df[key_feature], df[target_col], alpha=0.5)
    plt.xlabel(key_feature)
    plt.ylabel(target_col)
    plt.title(f'{key_feature} vs {target_col}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/distributions.png')
plt.show()

Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
print(f"1. Есть ли выбросы в целевом признаке? Да, {len(outliers)} выбросов ({len(outliers)/len(df)*100:.1f}%)")
print(f"   Диапазон нормальных значений: [{lower_bound:.2f}, {upper_bound:.2f}]")

if key_feature:
    corr = df[key_feature].corr(df[target_col])
    print(f"2. Видна ли линейная зависимость между {key_feature} и продажами?")
    print(f"   Корреляция = {corr:.4f} - {'сильная положительная' if corr > 0.7 else 'умеренная' if corr > 0.5 else 'слабая'} связь")


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)  

print(f"Категориальные признаки до кодирования: {categorical_cols}")
print(f"Числовые признаки: {numerical_cols}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())                    
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))  
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X = df.drop(columns=[target_col])
y = df[target_col]

X_processed = preprocessor.fit_transform(X)

feature_names = numerical_cols.copy()
if len(categorical_cols) > 0:
    encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = encoder.get_feature_names_out(categorical_cols)
    feature_names.extend(cat_features)

print(f"\nНазвания всех столбцов после обработки:")
for i, name in enumerate(feature_names[:20]):  # Показываем первые 20, если их много
    print(f"  {i+1}. {name}")
if len(feature_names) > 20:
    print(f"  ... и еще {len(feature_names) - 20} признаков")

print(f"\nФорма датасета после обработки: {X_processed.shape}")


print(f"\n1. Нужно ли было преобразование? {'Да' if categorical_cols else 'Нет, категориальных признаков нет'}")
if categorical_cols:
    unique_counts = [df[col].nunique() for col in categorical_cols]
    new_features = sum(unique_counts) - len(categorical_cols)  # drop_first=True
    print(f"2. Сколько новых признаков появилось после One-Hot Encoding? {new_features}")
    print(f"   Всего признаков стало: {X_processed.shape[1]}")


X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
df_encoded = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

correlations = df_encoded.corr()[target_col].drop(target_col).sort_values(ascending=False)

print("\nТоп-10 признаков по корреляции с целевой переменной:")
for i, (feature, corr) in enumerate(correlations.head(10).items()):
    print(f"  {i+1}. {feature}: {corr:.4f}")

plt.figure(figsize=(16, 12))
top_features = correlations.head(20).index.tolist()
top_features.append(target_col)
corr_matrix = df_encoded[top_features].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5)
plt.title('Корреляционная матрица (топ-20 признаков)')
plt.tight_layout()
plt.savefig('results/correlation_heatmap.png')
plt.show()



# Самый коррелированный признак
if len(correlations) > 0:
    top_feature = correlations.index[0]
    top_corr = correlations.iloc[0]
    print(f"1. Какой признак имеет самую высокую корреляцию с целевым? {top_feature} ({top_corr:.4f})")

    # Сильно коррелированные между собой признаки
    corr_matrix_full = df_encoded.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix_full.columns)):
        for j in range(i+1, len(corr_matrix_full.columns)):
            if abs(corr_matrix_full.iloc[i, j]) > 0.8:
                feat1 = corr_matrix_full.columns[i]
                feat2 = corr_matrix_full.columns[j]
                if feat1 != target_col and feat2 != target_col:
                    high_corr_pairs.append((feat1, feat2, corr_matrix_full.iloc[i, j]))

    print(f"2. Есть ли сильно коррелированные между собой признаки (кроме целевой)?")
    if high_corr_pairs:
        print(f"   Найдено {len(high_corr_pairs)} пар с корреляцией > 0.8:")
        for i, (f1, f2, c) in enumerate(high_corr_pairs[:3]):  # Показываем первые 3
            print(f"   - {f1} и {f2}: {c:.4f}")
    else:
        print(f"   Нет сильно коррелированных признаков")

    high_pos = correlations[correlations > 0.5]
    high_neg = correlations[correlations < -0.5]
    print(f"3. Признаки с корреляцией с целевым > 0.5: {len(high_pos)}")
    for feat, corr in high_pos.items():
        print(f"   - {feat}: {corr:.4f}")
    print(f"   Признаки с корреляцией < -0.5: {len(high_neg)}")
    for feat, corr in high_neg.items():
        print(f"   - {feat}: {corr:.4f}")


X_processed_array = X_processed  #
y_array = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X_processed_array, y_array, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape[0]} строк")
print(f"Размер тестовой выборки: {X_test.shape[0]} строк")


knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Предсказания на обучающей и тестовой выборках
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

#метрики качества
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\nМетрики на обучающей выборке:")
print(f"  MAE: {train_mae:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  R2: {train_r2:.4f}")

print("\nМетрики на тестовой выборке:")
print(f"  MAE: {test_mae:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  R2: {test_r2:.4f}")

model_filename = 'results/knn_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump({
        'model': knn_model,
        'preprocessor': preprocessor,  
        'feature_names': feature_names
    }, f)
print(f"\nМодель и препроцессор сохранены в {model_filename}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_array.min(), y_array.max()], [y_array.min(), y_array.max()], 'r--', lw=2)
plt.xlabel('Фактические продажи')
plt.ylabel('Предсказанные продажи')
plt.title(f'KNN: Факт vs Предсказание\nR2 = {test_r2:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные продажи')
plt.ylabel('Остатки (факт - предсказание)')
plt.title('График остатков')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_results.png')
plt.show()

print(f"1. Какой результат получился на тестовой выборке?")
print(f"   - R2 Score: {test_r2:.4f} (модель объясняет {test_r2*100:.1f}% дисперсии)")
print(f"   - MAE: {test_mae:.4f} млн $ (средняя ошибка прогноза)")
print(f"   - RMSE: {test_rmse:.4f} млн $")

diff_r2 = train_r2 - test_r2
print(f"\n2. Есть ли признаки переобучения?")
if diff_r2 > 0.1:
    print(f"   ДА. Разница между train и test R2: {diff_r2:.4f} (> 0.1)")
    print(f"   Модель показывает значительно лучшие результаты на обучающих данных")
elif diff_r2 > 0.05:
    print(f"   УМЕРЕННО. Разница между train и test R2: {diff_r2:.4f}")
    print(f"   Модель немного переобучилась, но это приемлемо")
else:
    print(f"   НЕТ. Разница между train и test R2: {diff_r2:.4f}")
    print(f"   Модель хорошо обобщается на новых данных")

if train_r2 > 0.95:
    print(f"   Также обращает внимание очень высокий R2 на train ({train_r2:.4f}) - это может указывать на переобучение")

print("\nИТОГОВЫЙ ВЫВОД:")

print(f"""
На основе проведенного анализа:

1. Тип задачи: РЕГРЕССИЯ (целевая переменная - непрерывная величина - продажи в млн $)

2. Ключевые признаки: 
   - Наибольшую корреляцию с продажами имеет {top_feature if 'top_feature' in locals() else 'не определено'} ({top_corr if 'top_corr' in locals() else 'не определена'})
   - Всего выявлено {len(high_pos) if 'high_pos' in locals() else 0} признаков с положительной корреляцией > 0.5
   - {len(high_neg) if 'high_neg' in locals() else 0} признаков с отрицательной корреляцией < -0.5

3. Качество модели KNN:
   - R2 на тесте: {test_r2:.4f}
   - Средняя ошибка (MAE): {test_mae:.4f} млн $
   - {'Модель можно использовать для прогнозирования' if test_r2 > 0.6 else 'Качество модели среднее, требуется улучшение'}

4. {'ЕСТЬ' if diff_r2 > 0.1 else 'НЕТ'} признаки переобучения

5. Рекомендации:
   - {'Попробовать другие модели (Ridge, ElasticNet)' if diff_r2 > 0.1 else 'Модель работает хорошо'}
   - {'Настроить гиперпараметры KNN' if test_r2 < 0.6 else 'Можно использовать текущую модель'}
""")