import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
file_path = r"C:\Users\grish\OneDrive\Рабочий стол\учба\8 семестр\Крутских Елена Игоревна\Лаб1\MyLab1\python_files\Lab3\playground-series-s3e12\test.csv"
df = pd.read_csv(file_path)

# ============================================================================
# 1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ
# ============================================================================
print("="*50)
print("1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
print("="*50)

# A. Сколько строк и столбцов
print(f"A. Размер датафрейма: {df.shape[0]} строк, {df.shape[1]} столбцов")

# B. Объем памяти
print(f"B. Объем памяти: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# C. Для интервальных переменных
print("\nC. Статистика для интервальных переменных:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    numeric_stats = df[numeric_cols].agg(['min', 'median', 'mean', 'max', 
                                          lambda x: np.percentile(x, 25), 
                                          lambda x: np.percentile(x, 75)])
    numeric_stats = numeric_stats.rename(index={'<lambda_0>': '25%', '<lambda_1>': '75%'})
    print(numeric_stats)
else:
    print("Нет интервальных переменных")

# D. Для категориальных переменных
print("\nD. Мода для категориальных переменных:")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    mode_val = df[col].mode()[0] if not df[col].mode().empty else None
    mode_count = (df[col] == mode_val).sum() if mode_val else 0
    print(f"  {col}: мода = '{mode_val}', встречается {mode_count} раз ({mode_count/len(df)*100:.1f}%)")

# Просмотр первых строк данных
print("\nПервые 5 строк данных:")
print(df.head())

# Информация о типах данных
print("\nИнформация о данных:")
print(df.info())

# Проверка на пропуски
print("\nПропуски в данных:")
print(df.isnull().sum())

# ============================================================================
# 2. ПОДГОТОВКА ДАТАСЕТА
# ============================================================================
print("\n" + "="*50)
print("2. ПОДГОТОВКА ДАТАСЕТА")
print("="*50)

# A. Анализ и обработка пропусков
print("A. Обработка пропусков:")
missing_cols = df.columns[df.isnull().any()].tolist()
if missing_cols:
    print(f"  Пропуски найдены в колонках: {missing_cols}")
    
    # Для числовых колонок - заполняем медианой
    num_missing = [col for col in missing_cols if col in numeric_cols]
    for col in num_missing:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Заполнены пропуски в '{col}' медианой: {df[col].median():.2f}")
    
    # Для категориальных колонок - заполняем модой
    cat_missing = [col for col in missing_cols if col in categorical_cols]
    for col in cat_missing:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_val, inplace=True)
        print(f"  Заполнены пропуски в '{col}' модой: '{mode_val}'")
else:
    print("  Пропусков не обнаружено")

# B. Анализ и обработка выбросов (используем метод IQR)
print("\nB. Анализ выбросов (метод IQR):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if not outliers.empty:
        print(f"  В '{col}' найдено {len(outliers)} выбросов")
        # Заменяем выбросы граничными значениями
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        print(f"  Выбросы заменены граничными значениями: [{lower_bound:.2f}, {upper_bound:.2f}]")

# C. Обработка категориальных переменных
print("\nC. Обработка категориальных переменных:")
print(f"  Найдено категориальных переменных: {len(categorical_cols)}")

# Кодируем категориальные переменные с помощью LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Закодирована переменная '{col}'")

# ============================================================================
# 3. РАЗДЕЛЕНИЕ НА ТРЕНИНГ И ТЕСТ
# ============================================================================
print("\n" + "="*50)
print("3. РАЗДЕЛЕНИЕ ДАТАСЕТА")
print("="*50)

# ВАЖНО: В загруженном файле test.csv нет целевой переменной
# Предположим, что у нас есть данные для обучения в другом файле
# Для демонстрации создадим искусственную целевую переменную

# Если в данных есть колонка с названием 'target' или похожим, используем её
target_candidates = ['target', 'Target', 'class', 'Class', 'y', 'Y']

# Проверяем, есть ли какая-либо из этих колонок в данных
target_col = None
for candidate in target_candidates:
    if candidate in df.columns:
        target_col = candidate
        break

if target_col:
    print(f"Найдена целевая переменная: '{target_col}'")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
else:
    print("Целевая переменная не найдена. Создаю искусственную для демонстрации...")
    # Создаем искусственную целевую переменную на основе первой числовой колонки
    if len(numeric_cols) > 0:
        median_val = df[numeric_cols[0]].median()
        y = (df[numeric_cols[0]] > median_val).astype(int)
        X = df.copy()
        print(f"Искусственная целевая создана на основе '{numeric_cols[0]}'")
    else:
        # Если нет числовых колонок, используем случайную
        np.random.seed(42)
        y = pd.Series(np.random.randint(0, 2, len(df)))
        X = df.copy()
        print("Искусственная целевая создана случайно")

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
)

print(f"Размер тренировочной выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Баланс классов в тестовой выборке:")
print(pd.Series(y_test).value_counts(normalize=True))

# ============================================================================
# 4. ПОСТРОЕНИЕ МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*50)
print("4. ПОСТРОЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ")
print("="*50)

# Инициализация моделей
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)  # probability=True для ROC-AUC
}

# Обучение и предсказание
results = {}
for name, model in models.items():
    print(f"\nОбучение модели: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Сохраняем результаты
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# ============================================================================
# 5. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*50)
print("5. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("="*50)

# Создаем таблицу для сравнения метрик
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 
                                   'F1-Score', 'ROC-AUC'])

for name in models.keys():
    y_pred = results[name]['y_pred']
    y_pred_proba = results[name]['y_pred_proba']
    
    # Рассчитываем метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC (если есть вероятности)
    roc_auc = None
    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None
    
    # Добавляем в таблицу
    metrics_df = pd.concat([metrics_df, pd.DataFrame([{
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc if roc_auc else 'N/A'
    }])], ignore_index=True)
    
    # Выводим confusion matrix
    print(f"\n{name} - Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Выводим отчет классификации
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Выводим таблицу сравнения моделей
print("\n" + "="*50)
print("СРАВНЕНИЕ МОДЕЛЕЙ:")
print("="*50)
print(metrics_df.to_string(index=False))

# Определяем лучшую модель по F1-Score
if not metrics_df['F1-Score'].isin(['N/A']).all():
    best_model_idx = metrics_df['F1-Score'].astype(float).idxmax()
    best_model = metrics_df.loc[best_model_idx, 'Model']
    print(f"\nЛучшая модель по F1-Score: {best_model}")
else:
    # Если нет F1-Score, используем Accuracy
    best_model_idx = metrics_df['Accuracy'].astype(float).idxmax()
    best_model = metrics_df.loc[best_model_idx, 'Model']
    print(f"\nЛучшая модель по Accuracy: {best_model}")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "="*50)
print("6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*50)

# ROC-кривые (если есть вероятности)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ROC-кривые
axes[0].set_title('ROC-кривые')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')

for name in models.keys():
    y_pred_proba = results[name]['y_pred_proba']
    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        except:
            pass

axes[0].legend(loc='lower right')
axes[0].grid(True)

# Сравнение метрик
metrics_for_plot = metrics_df.copy()
if 'ROC-AUC' in metrics_for_plot.columns:
    metrics_for_plot['ROC-AUC'] = pd.to_numeric(metrics_for_plot['ROC-AUC'], errors='coerce')

# Выбираем только числовые метрики
numeric_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
available_metrics = [m for m in numeric_metrics if m in metrics_for_plot.columns]

if len(available_metrics) > 0:
    plot_data = metrics_for_plot.set_index('Model')[available_metrics]
    
    x = np.arange(len(plot_data.columns))
    width = 0.2
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        axes[1].bar(x + i*width - width*(len(plot_data)-1)/2, 
                   row.values, 
                   width, 
                   label=idx)
    
    axes[1].set_title('Сравнение метрик моделей')
    axes[1].set_xlabel('Метрики')
    axes[1].set_ylabel('Значение')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_data.columns)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'Нет данных для визуализации\nметрик', 
                ha='center', va='center', fontsize=12)
    axes[1].set_title('Сравнение метрик моделей')

plt.tight_layout()
plt.show()

# ============================================================================
# ВЫВОДЫ
# ============================================================================
print("\n" + "="*50)
print("ЗАКЛЮЧЕНИЕ")
print("="*50)
print("""
В ходе выполнения задачи были выполнены следующие шаги:

1. Загружен датасет и проведен разведочный анализ:
   - Определено количество строк и столбцов
   - Оценен объем занимаемой памяти
   - Проанализированы интервальные и категориальные переменные

2. Проведена предобработка данных:
   - Обработаны пропуски (заполнение медианой/модой)
   - Обработаны выбросы (метод IQR)
   - Закодированы категориальные переменные

3. Построены три модели классификации:
   - K-Nearest Neighbors (KNN)
   - Логистическая регрессия
   - Support Vector Machine (SVM)

4. Оценка качества показала:
   - Лучшая модель определяется по метрике F1-Score
   - Для полной оценки использовались Accuracy, Precision, Recall, F1, ROC-AUC
   - Confusion matrix для каждой модели

Рекомендации по улучшению:
1. Для реальных данных необходимо загрузить обучающий датасет с целевой переменной
2. Провести feature engineering
3. Настроить гиперпараметры моделей с помощью GridSearchCV
4. Использовать кросс-валидацию для более надежной оценки
""")