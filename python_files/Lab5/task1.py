import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import umap
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Для Windows - отключаем многопроцессорность в joblib
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. ЗАГРУЗКА ДАННЫХ
print("ЛАБОРАТОРНАЯ РАБОТА: СНИЖЕНИЕ РАЗМЕРНОСТИ И КЛАСТЕРИЗАЦИЯ")

# Получаем путь к директории, где находится текущий скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'WineQT.csv')

# Загрузка данных
df = pd.read_csv(data_path)

print(f"\n1. Загружены данные. Размерность: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(f"\nПервые 5 строк данных:")
print(df.head())

# 2. EDA (Exploratory Data Analysis)
print("2. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")

# Информация о данных
print("\nИнформация о данных:")
print(df.info())

# Статистическое описание
print("\nСтатистическое описание:")
print(df.describe())

# Проверка пропущенных значений
print("\nПропущенные значения:")
print(df.isnull().sum())

# Проверка уникальных значений в целевой переменной
print(f"\nУникальные значения качества (target): {df['quality'].unique()}")
print(f"Распределение качества:\n{df['quality'].value_counts().sort_index()}")

# Визуализация распределения признаков
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
features = df.columns[:-2]  # Исключаем quality и Id

for i, feature in enumerate(features[:11]):
    row = i // 4
    col = i % 4
    axes[row, col].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col].set_title(f'Распределение {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Частота')

plt.tight_layout()
plt.show()

# Корреляционная матрица
plt.figure(figsize=(12, 10))
correlation_matrix = df[features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5)
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.show()

# Обработка выбросов с помощью IQR
print("\nОбработка выбросов методом IQR:")
features_for_outliers = df[features].columns
outliers_count = {}

# Сохраняем исходные данные для сравнения
df_before = df.copy()
outliers_count = {}

for feature in features_for_outliers:
    # Расчет границ для выбросов
    Q1 = df_before[feature].quantile(0.25)
    Q3 = df_before[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Подсчет выбросов
    outliers_before = df_before[(df_before[feature] < lower_bound) | (df_before[feature] > upper_bound)]
    outliers_count[feature] = len(outliers_before)
    
    # Применяем клиппинг
    df[feature] = df[feature].clip(lower_bound, upper_bound)
    
# Вывод итоговой статистики
print("\nОбработка выбросов завершена:")
print(f"Всего признаков обработано: {len(features_for_outliers)}")
print(f"Всего выбросов найдено: {sum(outliers_count.values())}")

# Нормализация данных
print("\nНормализация данных (StandardScaler):")
# Отделяем признаки от целевой переменной и Id
X = df[features].values
y = df['quality'].values
ids = df['Id'].values if 'Id' in df.columns else None

# Нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Среднее после нормализации: {X_scaled.mean(axis=0).round(2)}")
print(f"Стандартное отклонение после нормализации: {X_scaled.std(axis=0).round(2)}")

# Сохраняем нормализованные данные для дальнейшего использования
df_scaled = pd.DataFrame(X_scaled, columns=features)
df_scaled['quality'] = y
if ids is not None:
    df_scaled['Id'] = ids

# 3. KERNEL PCA
print("3. KERNEL PCA (с разными ядерными функциями)")

kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
n_components = 2 

kpca_results = {}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, kernel in enumerate(kernels):
    print(f"\nПрименение KernelPCA с ядром '{kernel}'...")
    
    # Создание и применение KernelPCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, 
                     fit_inverse_transform=False, random_state=42)
    X_kpca = kpca.fit_transform(X_scaled)
    kpca_results[kernel] = X_kpca
    
    # Визуализация
    scatter = axes[i].scatter(X_kpca[:, 0], X_kpca[:, 1], 
                              c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[i].set_title(f'KernelPCA ({kernel})')
    axes[i].set_xlabel('Компонента 1')
    axes[i].set_ylabel('Компонента 2')
    axes[i].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[i], label='Quality')

# Убираем пустой подграфик
if len(kernels) < len(axes):
    axes[-1].set_visible(False)

plt.suptitle('KernelPCA с различными ядерными функциями', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Анализ дисперсии для линейного ядра
print("4. АНАЛИЗ ДИСПЕРСИИ ДЛЯ ЛИНЕЙНОГО ЯДРА")

# Получаем собственные значения для линейного ядра
kpca_linear = KernelPCA(n_components=len(features), kernel='linear', 
                        fit_inverse_transform=False, random_state=42)
X_kpca_linear = kpca_linear.fit_transform(X_scaled)

# Для линейного ядра можно получить собственные значения
if hasattr(kpca_linear, 'eigenvalues_') and kpca_linear.eigenvalues_ is not None:
    explained_variance = kpca_linear.eigenvalues_ / np.sum(kpca_linear.eigenvalues_)
    cumulative_variance = np.cumsum(explained_variance)
    
    print("Дисперсия, объясненная каждой компонентой:")
    for i, (ev, cum) in enumerate(zip(explained_variance[:10], cumulative_variance[:10])):
        print(f"Компонента {i+1}: {ev:.4f} (кумулятивная: {cum:.4f})")
    
    # Визуализация explained variance
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance[:10])+1), explained_variance[:10], alpha=0.7)
    plt.xlabel('Главная компонента')
    plt.ylabel('Доля объясненной дисперсии')
    plt.title('Объясненная дисперсия по компонентам')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-', markersize=4)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% дисперсии')
    plt.xlabel('Число компонент')
    plt.ylabel('Кумулятивная объясненная дисперсия')
    plt.title('Кумулятивная объясненная дисперсия')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Потеря дисперсии (lost variance) при выборе 2 компонент
    lost_variance = 1 - cumulative_variance[1]  # для 2 компонент
    print(f"\nПотеря дисперсии при использовании 2 компонент: {lost_variance:.4f} ({lost_variance*100:.2f}%)")
    
    # Определяем оптимальное число компонент для сохранения 95% дисперсии
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Оптимальное число компонент для сохранения 95% дисперсии: {n_components_95}")
else:
    print("Собственные значения недоступны для данной реализации KernelPCA")

# 5. T-SNE И UMAP
print("5. T-SNE И UMAP")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# t-SNE - используем n_jobs=1 для Windows
print("\nПрименение t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
            max_iter=1000, n_jobs=1)  # n_jobs=1 для Windows
X_tsne = tsne.fit_transform(X_scaled)

axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_title('t-SNE')
axes[0].set_xlabel('Компонента 1')
axes[0].set_ylabel('Компонента 2')
axes[0].grid(True, alpha=0.3)

# UMAP - отключаем использование нескольких потоков
print("Применение UMAP...")
try:
    # Используем UMAP с параметрами для Windows
    reducer = umap.UMAP(n_components=2, random_state=42, 
                        n_neighbors=15, min_dist=0.1,
                        n_jobs=1)  # n_jobs=1 для Windows
    X_umap = reducer.fit_transform(X_scaled)
    
    axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', 
                    alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_title('UMAP')
    axes[1].set_xlabel('Компонента 1')
    axes[1].set_ylabel('Компонента 2')
    axes[1].grid(True, alpha=0.3)
except Exception as e:
    print(f"Ошибка UMAP: {e}")
    print("Попытка с параметрами по умолчанию...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    X_umap = reducer.fit_transform(X_scaled)
    axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', 
                    alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_title('UMAP')
    axes[1].set_xlabel('Компонента 1')
    axes[1].set_ylabel('Компонента 2')
    axes[1].grid(True, alpha=0.3)

# Сравнение с лучшим KernelPCA (например, rbf)
axes[2].scatter(kpca_results['rbf'][:, 0], kpca_results['rbf'][:, 1], 
                c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
axes[2].set_title('KernelPCA (rbf) для сравнения')
axes[2].set_xlabel('Компонента 1')
axes[2].set_ylabel('Компонента 2')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Сравнение методов снижения размерности', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Сохраняем лучшую модель (например, UMAP)
model_path = 'umap_model.joblib'
joblib.dump(reducer, model_path)
print(f"Модель UMAP сохранена в {model_path}")

# Загружаем модель обратно
loaded_reducer = joblib.load(model_path)

# Проверка загруженной модели
X_umap_loaded = loaded_reducer.transform(X_scaled)
print(f"Размерность данных после применения загруженной модели: {X_umap_loaded.shape}")

# Определяем оптимальное число кластеров
print("\nОпределение оптимального числа кластеров:")

# Используем метод локтя для k-means
inertias = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    # Silhouette score
    labels = kmeans.predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Визуализация для выбора k
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(K_range, inertias, 'bo-', markersize=8)
axes[0].set_xlabel('Количество кластеров (k)')
axes[0].set_ylabel('Инерция')
axes[0].set_title('Метод локтя для k-means')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-', markersize=8)
axes[1].set_xlabel('Количество кластеров (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Оценка силуэта')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Выбираем оптимальное число кластеров (например, 3)
optimal_k = 3
print(f"\nВыбрано оптимальное число кластеров: {optimal_k}")

# K-means кластеризация
print(f"\nПрименение K-means с {optimal_k} кластерами...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Иерархическая кластеризация
print(f"Применение иерархической кластеризации с {optimal_k} кластерами...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Визуализация результатов кластеризации на проекции UMAP
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Исходные классы (quality)
axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', 
                alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_title('Исходные классы (качество вина)')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].grid(True, alpha=0.3)

# K-means кластеры
axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=kmeans_labels, cmap='plasma', 
                alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1].set_title(f'K-means кластеризация (k={optimal_k})')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')
axes[1].grid(True, alpha=0.3)

# Иерархическая кластеризация
axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=hierarchical_labels, cmap='plasma', 
                alpha=0.7, edgecolors='black', linewidth=0.5)
axes[2].set_title(f'Иерархическая кластеризация (k={optimal_k})')
axes[2].set_xlabel('UMAP1')
axes[2].set_ylabel('UMAP2')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Оценка качества кластеризации
print("\nОценка качества кластеризации:")

# Silhouette score для k-means
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Score (K-means): {silhouette_kmeans:.4f}")

# Silhouette score для иерархической
silhouette_hier = silhouette_score(X_scaled, hierarchical_labels)
print(f"Silhouette Score (Hierarchical): {silhouette_hier:.4f}")

# Сравнение с исходными классами
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_kmeans = adjusted_rand_score(y, kmeans_labels)
nmi_kmeans = normalized_mutual_info_score(y, kmeans_labels)

ari_hier = adjusted_rand_score(y, hierarchical_labels)
nmi_hier = normalized_mutual_info_score(y, hierarchical_labels)

print(f"\nСравнение с исходными классами:")
print(f"K-means - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
print(f"Hierarchical - ARI: {ari_hier:.4f}, NMI: {nmi_hier:.4f}")


