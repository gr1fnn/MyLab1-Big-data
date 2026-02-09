import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve)
import warnings
warnings.filterwarnings('ignore')


class MyKNNClassifier:
    """Собственная реализация K-Nearest Neighbors"""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            unique, counts = np.unique(k_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.array(predictions)
    
    def predict_proba(self, X):
        probas = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            prob_0 = np.sum(k_labels == 0) / self.n_neighbors
            prob_1 = np.sum(k_labels == 1) / self.n_neighbors
            probas.append([prob_0, prob_1])
        return np.array(probas)


class DataAnalyzer:
    """Класс для разведочного анализа данных"""
    
    def __init__(self, df):
        self.df = df
    
    def analyze_data(self):
        print(f"A. Размер датафрейма: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
        print(f"B. Объем памяти: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nC. Статистика для интервальных переменных:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats = pd.DataFrame({
                'min': self.df[numeric_cols].min(),
                'median': self.df[numeric_cols].median(),
                'mean': self.df[numeric_cols].mean(),
                'max': self.df[numeric_cols].max(),
                '25%': self.df[numeric_cols].quantile(0.25),
                '75%': self.df[numeric_cols].quantile(0.75)
            }).T
            print(stats)
        
        print("\nD. Мода для категориальных переменных:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                mode_count = (self.df[col] == mode_val).sum() if mode_val else 0
                print(f"  {col}: мода = '{mode_val}', встречается {mode_count} раз")
        else:
            print("  Категориальных переменных нет")


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, df):
        self.df = df
        self.label_encoders = {}
    
    def handle_missing_values(self):
        print("\nA. Анализ и обработка пропусков:")
        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            print("  Пропусков не обнаружено")
            return
        
        print(f"  Общее количество пропусков: {total_missing}")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  '{col}': {count} пропусков ({count/len(self.df)*100:.1f}%)")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in missing_counts[missing_counts > 0].index:
            if col in numeric_cols:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  Заполнены пропуски в '{col}' медианой: {median_val:.2f}")
            elif col in categorical_cols:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                print(f"  Заполнены пропуски в '{col}' модой: '{mode_val}'")
    
    def handle_outliers(self):
        print("\nB. Анализ и обработка выбросов:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        total_outliers = 0
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            total_outliers += outlier_count
            
            if outlier_count > 0:
                print(f"  '{col}': {outlier_count} выбросов ({outlier_count/len(self.df)*100:.1f}%)")
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        if total_outliers == 0:
            print("  Выбросов не обнаружено")
        else:
            print(f"  Всего найдено и обработано {total_outliers} выбросов")
    
    def encode_categorical_variables(self):
        print("\nC. Анализ и обработка категориальных переменных:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            print("  Категориальных переменных не найдено")
            return
        
        print(f"  Найдено {len(categorical_cols)} категориальных переменных:")
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            unique_count = len(self.df[col].unique())
            print(f"  Закодирована '{col}' (Label Encoding, {unique_count} уникальных значений)")
    
    def normalize_data(self, X_train, X_test):
        """Нормализация данных методом min-max"""
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        
        for i in range(X_train.shape[1]):
            min_val = X_train[:, i].min()
            max_val = X_train[:, i].max()
            if max_val - min_val > 0:
                X_train_norm[:, i] = (X_train[:, i] - min_val) / (max_val - min_val)
                X_test_norm[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)
        
        return X_train_norm, X_test_norm
    
    def split_data(self):
        print("\nD. Разделение датасета на трейн и тест:")
        
        if 'quality' not in self.df.columns:
            raise ValueError("Колонка 'quality' не найдена")
        
        median_quality = self.df['quality'].median()
        self.df['target'] = (self.df['quality'] >= median_quality).astype(int)
        
        X = self.df.drop(['target', 'quality', 'Id'], axis=1, errors='ignore').values
        y = self.df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_norm, X_test_norm = self.normalize_data(X_train, X_test)
        
        print(f"  Размер тренировочной выборки: {X_train_norm.shape}")
        print(f"  Размер тестовой выборки: {X_test_norm.shape}")
        print(f"  Распределение классов в тестовой выборке: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")
        
        return X_train_norm, X_test_norm, y_train, y_test


class ClassificationModels:
    """Класс для построения моделей классификации"""
    
    def __init__(self):
        self.models = {
            "KNN": MyKNNClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        self.results = {}
    
    def train_models(self, X_train, y_train, X_test):
        print("\nПостроение моделей классификации:")
        for name, model in self.models.items():
            print(f"  Обучение {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
    
    def evaluate_models(self, y_test):
        print("\nОценка качества алгоритмов:")
        metrics_data = []
        
        for name in self.models.keys():
            y_pred = self.results[name]['y_pred']
            y_pred_proba = self.results[name]['y_pred_proba']
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            cm = confusion_matrix(y_test, y_pred)
            
            metrics_data.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc if roc_auc is not None else 'N/A',
                'Confusion_Matrix': cm
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        print("\nМетрики качества моделей:")
        for _, row in metrics_df.iterrows():
            print(f"\n{row['Model']}:")
            print(f"  Accuracy (A): {row['Accuracy']:.4f}")
            print(f"  Precision (P): {row['Precision']:.4f}")
            print(f"  Recall (R): {row['Recall']:.4f}")
            print(f"  F1-Score (E): {row['F1-Score']:.4f}")
            if row['ROC-AUC'] != 'N/A':
                print(f"  ROC-AUC: {row['ROC-AUC']:.4f}")
            else:
                print(f"  ROC-AUC: N/A")
            print(f"  Confusion Matrix:\n{row['Confusion_Matrix']}")
        
        if not metrics_df['F1-Score'].isin(['N/A']).all():
            best_idx = metrics_df['F1-Score'].astype(float).idxmax()
            best_model = metrics_df.loc[best_idx, 'Model']
            best_f1 = metrics_df.loc[best_idx, 'F1-Score']
        else:
            best_idx = metrics_df['Accuracy'].astype(float).idxmax()
            best_model = metrics_df.loc[best_idx, 'Model']
            best_f1 = metrics_df.loc[best_idx, 'Accuracy']
        
        print(f"\nСамый оптимальный алгоритм: {best_model} (F1-Score = {best_f1:.4f})")
        
        return metrics_df, best_model
    
    def plot_roc_curves(self, y_test):
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
        
        for name in self.models.keys():
            y_pred_proba = self.results[name]['y_pred_proba']
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    file_path = r"C:\Users\grish\Desktop\учеба\8 семестр\Крутских Елена Игоревна\MyLab1-Big-data\python_files\Lab3\WineQT.csv"
    
    try:
        df = pd.read_csv(file_path)
        print("1. Загружен набор данных:")
        print(f"   - Файл: {file_path}")
        print(f"   - Строк: {df.shape[0]}, столбцов: {df.shape[1]}")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    print("\n2. Разведочный анализ данных:")
    analyzer = DataAnalyzer(df)
    analyzer.analyze_data()
    
    print("\n3. Подготовка датасета:")
    preprocessor = DataPreprocessor(df.copy())
    
    preprocessor.handle_missing_values()
    preprocessor.handle_outliers()
    preprocessor.encode_categorical_variables()
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.split_data()
    except ValueError as e:
        print(f"Ошибка: {e}")
        return
    
    print("\n4. Построение классификационных алгоритмов:")
    model_builder = ClassificationModels()
    model_builder.train_models(X_train, y_train, X_test)
    
    print("\n5. Оценка качества алгоритмов:")
    metrics_df, best_model = model_builder.evaluate_models(y_test)
    
    print("\n6. Визуализация ROC-кривых:")
    model_builder.plot_roc_curves(y_test)


if __name__ == "__main__":
    main()