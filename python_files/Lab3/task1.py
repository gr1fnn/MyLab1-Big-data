import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve)
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """Класс для разведочного анализа данных"""
    
    def __init__(self, df):
        self.df = df
    
    def analyze_data(self):
        """Провести разведочный анализ данных"""
        
        # A. Количество строк и столбцов
        print(f"A. Размер датафрейма: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
        
        # B. Объем памяти
        print(f"B. Объем памяти: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # C. Статистика для интервальных переменных
        print("\nC. Статистика для интервальных переменных:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats_data = {
                'min': self.df[numeric_cols].min(),
                'median': self.df[numeric_cols].median(),
                'mean': self.df[numeric_cols].mean(),
                'max': self.df[numeric_cols].max(),
                '25%': self.df[numeric_cols].quantile(0.25),
                '75%': self.df[numeric_cols].quantile(0.75)
            }
            numeric_stats = pd.DataFrame(stats_data).T
            print(numeric_stats)
        else:
            print("Нет интервальных переменных")
        
        # D. Мода для категориальных переменных
        print("\nD. Мода для категориальных переменных:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else None
            mode_count = (self.df[col] == mode_val).sum() if mode_val else 0
            print(f"  {col}: мода = '{mode_val}', встречается {mode_count} раз")


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, df):
        self.df = df
        self.label_encoders = {}
    
    def handle_missing_values(self):
        """Обработать пропущенные значения"""
        print("\nA. Обработка пропусков:")
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        
        if not missing_cols:
            print("  Пропусков не обнаружено")
            return
        
        print(f"  Пропуски найдены в колонках: {missing_cols}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Для числовых колонок - заполняем медианой
        for col in missing_cols:
            if col in numeric_cols:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  Заполнены пропуски в '{col}' медианой: {median_val:.2f}")
            elif col in categorical_cols:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                print(f"  Заполнены пропуски в '{col}' модой: '{mode_val}'")
    
    def handle_outliers(self):
        """Обработать выбросы методом IQR"""
        print("\nB. Обработка выбросов:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers.empty:
                print(f"  В '{col}' найдено {len(outliers)} выбросов")
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                print(f"  Выбросы заменены граничными значениями")
    
    def encode_categorical_variables(self):
        """Закодировать категориальные переменные"""
        print("\nC. Обработка категориальных переменных:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            print("  Категориальных переменных не найдено")
            return
        
        print(f"  Найдено {len(categorical_cols)} категориальных переменных")
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  Закодирована переменная '{col}' (Label Encoding)")
    
    def split_data(self, target_col=None, test_size=0.2):
        """Разделить данные на трейн и тест"""
        print("\nD. Разделение датасета на трейн и тест:")
        
        # Если целевая переменная не указана, ищем её
        if target_col is None:
            target_candidates = ['target', 'Target', 'class', 'Class', 'y']
            for candidate in target_candidates:
                if candidate in self.df.columns:
                    target_col = candidate
                    break
            
            if target_col is None:
                # Берём последний столбец, если он имеет бинарные значения
                last_col = self.df.columns[-1]
                if self.df[last_col].nunique() <= 2:
                    target_col = last_col
                else:
                    raise ValueError("Не удалось определить целевую переменную")
        
        print(f"  Целевая переменная: '{target_col}'")
        
        X = self.df.drop(target_col, axis=1)
        y = self.df[target_col]
        
        # Проверяем, что задача бинарной классификации
        unique_classes = y.nunique()
        if unique_classes != 2:
            print(f"  Внимание: найдено {unique_classes} классов, требуется бинарная классификация")
            print("  Будет выполнена бинарная классификация (первые два класса)")
            y = y.apply(lambda x: 0 if x == y.unique()[0] else 1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"  Размер тренировочной выборки: {X_train.shape}")
        print(f"  Размер тестовой выборки: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test


class ClassificationModels:
    """Класс для построения моделей классификации"""
    
    def __init__(self):
        self.models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        self.results = {}
    
    def train_models(self, X_train, y_train, X_test):
        """Обучить модели"""
        
        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'X_test_scaled': X_test_scaled
            }
        
        return scaler
    
    def evaluate_models(self, y_test):
        """Оценить качество моделей"""
        
        metrics_data = []
        
        for name in self.models.keys():
            y_pred = self.results[name]['y_pred']
            y_pred_proba = self.results[name]['y_pred_proba']
            
            # Рассчитываем метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC-AUC
            roc_auc = None
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            metrics_data.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc if roc_auc is not None else 'N/A'
            })
            
            # Выводим метрики для каждой модели
            print(f"\n{name}:")
            print(f"  Accuracy (A): {accuracy:.4f}")
            print(f"  Precision (P): {precision:.4f}")
            print(f"  Recall (R): {recall:.4f}")
            print(f"  F1-Score (E): {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else "  ROC-AUC: N/A")
            print(f"  Confusion Matrix:\n{cm}")
        
        # Создаем DataFrame с метриками
        metrics_df = pd.DataFrame(metrics_data)
        
        # Выбираем лучшую модель по F1-Score
        if not metrics_df['F1-Score'].isin(['N/A']).all():
            best_idx = metrics_df['F1-Score'].astype(float).idxmax()
            best_model = metrics_df.loc[best_idx, 'Model']
            best_f1 = metrics_df.loc[best_idx, 'F1-Score']
        else:
            best_idx = metrics_df['Accuracy'].astype(float).idxmax()
            best_model = metrics_df.loc[best_idx, 'Model']
            best_f1 = metrics_df.loc[best_idx, 'Accuracy']
        
        print("\n" + "="*50)
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*50)
        print(metrics_df.to_string(index=False))
        
        print(f"\nСамый оптимальный алгоритм: {best_model} (F1-Score = {best_f1:.4f})")
        
        return metrics_df, best_model
    
    def plot_roc_curves(self, y_test):
        """Построить ROC-кривые"""
        print("\n" + "="*50)
        print("ROC КРИВЫЕ")
        print("="*50)
        
        plt.figure(figsize=(10, 8))
        
        for name in self.models.keys():
            y_pred_proba = self.results[name]['y_pred_proba']
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые моделей классификации')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    """Основная функция"""
    
    # 1. Загрузка данных
    file_path = r"C:\Users\grish\Desktop\учеба\8 семестр\Крутских Елена Игоревна\MyLab1-Big-data\python_files\Lab3\playground-series-s3e12\train.csv"
    
    try:
        df = pd.read_csv(file_path)
        print(f"Данные успешно загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    # 2. Разведочный анализ данных
    analyzer = DataAnalyzer(df)
    analyzer.analyze_data()
    
    preprocessor = DataPreprocessor(df.copy())
    
    # A. Обработка пропусков
    preprocessor.handle_missing_values()
    
    # B. Обработка выбросов
    preprocessor.handle_outliers()
    
    # C. Обработка категориальных переменных
    preprocessor.encode_categorical_variables()
    
    # D. Разделение на трейн и тест
    try:
        X_train, X_test, y_train, y_test = preprocessor.split_data()
    except ValueError as e:
        print(f"Ошибка: {e}")
        return
    
    # 4. Построение моделей классификации
    model_builder = ClassificationModels()
    scaler = model_builder.train_models(X_train, y_train, X_test)
    
    # 5. Оценка качества алгоритмов
    metrics_df, best_model = model_builder.evaluate_models(y_test)
    
    # Визуализация ROC-кривых
    model_builder.plot_roc_curves(y_test)

if __name__ == "__main__":
    main()