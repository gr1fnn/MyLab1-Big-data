import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class MPGAnalysis:
    """Класс для анализа данных mpg (общая часть)"""
    
    def __init__(self):
        self.mpg_df = None
        self.mpg_encoded_df = None
    
    def load_and_analyze_mpg(self):
        """Загрузка и анализ набора данных mpg из seaborn"""
        try:
            self.mpg_df = sns.load_dataset('mpg')
            
            report = "📊 ОБЩАЯ ЧАСТЬ: АНАЛИЗ ДАННЫХ MPG\n"
            report += "="*60 + "\n\n"
            
            # Основная информация
            rows, cols = self.mpg_df.shape
            report += "1. ОСНОВНАЯ ИНФОРМАЦИЯ:\n"
            report += f"   • Количество строк: {rows}\n"
            report += f"   • Количество столбцов: {cols}\n"
            report += f"   • Столбцы: {', '.join(self.mpg_df.columns.tolist())}\n\n"
            
            # Разведочный анализ
            report += self._exploratory_analysis()
            
            # Кодирование категориальных переменных
            report += self._encode_categorical_variables()
            
            report += "\n✅ Данные mpg успешно загружены и проанализированы!\n"
            
            return report
            
        except Exception as e:
            import traceback
            return f"❌ Ошибка анализа mpg: {e}\n\n{traceback.format_exc()}"
    
    def _exploratory_analysis(self):
        """Разведочный анализ данных mpg"""
        report = "2. РАЗВЕДОЧНЫЙ АНАЛИЗ:\n"
        
        # Числовые переменные
        numeric_cols = self.mpg_df.select_dtypes(include=[np.number]).columns
        report += "   ЧИСЛОВЫЕ ПЕРЕМЕННЫЕ:\n"
        for col in numeric_cols:
            data = self.mpg_df[col].dropna()
            report += f"   • {col}:\n"
            report += f"      - Доля пропусков: {(self.mpg_df[col].isna().sum() / len(self.mpg_df)):.2%}\n"
            report += f"      - Минимум: {data.min():.2f}\n"
            report += f"      - Максимум: {data.max():.2f}\n"
            report += f"      - Среднее: {data.mean():.2f}\n"
            report += f"      - Медиана: {data.median():.2f}\n"
            report += f"      - Дисперсия: {data.var():.2f}\n"
            report += f"      - Q0.1: {data.quantile(0.1):.2f}\n"
            report += f"      - Q0.9: {data.quantile(0.9):.2f}\n"
            report += f"      - Q1: {data.quantile(0.25):.2f}\n"
            report += f"      - Q3: {data.quantile(0.75):.2f}\n"
        
        # Категориальные переменные
        categorical_cols = self.mpg_df.select_dtypes(include=['object', 'category']).columns
        report += "\n   КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ:\n"
        for col in categorical_cols:
            data = self.mpg_df[col].dropna()
            report += f"   • {col}:\n"
            report += f"      - Доля пропусков: {(self.mpg_df[col].isna().sum() / len(self.mpg_df)):.2%}\n"
            report += f"      - Уникальных значений: {data.nunique()}\n"
            report += f"      - Мода: {data.mode().iloc[0] if len(data.mode()) > 0 else 'N/A'}\n"
        
        return report + "\n"
    
    def _encode_categorical_variables(self):
        """Кодирование категориальных переменных"""
        report = "3. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ:\n"
        
        encoded_df = self.mpg_df.copy()
        categorical_cols = self.mpg_df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in encoded_df.columns:
                # Label Encoding для переменных с небольшим количеством уникальных значений
                if encoded_df[col].nunique() <= 10:
                    le = LabelEncoder()
                    encoded_df[f'{col}_encoded'] = le.fit_transform(encoded_df[col].fillna('Unknown'))
                    report += f"   • {col}: применен LabelEncoding ({encoded_df[col].nunique()} уникальных значений)\n"
                else:
                    # OneHotEncoding для переменных со многими уникальными значениями
                    dummies = pd.get_dummies(encoded_df[col], prefix=col, drop_first=True)
                    encoded_df = pd.concat([encoded_df, dummies], axis=1)
                    report += f"   • {col}: применен OneHotEncoding (создано {dummies.shape[1]} новых столбцов)\n"
        
        self.mpg_encoded_df = encoded_df
        return report
    
    def test_hypotheses_mpg(self):
        """Проверка статистических гипотез для данных mpg"""
        if self.mpg_df is None:
            return "❌ Сначала загрузите данные mpg"
        
        try:
            report = "🔬 ПРОВЕРКА СТАТИСТИЧЕСКИХ ГИПОТЕЗ (MPG)\n"
            report += "="*60 + "\n\n"
            
            # Гипотеза 1: Различие в расходе топлива
            report += self._test_hypothesis_1()
            
            # Гипотеза 2: Корреляция мощность-расход
            report += self._test_hypothesis_2()
            
            # Визуализация
            self._visualize_hypotheses()
            
            return report
            
        except Exception as e:
            import traceback
            return f"❌ Ошибка проверки гипотез: {e}\n\n{traceback.format_exc()}"
    
    def _test_hypothesis_1(self):
        """Гипотеза 1: Различие в расходе топлива между американскими и японскими автомобилями"""
        report = "ГИПОТЕЗА 1: Различие в расходе топлива между американскими и японскими автомобилями\n"
        report += "Нулевая гипотеза H0: Средний расход топлива одинаковый\n"
        report += "Альтернативная гипотеза H1: Средний расход топлива разный\n\n"
        
        us_cars = self.mpg_df[self.mpg_df['origin'] == 'usa']['mpg'].dropna()
        japan_cars = self.mpg_df[self.mpg_df['origin'] == 'japan']['mpg'].dropna()
        
        # Проверка на нормальность распределения
        _, us_normality = stats.shapiro(us_cars)
        _, jp_normality = stats.shapiro(japan_cars)
        
        report += f"Проверка нормальности распределения:\n"
        report += f"• Американские автомобили: p-value = {us_normality:.4f} ({'нормальное' if us_normality > 0.05 else 'не нормальное'})\n"
        report += f"• Японские автомобили: p-value = {jp_normality:.4f} ({'нормальное' if jp_normality > 0.05 else 'не нормальное'})\n\n"
        
        if us_normality > 0.05 and jp_normality > 0.05:
            # Если оба распределения нормальные, используем t-тест
            t_stat, p_value = stats.ttest_ind(us_cars, japan_cars, equal_var=False)
            test_name = "Двухвыборочный t-тест (Велча)"
        else:
            # Если хотя бы одно распределение не нормальное, используем U-тест
            u_stat, p_value = stats.mannwhitneyu(us_cars, japan_cars)
            test_name = "U-тест Манна-Уитни"
        
        report += f"Используемый критерий: {test_name}\n"
        report += f"p-value = {p_value:.6f}\n"
        report += f"Статистическая значимость (α=0.05): {'ОТКЛОНЯЕМ H0' if p_value < 0.05 else 'НЕ ОТКЛОНЯЕМ H0'}\n\n"
        
        report += f"Описательная статистика:\n"
        report += f"• Американские автомобили (n={len(us_cars)}): среднее = {us_cars.mean():.2f} ± {us_cars.std():.2f}\n"
        report += f"• Японские автомобили (n={len(japan_cars)}): среднее = {japan_cars.mean():.2f} ± {japan_cars.std():.2f}\n\n"
        
        report += "ВЫВОД: "
        if p_value < 0.05:
            report += "Существует статистически значимое различие в расходе топлива между американскими и японскими автомобилями.\n"
            if us_cars.mean() > japan_cars.mean():
                report += "Американские автомобили в среднем имеют больший расход топлива.\n"
            else:
                report += "Японские автомобили в среднем имеют больший расход топлива.\n"
        else:
            report += "Нет статистически значимых различий в расходе топлива между американскими и японскими автомобилями.\n"
        
        return report + "\n" + "="*60 + "\n\n"
    
    def _test_hypothesis_2(self):
        """Гипотеза 2: Корреляция между мощностью двигателя и расходом топлива"""
        report = "ГИПОТЕЗА 2: Корреляция между мощностью двигателя и расходом топлива\n"
        report += "Нулевая гипотеза H0: Корреляция отсутствует (ρ = 0)\n"
        report += "Альтернативная гипотеза H1: Корреляция существует (ρ ≠ 0)\n\n"
        
        data = self.mpg_df[['horsepower', 'mpg']].dropna()
        
        # Проверка на нормальность
        _, hp_normality = stats.shapiro(data['horsepower'])
        _, mpg_normality = stats.shapiro(data['mpg'])
        
        if hp_normality > 0.05 and mpg_normality > 0.05:
            # Если оба распределения нормальные, используем корреляцию Пирсона
            corr, p_value = stats.pearsonr(data['horsepower'], data['mpg'])
            corr_name = "Пирсона"
        else:
            # Если хотя бы одно распределение не нормальное, используем корреляцию Спирмена
            corr, p_value = stats.spearmanr(data['horsepower'], data['mpg'])
            corr_name = "Спирмена"
        
        report += f"Используемый критерий: корреляция {corr_name}\n"
        report += f"Коэффициент корреляции: {corr:.4f}\n"
        report += f"p-value = {p_value:.6f}\n"
        report += f"Статистическая значимость (α=0.05): {'ОТКЛОНЯЕМ H0' if p_value < 0.05 else 'НЕ ОТКЛОНЯЕМ H0'}\n\n"
        
        report += "Интерпретация силы корреляции:\n"
        if abs(corr) < 0.3:
            report += "• Очень слабая корреляция\n"
        elif abs(corr) < 0.5:
            report += "• Слабая корреляция\n"
        elif abs(corr) < 0.7:
            report += "• Умеренная корреляция\n"
        elif abs(corr) < 0.9:
            report += "• Сильная корреляция\n"
        else:
            report += "• Очень сильная корреляция\n"
        
        report += f"\nЗнак корреляции: {'отрицательный' if corr < 0 else 'положительный'}\n\n"
        
        report += "ВЫВОД: "
        if p_value < 0.05:
            if corr < 0:
                report += "Существует статистически значимая отрицательная корреляция между мощностью двигателя и расходом топлива.\n"
                report += "Чем больше мощность двигателя, тем меньше расход топлива (миль на галлон).\n"
            else:
                report += "Существует статистически значимая положительная корреляция между мощностью двигателя и расходом топлива.\n"
                report += "Чем больше мощность двигателя, тем больше расход топлива (миль на галлон).\n"
        else:
            report += "Нет статистически значимой корреляции между мощностью двигателя и расходом топлива.\n"
        
        return report
    
    def _visualize_hypotheses(self):
        """Визуализация результатов проверки гипотез"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Боксплот для гипотезы 1
        origins = ['usa', 'japan', 'europe']
        mpg_by_origin = [self.mpg_df[self.mpg_df['origin'] == origin]['mpg'].dropna() for origin in origins]
        
        ax1.boxplot(mpg_by_origin, labels=origins)
        ax1.set_xlabel('Страна происхождения')
        ax1.set_ylabel('MPG (миль на галлон)')
        ax1.set_title('Распределение расхода топлива по странам')
        ax1.grid(True, alpha=0.3)
        
        # Диаграмма рассеяния для гипотезы 2
        data = self.mpg_df[['horsepower', 'mpg']].dropna()
        corr, _ = stats.pearsonr(data['horsepower'], data['mpg'])
        
        ax2.scatter(data['horsepower'], data['mpg'], alpha=0.6, color='blue', s=30)
        
        # Линия регрессии
        z = np.polyfit(data['horsepower'], data['mpg'], 1)
        p = np.poly1d(z)
        ax2.plot(data['horsepower'].sort_values(), p(data['horsepower'].sort_values()), "r--", alpha=0.8)
        
        ax2.set_xlabel('Мощность двигателя (лошадиные силы)')
        ax2.set_ylabel('MPG (миль на галлон)')
        ax2.set_title(f'Корреляция мощность-MPG (r={corr:.3f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def gradient_descent_mpg(self):
        """Реализация градиентного спуска для данных mpg"""
        if self.mpg_df is None:
            return "❌ Сначала загрузите данные mpg"
        
        try:
            report = "📉 ГРАДИЕНТНЫЙ СПУСК (MPG)\n"
            report += "="*60 + "\n\n"
            
            # Подготовка данных
            data = self.mpg_df[['horsepower', 'weight', 'mpg']].dropna()
            X_raw = data[['horsepower', 'weight']].values
            y = data['mpg'].values
            
            # Нормализация
            X_mean = X_raw.mean(axis=0)
            X_std = X_raw.std(axis=0)
            X = (X_raw - X_mean) / X_std
            X = np.c_[np.ones(X.shape[0]), X]
            
            # Параметры
            report += "ПАРАМЕТРЫ МОДЕЛИ:\n"
            report += f"• Целевая переменная (y): mpg\n"
            report += f"• Признаки (X): horsepower, weight\n"
            report += f"• Количество наблюдений: {len(y)}\n"
            report += f"• Количество признаков: {X.shape[1] - 1}\n\n"
            
            # Обычный градиентный спуск
            theta = np.zeros(X.shape[1])
            alpha = 0.01
            iterations = 1000
            
            report += "1. ОБЫЧНЫЙ ГРАДИЕНТНЫЙ СПУСК (BATCH):\n"
            theta_batch, cost_history_batch = self._batch_gradient_descent(X, y, theta.copy(), alpha, iterations)
            
            report += f"• Начальная стоимость: {cost_history_batch[0]:.4f}\n"
            report += f"• Финальная стоимость: {cost_history_batch[-1]:.4f}\n"
            report += f"• Уменьшение стоимости: {((cost_history_batch[0] - cost_history_batch[-1]) / cost_history_batch[0]) * 100:.2f}%\n"
            report += f"• Параметры модели: θ0 = {theta_batch[0]:.4f}, θ1 = {theta_batch[1]:.4f}, θ2 = {theta_batch[2]:.4f}\n\n"
            
            # Стохастический градиентный спуск
            report += "2. СТОХАСТИЧЕСКИЙ ГРАДИЕНТНЫЙ СПУСК:\n"
            theta_stochastic, cost_history_stochastic = self._stochastic_gradient_descent(X, y, theta.copy(), alpha, iterations)
            
            report += f"• Начальная стоимость: {cost_history_stochastic[0]:.4f}\n"
            report += f"• Финальная стоимость: {cost_history_stochastic[-1]:.4f}\n"
            report += f"• Уменьшение стоимости: {((cost_history_stochastic[0] - cost_history_stochastic[-1]) / cost_history_stochastic[0]) * 100:.2f}%\n"
            report += f"• Параметры модели: θ0 = {theta_stochastic[0]:.4f}, θ1 = {theta_stochastic[1]:.4f}, θ2 = {theta_stochastic[2]:.4f}\n\n"
            
            # Визуализация
            self._visualize_gradient_descent(cost_history_batch, cost_history_stochastic)
            
            # Сравнение с sklearn
            report += self._compare_with_sklearn(X_raw, y, X_mean, X_std)
            
            return report
            
        except Exception as e:
            import traceback
            return f"❌ Ошибка градиентного спуска: {e}\n\n{traceback.format_exc()}"
    
    def _batch_gradient_descent(self, X, y, theta, alpha, iterations):
        """Обычный градиентный спуск"""
        m = len(y)
        cost_history = []
        
        for i in range(iterations):
            predictions = X.dot(theta)
            errors = predictions - y
            gradient = (1/m) * X.T.dot(errors)
            theta = theta - alpha * gradient
            cost = self._compute_cost(X, y, theta)
            cost_history.append(cost)
        
        return theta, cost_history
    
    def _stochastic_gradient_descent(self, X, y, theta, alpha, iterations):
        """Стохастический градиентный спуск"""
        m = len(y)
        cost_history = []
        
        for i in range(iterations):
            cost = 0
            for j in range(m):
                rand_index = np.random.randint(0, m)
                X_i = X[rand_index:rand_index+1]
                y_i = y[rand_index:rand_index+1]
                
                prediction = X_i.dot(theta)
                error = prediction - y_i
                gradient = X_i.T.dot(error)
                theta = theta - alpha * gradient.flatten()
                
                cost += self._compute_cost(X_i, y_i, theta)
            
            cost_history.append(cost/m)
        
        return theta, cost_history
    
    def _compute_cost(self, X, y, theta):
        """Вычисление функции стоимости (MSE)"""
        m = len(y)
        predictions = X.dot(theta)
        errors = predictions - y
        return (1/(2*m)) * np.sum(errors**2)
    
    def _visualize_gradient_descent(self, cost_history_batch, cost_history_stochastic):
        """Визуализация процесса градиентного спуска"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # График 1: Batch Gradient Descent
        ax1.plot(range(len(cost_history_batch)), cost_history_batch, 'b-', linewidth=2)
        ax1.set_xlabel('Итерации')
        ax1.set_ylabel('Функция стоимости (MSE)')
        ax1.set_title('Обычный градиентный спуск (Batch)')
        ax1.grid(True, alpha=0.3)
        
        # Добавляем логарифмическую шкалу для y
        ax1.set_yscale('log')
        
        # График 2: Stochastic Gradient Descent
        ax2.plot(range(len(cost_history_stochastic)), cost_history_stochastic, 'r-', linewidth=2)
        ax2.set_xlabel('Итерации')
        ax2.set_ylabel('Функция стоимости (MSE)')
        ax2.set_title('Стохастический градиентный спуск')
        ax2.grid(True, alpha=0.3)
        
        # Добавляем логарифмическую шкалу для y
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def _compare_with_sklearn(self, X_raw, y, X_mean, X_std):
        """Сравнение с реализацией sklearn"""
        report = "3. СРАВНЕНИЕ С SCIKIT-LEARN:\n"
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_squared_error, r2_score
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            y_pred = model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            report += f"• Intercept (θ0): {model.intercept_:.4f}\n"
            report += f"• Коэффициенты (θ1, θ2): {model.coef_}\n"
            report += f"• MSE: {mse:.4f}\n"
            report += f"• R²: {r2:.4f}\n\n"
            
            
        except Exception as e:
            report += f"⚠️ Ошибка сравнения с sklearn: {e}\n"
        
        return report