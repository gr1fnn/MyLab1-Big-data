import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

class MultivariateAnalysis:
    """Класс для многомерного анализа"""
    
    def __init__(self, dataframe, all_dataframes=None):
        self.df = dataframe
        self.all_dataframes = all_dataframes or {}
    
    def plot_multivariate(self, graph_type):
        """Построение многомерных графиков"""
        if self.df is None:
            return "❌ Сначала объедините данные", None
        
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig = None
            
            if graph_type == "Доход-Возраст-Пол":
                fig, report = self._plot_income_age_gender()
            elif graph_type == "Автомобили-Доход-Пол":
                fig, report = self._plot_cars_income_gender()
            elif graph_type == "Преступления по городам":
                fig, report = self._plot_crimes_by_city()
            elif graph_type == "Корреляционная матрица":
                fig, report = self._plot_correlation_matrix()
            else:
                report = f"❌ Неизвестный тип графика: {graph_type}"
            
            if fig:
                plt.tight_layout()
            
            return report, fig
            
        except Exception as e:
            import traceback
            return f"❌ Ошибка построения графика: {e}\n\n{traceback.format_exc()}", None
    
    def _plot_income_age_gender(self):
        """График: Доход-Возраст-Пол"""
        if not all(col in self.df.columns for col in ['age', 'annual_income', 'gender']):
            return None, "❌ Отсутствуют необходимые колонки: age, annual_income, gender"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_data = self.df[['age', 'annual_income', 'gender']].dropna()
        
        colors = {'male': 'blue', 'female': 'red'}
        markers = {'male': 'o', 'female': '^'}
        
        for gender, color in colors.items():
            gender_data = plot_data[plot_data['gender'] == gender]
            if len(gender_data) > 0:
                ax.scatter(gender_data['age'], gender_data['annual_income'],
                         alpha=0.6, label=gender, color=color, s=50,
                         marker=markers[gender])
        
        ax.set_xlabel('Возраст (лет)', fontsize=12)
        ax.set_ylabel('Годовой доход ($)', fontsize=12)
        ax.set_title('Зависимость дохода от возраста и пола', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
        
        report = self._create_income_age_gender_report(plot_data)
        return fig, report
    
    def _plot_cars_income_gender(self):
        """График: Автомобили-Доход-Пол"""
        if not all(col in self.df.columns for col in ['car_make', 'annual_income', 'gender']):
            return None, "❌ Отсутствуют необходимые колонки: car_make, annual_income, gender"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        car_data = self.df[['car_make', 'annual_income', 'gender']].dropna()
        top_cars = car_data['car_make'].value_counts().head(8).index
        car_data = car_data[car_data['car_make'].isin(top_cars)]
        
        pivot_data = car_data.pivot_table(
            values='annual_income',
            index='car_make',
            columns='gender',
            aggfunc='mean'
        ).fillna(0)
        
        x = np.arange(len(pivot_data.index))
        width = 0.35
        
        if 'male' in pivot_data.columns:
            ax.bar(x - width/2, pivot_data['male'], width, label='Мужчины', alpha=0.8)
        if 'female' in pivot_data.columns:
            ax.bar(x + width/2, pivot_data['female'], width, label='Женщины', alpha=0.8)
        
        ax.set_xlabel('Марка автомобиля', fontsize=12)
        ax.set_ylabel('Средний доход ($)', fontsize=12)
        ax.set_title('Средний доход по маркам автомобилей и полу', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
        
        report = self._create_cars_income_gender_report(car_data)
        return fig, report
    
    def _plot_crimes_by_city(self):
        """График: Преступления по городам"""
        if 'crime_scene_report' not in self.all_dataframes:
            return None, "❌ Отсутствует таблица crime_scene_report"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        crime_df = self.all_dataframes['crime_scene_report']
        city_crimes = crime_df['city'].value_counts().head(10)
        
        ax.bar(city_crimes.index, city_crimes.values, color='coral', alpha=0.7)
        ax.set_xlabel('Город', fontsize=12)
        ax.set_ylabel('Количество преступлений', fontsize=12)
        ax.set_title('Распределение преступлений по городам', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        report = self._create_crimes_report(crime_df)
        return fig, report
    
    def _plot_correlation_matrix(self):
        """Корреляционная матрица"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return None, "❌ Недостаточно числовых данных для корреляционной матрицы"
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = self.df[numeric_cols].corr()
        im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
        
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                value = corr_matrix.iloc[i, j]
                color = "white" if abs(value) > 0.5 else "black"
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=color, fontweight='bold')
        
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_yticklabels(corr_matrix.index)
        ax.set_title('Корреляционная матрица числовых признаков', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Коэффициент корреляции')
        
        report = self._create_correlation_report(corr_matrix)
        return fig, report
    
    def _create_income_age_gender_report(self, plot_data):
        """Отчет для графика Доход-Возраст-Пол"""
        report = "📈 МНОГОМЕРНЫЙ АНАЛИЗ: Доход-Возраст-Пол\n\n"
        report += "АНАЛИЗ ЗАВИСИМОСТИ ДОХОДА ОТ ВОЗРАСТА И ПОЛА:\n\n"
        
        for gender in ['male', 'female']:
            gender_data = plot_data[plot_data['gender'] == gender]
            if len(gender_data) > 0:
                report += f"{'Мужчины' if gender == 'male' else 'Женщины'}:\n"
                report += f"  • Количество: {len(gender_data):,}\n"
                report += f"  • Средний возраст: {gender_data['age'].mean():.1f} лет\n"
                report += f"  • Средний доход: ${gender_data['annual_income'].mean():,.0f}\n"
                report += f"  • Медианный доход: ${gender_data['annual_income'].median():,.0f}\n\n"
        
        male_corr = plot_data[plot_data['gender'] == 'male'][['age', 'annual_income']].corr().iloc[0, 1]
        female_corr = plot_data[plot_data['gender'] == 'female'][['age', 'annual_income']].corr().iloc[0, 1]
        
        report += "КОРРЕЛЯЦИЯ ВОЗРАСТ-ДОХОД:\n"
        report += f"  • Мужчины: {male_corr:.3f}\n"
        report += f"  • Женщины: {female_corr:.3f}\n\n"
        
        report += "ВЫВОДЫ:\n"
        report += "1. Наличие положительной/отрицательной корреляции между возрастом и доходом\n"
        report += "2. Гендерные различия в уровне доходов\n"
        report += "3. Возрастные пики доходов для разных групп\n"
        
        return report
    
    # Остальные методы _create_*_report остаются аналогичными
    # ... (продолжение следует в полной версии)