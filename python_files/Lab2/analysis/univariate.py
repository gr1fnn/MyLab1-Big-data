import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

class UnivariateAnalysis:
    """Класс для одномерного анализа"""
    
    def __init__(self, dataframe):
        self.df = dataframe
    
    def plot_histograms(self, feature1, feature2, bins=30):
        """Построение 2 гистограмм"""
        if self.df is None:
            return "❌ Сначала объедините данные", None
        
        try:
            feature_map = {
                "Возраст": "age",
                "Доход": "annual_income", 
                "Рост": "height",
                "Вес": "weight"
            }
            
            col1 = feature_map.get(feature1, feature1)
            col2 = feature_map.get(feature2, feature2)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Гистограмма 1
            report = "📊 ОДНОМЕРНЫЙ АНАЛИЗ (ГИСТОГРАММЫ)\n\n"
            
            if col1 in self.df.columns:
                data1 = self.df[col1].dropna()
                ax1.hist(data1, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
                ax1.set_xlabel(feature1, fontsize=12)
                ax1.set_ylabel('Количество', fontsize=12)
                ax1.set_title(f'Распределение {feature1.lower()}', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                stats1 = f"Среднее: {data1.mean():.1f}\nМедиана: {data1.median():.1f}\nСт. откл.: {data1.std():.1f}"
                ax1.text(0.95, 0.95, stats1, transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                report += self._get_feature_stats(feature1, col1, data1)
            
            # Гистограмма 2
            if col2 in self.df.columns:
                data2 = self.df[col2].dropna()
                
                if col2 == 'annual_income':
                    ax2.hist(data2, bins=bins, edgecolor='black', alpha=0.7, color='lightgreen')
                    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
                else:
                    ax2.hist(data2, bins=bins, edgecolor='black', alpha=0.7, color='salmon')
                
                ax2.set_xlabel(feature2, fontsize=12)
                ax2.set_ylabel('Количество', fontsize=12)
                ax2.set_title(f'Распределение {feature2.lower()}', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                stats2 = f"Среднее: {data2.mean():.1f}\nМедиана: {data2.median():.1f}\nСт. откл.: {data2.std():.1f}"
                ax2.text(0.95, 0.95, stats2, transform=ax2.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                report += self._get_feature_stats(feature2, col2, data2)
            
            plt.tight_layout()
            
            report += "\nВЫВОДЫ ПО ОДНОМЕРНОМУ АНАЛИЗУ:\n"
            report += "1. Анализ формы распределения (нормальное, смещенное, бимодальное)\n"
            report += "2. Оценка центральных тенденций (среднее, медиана)\n"
            report += "3. Выявление выбросов и аномалий\n"
            report += "4. Проверка на пропущенные значения\n"
            
            return report, fig
            
        except Exception as e:
            import traceback
            return f"❌ Ошибка построения гистограмм: {e}\n\n{traceback.format_exc()}", None
    
    def _get_feature_stats(self, feature_name, column_name, data):
        """Получение статистики по признаку"""
        stats_text = f"{feature_name.upper()}:\n"
        stats_text += f"  • Количество значений: {len(data):,}\n"
        stats_text += f"  • Среднее значение: {data.mean():.2f}\n"
        stats_text += f"  • Медиана: {data.median():.2f}\n"
        stats_text += f"  • Стандартное отклонение: {data.std():.2f}\n"
        stats_text += f"  • Минимум: {data.min():.2f}\n"
        stats_text += f"  • Максимум: {data.max():.2f}\n"
        stats_text += f"  • Пропущенных значений: {self.df[column_name].isna().sum():,}\n\n"
        return stats_text