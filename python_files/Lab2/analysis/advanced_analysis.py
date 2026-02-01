import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalysis:
    """Класс для расширенного анализа данных Murder Mystery"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self._identify_left_join_columns()
    
    def _identify_left_join_columns(self):
        """Идентифицирует колонки, которые появились из LEFT JOIN"""
        self.left_join_columns = []
        left_join_keywords = [
            'get_fit_now_member', 'membership', 'interview',
            'facebook_event', 'transcript', '_y'
        ]
        
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in left_join_keywords):
                self.left_join_columns.append(col)
    
    def exploratory_analysis(self):
        """Разведочный анализ объединенной таблицы"""
        if self.df is None or len(self.df) == 0:
            return "❌ Нет данных для анализа"
        
        try:
            report = "🔍 РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ MURDER MYSTERY\n"
            report += "="*70 + "\n\n"
            
            # 1. Общая информация
            report += "1. ОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ:\n"
            report += f"   • Количество строк (людей): {len(self.df):,}\n"
            report += f"   • Количество столбцов: {len(self.df.columns)}\n"
            report += f"   • Объем данных в памяти: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
            
            # 2. Типы данных
            report += "2. ТИПЫ ДАННЫХ:\n"
            dtypes = self.df.dtypes.value_counts()
            for dtype, count in dtypes.items():
                report += f"   • {dtype}: {count} столбцов\n"
            report += "\n"
            
            # 3. Пропущенные значения с учетом LEFT JOIN
            report += "3. АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:\n"
            
            # Рассчитываем общее количество пропусков
            missing_total = self.df.isna().sum().sum()
            total_cells = len(self.df) * len(self.df.columns)
            missing_percent = (missing_total / total_cells) * 100
            
            report += f"   • Всего пропущенных значений: {missing_total:,}\n"
            report += f"   • Процент пропусков: {missing_percent:.2f}%\n"
            
            # Разделяем колонки на основные и LEFT JOIN
            basic_columns = [col for col in self.df.columns if col not in self.left_join_columns]
            left_join_columns = self.left_join_columns
            
            # Анализ основных колонок (не из LEFT JOIN)
            if basic_columns:
                report += f"   • ОСНОВНЫЕ КОЛОНКИ (не из LEFT JOIN):\n"
                basic_missing = self.df[basic_columns].isna().sum().sort_values(ascending=False)
                for col, count in basic_missing.head(5).items():
                    if count > 0:
                        percent = (count / len(self.df)) * 100
                        report += f"      - {col}: {count:,} пропусков ({percent:.1f}%)\n"
            
            # Анализ LEFT JOIN колонок
            if left_join_columns:
                report += f"   • КОЛОНКИ ИЗ LEFT JOIN (отсутствие = нет связи):\n"
                join_missing = self.df[left_join_columns].isna().sum().sort_values(ascending=False)
                for col, count in join_missing.head(5).items():
                    present_count = len(self.df) - count
                    percent_missing = (count / len(self.df)) * 100
                    percent_present = (present_count / len(self.df)) * 100
                    
                    if 'membership' in col.lower():
                        report += f"      - {col}: {present_count:,} в спортзале, {count:,} нет ({percent_present:.1f}% есть)\n"
                    elif 'transcript' in col.lower():
                        report += f"      - {col}: {present_count:,} с расшифровкой, {count:,} без ({percent_present:.1f}% есть)\n"
                    elif 'annual_income' in col.lower():
                        report += f"      - {col}: {present_count:,} с доходом, {count:,} без ({percent_present:.1f}% есть)\n"
                    else:
                        report += f"      - {col}: {present_count:,} записей есть, {count:,} нет ({percent_present:.1f}% есть)\n"
            
            report += "\n"
            
            # 4. Статистика числовых признаков (только не-ID колонки)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            # Исключаем ID колонки из статистики
            id_keywords = ['id', '_id', 'ssn', 'license']
            meaningful_numeric_cols = [
                col for col in numeric_cols 
                if not any(keyword in col.lower() for keyword in id_keywords)
                and col not in left_join_columns  # Исключаем LEFT JOIN колонки
            ]
            
            if len(meaningful_numeric_cols) > 0:
                report += "4. СТАТИСТИКА ЧИСЛОВЫХ ПРИЗНАКОВ (ключевые):\n"
                numeric_stats = self.df[meaningful_numeric_cols].describe().T
                
                for idx, row in numeric_stats.iterrows():
                    report += f"   • {idx}:\n"
                    report += f"      - Среднее: {row['mean']:.2f}\n"
                    report += f"      - Станд. откл.: {row['std']:.2f}\n"
                    report += f"      - Минимум: {row['min']:.2f}\n"
                    report += f"      - 25%: {row['25%']:.2f}\n"
                    report += f"      - 50%: {row['50%']:.2f}\n"
                    report += f"      - 75%: {row['75%']:.2f}\n"
                    report += f"      - Максимум: {row['max']:.2f}\n"
                    report += f"      - Количество значений: {row['count']:,}\n\n"
            
            # 5. Категориальные признаки
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            # Исключаем ID и LEFT JOIN колонки
            meaningful_categorical_cols = [
                col for col in categorical_cols 
                if not any(keyword in col.lower() for keyword in id_keywords)
                and col not in left_join_columns
            ]
            
            if len(meaningful_categorical_cols) > 0:
                report += "5. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (ключевые):\n"
                for col in meaningful_categorical_cols[:8]:  # Ограничим вывод
                    unique_count = self.df[col].nunique()
                    non_null_count = self.df[col].notna().sum()
                    
                    report += f"   • {col}:\n"
                    report += f"      - Уникальных значений: {unique_count}\n"
                    report += f"      - Непустых значений: {non_null_count:,}\n"
                    
                    if unique_count <= 10:  # Для колонок с малым числом уникальных значений
                        value_counts = self.df[col].value_counts()
                        report += f"      - Распределение:\n"
                        for value, count in value_counts.items():
                            percent = (count / non_null_count) * 100 if non_null_count > 0 else 0
                            report += f"        * '{value}': {count:,} ({percent:.1f}%)\n"
                    else:
                        top_values = self.df[col].value_counts().head(3)
                        report += f"      - Топ-3 значения:\n"
                        for value, count in top_values.items():
                            percent = (count / non_null_count) * 100 if non_null_count > 0 else 0
                            report += f"        * '{value}': {count:,} ({percent:.1f}%)\n"
                    report += "\n"
            
            # 6. Выбросы только в значимых числовых колонках
            report += "6. АНАЛИЗ ВЫБРОСОВ (по методу межквартильного размаха):\n"
            if len(meaningful_numeric_cols) > 0:
                for col in meaningful_numeric_cols[:5]:  # Ограничим вывод
                    data = self.df[col].dropna()
                    if len(data) > 0:
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR > 0:  # Проверяем, чтобы IQR не был 0
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = data[(data < lower_bound) | (data > upper_bound)]
                            outlier_percent = (len(outliers) / len(data)) * 100
                            
                            report += f"   • {col}: {len(outliers):,} выбросов из {len(data):,} ({outlier_percent:.2f}%)\n"
                        else:
                            report += f"   • {col}: IQR = 0, нельзя определить выбросы\n"
                report += "\n"
            
            # 7. Рекомендации с учетом LEFT JOIN
            report += "7. РЕКОМЕНДАЦИИ ПО ПРЕДОБРАБОТКЕ ДАННЫХ:\n"
            report += "   1. Для LEFT JOIN колонок (отсутствие = нет записи):\n"
            report += "      - Создать бинарные признаки: has_membership, has_interview и т.д.\n"
            report += "      - Заполнить пропуски значением 'Нет' или 0\n"
            report += "   2. Для основных пропусков:\n"
            report += "      - Числовые: медианой или средним\n"
            report += "      - Категориальные: модой или 'Unknown'\n"
            report += "   3. Обработать выбросы в значимых признаках\n"
            report += "   4. Удалить дублирующиеся ID колонки\n"
            report += "   5. Применить One-Hot Encoding для категориальных признаков\n"
            
            # Визуализация
            self._visualize_exploratory_analysis()
            
            return report
            
        except Exception as e:
            import traceback
            return f"❌ Ошибка разведочного анализа: {e}\n\n{traceback.format_exc()}"
    
    def correlation_analysis(self):
        """Корреляционный анализ с фильтрацией ID колонок"""
        if self.df is None or len(self.df) == 0:
            return "❌ Нет данных для анализа"
        
        try:
            report = "📊 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ДАННЫХ MURDER MYSTERY\n"
            report += "="*70 + "\n\n"
            
            # Отбираем только значимые числовые колонки (не ID, не LEFT JOIN)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            # Исключаем ID колонки и LEFT JOIN колонки
            exclude_keywords = ['id', '_id', 'ssn', 'license', 'person_id', 'date']
            filtered_numeric_cols = [
                col for col in numeric_cols 
                if not any(keyword in col.lower() for keyword in exclude_keywords)
                and col not in self.left_join_columns
            ]
            
            if len(filtered_numeric_cols) < 2:
                report += "❌ Недостаточно значимых числовых признаков для корреляционного анализа\n"
                report += "   (большинство числовых колонок - это ID или технические данные)\n"
                return report
            
            # Полная корреляционная матрица
            corr_matrix = self.df[filtered_numeric_cols].corr()
            
            report += "1. СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.7):\n"
            strong_corrs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_corrs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_value))
            
            if strong_corrs:
                strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, corr in strong_corrs[:10]:
                    direction = "положительная" if corr > 0 else "отрицательная"
                    report += f"   • {col1} ↔ {col2}: r = {corr:.4f} ({direction})\n"
            else:
                report += "   • Сильных корреляций не обнаружено\n"
            report += "\n"
            
            report += "2. УМЕРЕННЫЕ КОРРЕЛЯЦИИ (0.5 < |r| < 0.7):\n"
            moderate_corrs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_value = corr_matrix.iloc[i, j]
                    if 0.5 < abs(corr_value) < 0.7:
                        moderate_corrs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_value))
            
            if moderate_corrs:
                moderate_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, corr in moderate_corrs[:10]:
                    direction = "положительная" if corr > 0 else "отрицательная"
                    report += f"   • {col1} ↔ {col2}: r = {corr:.4f} ({direction})\n"
            else:
                report += "   • Умеренных корреляций не обнаружено\n"
            report += "\n"
            
            report += "3. СЛАБЫЕ КОРРЕЛЯЦИИ (0.3 < |r| < 0.5):\n"
            weak_corrs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_value = corr_matrix.iloc[i, j]
                    if 0.3 < abs(corr_value) < 0.5:
                        weak_corrs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_value))
            
            if weak_corrs:
                weak_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, corr in weak_corrs[:5]:  # Ограничим вывод
                    direction = "положительная" if corr > 0 else "отрицательная"
                    report += f"   • {col1} ↔ {col2}: r = {corr:.4f} ({direction})\n"
            else:
                report += "   • Слабых корреляций не обнаружено\n"
            report += "\n"
            
            # Анализ мультиколлинеарности (только для значимых признаков)
            report += "4. АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ:\n"
            
            numeric_data = self.df[filtered_numeric_cols].dropna()
            if len(numeric_data) > 2 and len(filtered_numeric_cols) > 1:
                try:
                    X = add_constant(numeric_data)
                    vif_data = pd.DataFrame()
                    vif_data["feature"] = X.columns
                    
                    # Вычисляем VIF только если достаточно данных
                    if len(X) > len(X.columns):
                        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                                          for i in range(X.shape[1])]
                        
                        high_vif = vif_data[vif_data['VIF'] > 10]
                        if len(high_vif) > 0:
                            report += "   • Признаки с высокой мультиколлинеарностью (VIF > 10):\n"
                            for _, row in high_vif.iterrows():
                                if row['feature'] != 'const':
                                    report += f"      - {row['feature']}: VIF = {row['VIF']:.2f}\n"
                        else:
                            report += "   • Признаков с высокой мультиколлинеарностью не обнаружено\n"
                    else:
                        report += "   • Недостаточно данных для анализа мультиколлинеарности\n"
                except Exception as e:
                    report += f"   • Ошибка анализа мультиколлинеарности: {str(e)[:100]}...\n"
            else:
                report += "   • Недостаточно данных для анализа мультиколлинеарности\n"
            report += "\n"
            
            # Статистика по корреляциям
            report += "5. СТАТИСТИКА КОРРЕЛЯЦИЙ:\n"
            all_corrs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    all_corrs.append(corr_matrix.iloc[i, j])
            
            if all_corrs:
                report += f"   • Всего пар сравнений: {len(all_corrs)}\n"
                report += f"   • Средний |r|: {np.mean(np.abs(all_corrs)):.3f}\n"
                report += f"   • Медиана |r|: {np.median(np.abs(all_corrs)):.3f}\n"
                
                positive = sum(1 for c in all_corrs if c > 0)
                negative = sum(1 for c in all_corrs if c < 0)
                report += f"   • Положительных корреляций: {positive} ({positive/len(all_corrs)*100:.1f}%)\n"
                report += f"   • Отрицательных корреляций: {negative} ({negative/len(all_corrs)*100:.1f}%)\n"
            report += "\n"
            
            # Визуализация
            try:
                self._visualize_correlation_analysis(corr_matrix)
            except Exception as e:
                report += f"⚠️ Ошибка визуализации: {str(e)[:100]}...\n\n"
            
            return report
            
        except Exception as e:
            import traceback
            return f"❌ Ошибка корреляционного анализа: {e}\n\n{traceback.format_exc()}"
    
    def _visualize_exploratory_analysis(self):
        """Визуализация разведочного анализа"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Распределение пропущенных значений (только основные)
            basic_columns = [col for col in self.df.columns if col not in self.left_join_columns]
            missing_data = self.df[basic_columns].isna().sum().sort_values(ascending=False).head(10)
            
            if len(missing_data) > 0:
                axes[0, 0].barh(range(len(missing_data)), missing_data.values, color='skyblue')
                axes[0, 0].set_yticks(range(len(missing_data)))
                axes[0, 0].set_yticklabels(missing_data.index)
                axes[0, 0].set_xlabel('Количество пропусков')
                axes[0, 0].set_title('Основные колонки с пропусками', fontsize=12)
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, 'Нет пропусков в основных колонках', 
                               ha='center', va='center', fontsize=12)
            
            # 2. Присутствие в LEFT JOIN таблицах
            if self.left_join_columns:
                presence_data = {}
                for col in self.left_join_columns[:5]:  # Первые 5 LEFT JOIN колонок
                    if col in self.df.columns:
                        presence = self.df[col].notna().sum()
                        presence_data[col] = presence
                
                if presence_data:
                    cols = list(presence_data.keys())
                    values = list(presence_data.values())
                    total = len(self.df)
                    percentages = [v/total*100 for v in values]
                    
                    x = range(len(cols))
                    bars = axes[0, 1].bar(x, percentages, color='lightgreen', alpha=0.7)
                    axes[0, 1].set_xticks(x)
                    axes[0, 1].set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in cols], 
                                              rotation=45, ha='right')
                    axes[0, 1].set_ylabel('Процент присутствия (%)')
                    axes[0, 1].set_title('Присутствие в связанных таблицах', fontsize=12)
                    axes[0, 1].grid(True, alpha=0.3, axis='y')
                    
                    # Добавляем значения на столбцы
                    for bar, value, percent in zip(bars, values, percentages):
                        height = bar.get_height()
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                                       f'{value:,}\n({percent:.1f}%)', ha='center', va='bottom', fontsize=9)
            
            # 3. Распределение возраста (если есть)
            if 'age' in self.df.columns:
                age_data = self.df['age'].dropna()
                if len(age_data) > 0:
                    axes[1, 0].hist(age_data, bins=20, alpha=0.7, color='salmon', edgecolor='black')
                    axes[1, 0].set_xlabel('Возраст')
                    axes[1, 0].set_ylabel('Количество')
                    axes[1, 0].set_title('Распределение возраста', fontsize=12)
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Добавляем статистику
                    stats_text = f"Среднее: {age_data.mean():.1f}\nМедиана: {age_data.median():.1f}\nN={len(age_data):,}"
                    axes[1, 0].text(0.95, 0.95, stats_text, transform=axes[1, 0].transAxes,
                                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 4. Распределение дохода (если есть)
            if 'annual_income' in self.df.columns:
                income_data = self.df['annual_income'].dropna()
                if len(income_data) > 0:
                    axes[1, 1].hist(income_data, bins=30, alpha=0.7, color='gold', edgecolor='black')
                    axes[1, 1].set_xlabel('Годовой доход ($)')
                    axes[1, 1].set_ylabel('Количество')
                    axes[1, 1].set_title('Распределение дохода', fontsize=12)
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Форматирование оси X для доходов
                    from matplotlib.ticker import FuncFormatter
                    def income_formatter(x, pos):
                        return f'${x/1000:.0f}K'
                    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(income_formatter))
                    
                    # Добавляем статистику
                    stats_text = f"Среднее: ${income_data.mean():,.0f}\nМедиана: ${income_data.median():,.0f}\nN={len(income_data):,}"
                    axes[1, 1].text(0.95, 0.95, stats_text, transform=axes[1, 1].transAxes,
                                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Ошибка визуализации: {e}")
    
    def _visualize_correlation_analysis(self, corr_matrix):
        """Визуализация корреляционного анализа"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. Тепловая карта корреляций
            im = axes[0].imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0].set_xticks(range(len(corr_matrix.columns)))
            axes[0].set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
            axes[0].set_yticks(range(len(corr_matrix.index)))
            axes[0].set_yticklabels(corr_matrix.index, fontsize=8)
            axes[0].set_title('Корреляционная матрица значимых признаков', 
                             fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=axes[0], label='Коэффициент корреляции')
            
            # 2. Распределение коэффициентов корреляции
            corr_values = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_values.append(corr_matrix.iloc[i, j])
            
            if corr_values:
                axes[1].hist(corr_values, bins=30, edgecolor='black', alpha=0.7, 
                           color='lightgreen', density=True)
                axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
                
                # Добавляем плотность распределения
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(corr_values)
                    x_range = np.linspace(min(corr_values), max(corr_values), 100)
                    axes[1].plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.7)
                except:
                    pass
                
                axes[1].set_xlabel('Коэффициент корреляции (r)')
                axes[1].set_ylabel('Плотность вероятности')
                axes[1].set_title('Распределение коэффициентов корреляции', 
                                 fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                # Статистика
                mean_corr = np.mean(corr_values)
                median_corr = np.median(corr_values)
                std_corr = np.std(corr_values)
                
                stats_text = (f"Среднее: {mean_corr:.3f}\n"
                            f"Медиана: {median_corr:.3f}\n"
                            f"Ст.откл.: {std_corr:.3f}\n"
                            f"N пар: {len(corr_values)}")
                axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                           fontsize=10, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Ошибка визуализации корреляций: {e}")