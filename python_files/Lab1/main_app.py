import sys
import warnings
warnings.filterwarnings('ignore')

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtCore import QObject, Signal, QTimer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import psycopg2
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Импорт UI
from murder_mystery_ui import Ui_MainWindow

class DatabaseAnalyzer(QObject):
    """Класс для анализа базы данных с загрузкой всех таблиц"""
    
    status_updated = Signal(str)
    progress_updated = Signal(int)
    
    def __init__(self):
        super().__init__()
        self.db_config = None
        self.connection = None
        self.dataframes = {}
        self.combined_df = None
        self.table_names = []  # Все имена таблиц в базе
        
    def test_connection(self, config):
        """Тест подключения к БД"""
        try:
            self.status_updated.emit("Тестируем подключение...")
            conn = psycopg2.connect(**config)
            
            # Получаем список всех таблиц
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            self.table_names = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            self.db_config = config
            self.status_updated.emit(f"✅ Подключение успешно! Найдено таблиц: {len(self.table_names)}")
            return True
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка подключения: {e}")
            return False
    
    def load_all_data(self):
        """Загрузка ВСЕХ таблиц из базы данных"""
        if not self.db_config:
            self.status_updated.emit("❌ Сначала настройте подключение")
            return False
        
        try:
            self.status_updated.emit(f"Загружаем ВСЕ таблицы ({len(self.table_names)} таблиц)...")
            self.connection = psycopg2.connect(**self.db_config)
            
            # Очищаем предыдущие данные
            self.dataframes = {}
            
            # Загружаем все таблицы
            total_tables = len(self.table_names)
            for i, table_name in enumerate(self.table_names):
                self.progress_updated.emit(int((i / total_tables) * 100))
                
                try:
                    self.dataframes[table_name] = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
                    self.status_updated.emit(f"✅ Таблица '{table_name}' загружена: {len(self.dataframes[table_name])} строк")
                except Exception as e:
                    self.status_updated.emit(f"⚠️ Ошибка загрузки таблицы '{table_name}': {e}")
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"✅ Все данные загружены! Загружено таблиц: {len(self.dataframes)}")
            
            # Показываем статистику
            self.show_data_statistics()
            return True
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка загрузки: {e}")
            return False
    
    def show_data_statistics(self):
        """Показ статистики по загруженным данным"""
        if not self.dataframes:
            return
        
        stats = "📊 СТАТИСТИКА ЗАГРУЖЕННЫХ ДАННЫХ:\n\n"
        
        for table_name, df in self.dataframes.items():
            stats += f"📋 {table_name}:\n"
            stats += f"   • Количество строк: {len(df):,}\n"
            stats += f"   • Количество столбцов: {len(df.columns)}\n"
            
            # Показываем первые 5 названий столбцов
            columns_preview = ", ".join(df.columns[:5])
            if len(df.columns) > 5:
                columns_preview += f" ... и еще {len(df.columns)-5} столбцов"
            stats += f"   • Столбцы: {columns_preview}\n\n"
        
        self.status_updated.emit(stats)
    
    def combine_data(self, feature_type="Основные демографические"):
        """Объединение данных в одну таблицу с разными вариантами"""
        try:
            self.status_updated.emit(f"Объединяем данные (тип: {feature_type})...")
            
            # Проверяем наличие основных таблиц
            required_tables = ['person']
            missing_tables = [t for t in required_tables if t not in self.dataframes]
            
            if missing_tables:
                self.status_updated.emit(f"❌ Отсутствуют таблицы: {', '.join(missing_tables)}")
                return None
            
            # Начинаем с таблицы person
            df = self.dataframes['person'].copy()
            
            # Вариант 1: Основные демографические
            if feature_type == "Основные демографические":
                # Добавляем только базовые таблицы
                if 'drivers_license' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['drivers_license'],
                        left_on='license_id',
                        right_on='id',
                        how='left',
                        suffixes=('', '_license')
                    )
                
                if 'income' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['income'],
                        on='ssn',
                        how='left'
                    )
            
            # Вариант 2: Полный набор
            elif feature_type == "Полный набор признаков":
                # Добавляем все возможные таблицы
                if 'drivers_license' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['drivers_license'],
                        left_on='license_id',
                        right_on='id',
                        how='left',
                        suffixes=('', '_license')
                    )
                
                if 'income' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['income'],
                        on='ssn',
                        how='left'
                    )
                
                if 'interview' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['interview'][['person_id', 'transcript']],
                        left_on='id',
                        right_on='person_id',
                        how='left'
                    )
                
                if 'get_fit_now_member' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['get_fit_now_member'][['person_id', 'membership_status', 'membership_start_date']],
                        left_on='id',
                        right_on='person_id',
                        how='left'
                    )
            
            # Вариант 3: Демография + Доходы + Авто
            elif feature_type == "Демография + Доходы + Авто":
                if 'drivers_license' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['drivers_license'],
                        left_on='license_id',
                        right_on='id',
                        how='left',
                        suffixes=('', '_license')
                    )
                
                if 'income' in self.dataframes:
                    df = pd.merge(
                        df,
                        self.dataframes['income'],
                        on='ssn',
                        how='left'
                    )
            
            # Вариант 4: Все доступные данные
            elif feature_type == "Все доступные данные":
                # Пытаемся добавить все таблицы по связям
                merge_attempts = [
                    ('drivers_license', 'license_id', 'id'),
                    ('income', 'ssn', 'ssn'),
                    ('interview', 'id', 'person_id'),
                    ('get_fit_now_member', 'id', 'person_id'),
                    ('facebook_event_checkin', 'id', 'person_id'),
                ]
                
                for table_name, left_key, right_key in merge_attempts:
                    if table_name in self.dataframes:
                        try:
                            if left_key == 'id' and right_key == 'person_id':
                                # Для таблиц с person_id
                                df = pd.merge(
                                    df,
                                    self.dataframes[table_name],
                                    left_on=left_key,
                                    right_on=right_key,
                                    how='left',
                                    suffixes=('', f'_{table_name}')
                                )
                            else:
                                # Для других таблиц
                                df = pd.merge(
                                    df,
                                    self.dataframes[table_name],
                                    left_on=left_key,
                                    right_on=right_key,
                                    how='left',
                                    suffixes=('', f'_{table_name}')
                                )
                        except Exception as e:
                            self.status_updated.emit(f"⚠️ Не удалось объединить {table_name}: {e}")
            
            # Очищаем дубликаты столбцов
            columns_to_drop = []
            for col in df.columns:
                if col.endswith('_y') or col in ['person_id', 'id_license']:
                    columns_to_drop.append(col)
            
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
            
            # Добавляем расчетные признаки
            if 'license_id' in df.columns:
                df['has_license'] = df['license_id'].notna().astype(int)
            
            if 'annual_income' in df.columns:
                df['has_income'] = df['annual_income'].notna().astype(int)
                # Группы дохода
                df['income_group'] = pd.qcut(
                    df['annual_income'].fillna(0),
                    q=4,
                    labels=['Низкий', 'Ниже среднего', 'Выше среднего', 'Высокий']
                )
            
            if 'age' in df.columns:
                # Возрастные группы
                df['age_group'] = pd.cut(
                    df['age'].fillna(df['age'].median()),
                    bins=[0, 20, 30, 40, 50, 60, 100],
                    labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
                )
            
            self.combined_df = df
            self.status_updated.emit(f"✅ Объединенная таблица создана: {len(df):,} строк, {len(df.columns)} столбцов")
            return df
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка объединения: {e}")
            import traceback
            self.status_updated.emit(traceback.format_exc())
            return None
    
    def plot_histograms(self, feature1, feature2, bins=30):
        """Построение 2 гистограмм"""
        if self.combined_df is None:
            self.status_updated.emit("❌ Сначала объедините данные")
            return None
        
        try:
            # Маппинг русских названий на английские
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
            if col1 in self.combined_df.columns:
                data1 = self.combined_df[col1].dropna()
                ax1.hist(data1, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
                ax1.set_xlabel(feature1, fontsize=12)
                ax1.set_ylabel('Количество', fontsize=12)
                ax1.set_title(f'Распределение {feature1.lower()}', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Статистика
                stats1 = f"Среднее: {data1.mean():.1f}\nМедиана: {data1.median():.1f}\nСт. откл.: {data1.std():.1f}"
                ax1.text(0.95, 0.95, stats1, transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Гистограмма 2
            if col2 in self.combined_df.columns:
                data2 = self.combined_df[col2].dropna()
                
                # Форматирование для дохода
                if col2 == 'annual_income':
                    ax2.hist(data2, bins=bins, edgecolor='black', alpha=0.7, color='lightgreen')
                    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
                else:
                    ax2.hist(data2, bins=bins, edgecolor='black', alpha=0.7, color='salmon')
                
                ax2.set_xlabel(feature2, fontsize=12)
                ax2.set_ylabel('Количество', fontsize=12)
                ax2.set_title(f'Распределение {feature2.lower()}', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Статистика
                stats2 = f"Среднее: {data2.mean():.1f}\nМедиана: {data2.median():.1f}\nСт. откл.: {data2.std():.1f}"
                ax2.text(0.95, 0.95, stats2, transform=ax2.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            # Формируем отчет
            report = "📊 ОДНОМЕРНЫЙ АНАЛИЗ (ГИСТОГРАММЫ)\n\n"
            
            for feature, col in [(feature1, col1), (feature2, col2)]:
                if col in self.combined_df.columns:
                    data = self.combined_df[col].dropna()
                    report += f"{feature.upper()}:\n"
                    report += f"  • Количество значений: {len(data):,}\n"
                    report += f"  • Среднее значение: {data.mean():.2f}\n"
                    report += f"  • Медиана: {data.median():.2f}\n"
                    report += f"  • Стандартное отклонение: {data.std():.2f}\n"
                    report += f"  • Минимум: {data.min():.2f}\n"
                    report += f"  • Максимум: {data.max():.2f}\n"
                    report += f"  • Пропущенных значений: {self.combined_df[col].isna().sum():,}\n\n"
            
            report += "ВЫВОДЫ ПО ОДНОМЕРНОМУ АНАЛИЗУ:\n"
            report += "1. Анализ формы распределения (нормальное, смещенное, бимодальное)\n"
            report += "2. Оценка центральных тенденций (среднее, медиана)\n"
            report += "3. Выявление выбросов и аномалий\n"
            report += "4. Проверка на пропущенные значения\n"
            
            self.status_updated.emit("✅ Гистограммы построены!")
            return report
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка построения гистограмм: {e}")
            import traceback
            return f"Ошибка: {e}\n\n{traceback.format_exc()}"
    
    def plot_multivariate(self, graph_type):
        """Построение многомерных графиков"""
        if self.combined_df is None:
            self.status_updated.emit("❌ Сначала объедините данные")
            return None
        
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            
            if graph_type == "Доход-Возраст-Пол":
                if all(col in self.combined_df.columns for col in ['age', 'annual_income', 'gender']):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    plot_data = self.combined_df[['age', 'annual_income', 'gender']].dropna()
                    
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
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Отчет
                    report = self._create_multivariate_report(graph_type)
                    
            elif graph_type == "Автомобили-Доход-Пол":
                if all(col in self.combined_df.columns for col in ['car_make', 'annual_income', 'gender']):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    car_data = self.combined_df[['car_make', 'annual_income', 'gender']].dropna()
                    
                    # Берем топ-8 марок
                    top_cars = car_data['car_make'].value_counts().head(8).index
                    car_data = car_data[car_data['car_make'].isin(top_cars)]
                    
                    # Средний доход по маркам и полу
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
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Отчет
                    report = self._create_multivariate_report(graph_type)
                    
            elif graph_type == "Преступления по городам":
                if 'crime_scene_report' in self.dataframes:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    crime_df = self.dataframes['crime_scene_report']
                    
                    # Анализ преступлений по городам
                    city_crimes = crime_df['city'].value_counts().head(10)
                    
                    ax.bar(city_crimes.index, city_crimes.values, color='coral', alpha=0.7)
                    ax.set_xlabel('Город', fontsize=12)
                    ax.set_ylabel('Количество преступлений', fontsize=12)
                    ax.set_title('Распределение преступлений по городам', fontsize=14, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Отчет
                    report = self._create_multivariate_report(graph_type)
                    
            elif graph_type == "Корреляционная матрица":
                # Выбираем числовые колонки
                numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    corr_matrix = self.combined_df[numeric_cols].corr()
                    
                    # Тепловая карта
                    im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
                    
                    # Добавляем значения
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
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Отчет
                    report = self._create_multivariate_report(graph_type)
                else:
                    report = "❌ Недостаточно числовых данных для корреляционной матрицы"
            
            self.status_updated.emit("✅ Многомерный график построен!")
            return report
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка построения графика: {e}")
            import traceback
            return f"Ошибка: {e}\n\n{traceback.format_exc()}"
    
    def _create_multivariate_report(self, graph_type):
        """Создание отчета для многомерного анализа"""
        report = f"📈 МНОГОМЕРНЫЙ АНАЛИЗ: {graph_type}\n\n"
        
        if graph_type == "Доход-Возраст-Пол":
            report += "АНАЛИЗ ЗАВИСИМОСТИ ДОХОДА ОТ ВОЗРАСТА И ПОЛА:\n\n"
            if all(col in self.combined_df.columns for col in ['age', 'annual_income', 'gender']):
                plot_data = self.combined_df[['age', 'annual_income', 'gender']].dropna()
                
                for gender in ['male', 'female']:
                    gender_data = plot_data[plot_data['gender'] == gender]
                    if len(gender_data) > 0:
                        report += f"{'Мужчины' if gender == 'male' else 'Женщины'}:\n"
                        report += f"  • Количество: {len(gender_data):,}\n"
                        report += f"  • Средний возраст: {gender_data['age'].mean():.1f} лет\n"
                        report += f"  • Средний доход: ${gender_data['annual_income'].mean():,.0f}\n"
                        report += f"  • Медианный доход: ${gender_data['annual_income'].median():,.0f}\n\n"
                
                # Корреляция
                male_corr = plot_data[plot_data['gender'] == 'male'][['age', 'annual_income']].corr().iloc[0, 1]
                female_corr = plot_data[plot_data['gender'] == 'female'][['age', 'annual_income']].corr().iloc[0, 1]
                
                report += "КОРРЕЛЯЦИЯ ВОЗРАСТ-ДОХОД:\n"
                report += f"  • Мужчины: {male_corr:.3f}\n"
                report += f"  • Женщины: {female_corr:.3f}\n\n"
                
                report += "ВЫВОДЫ:\n"
                report += "1. Наличие положительной/отрицательной корреляции между возрастом и доходом\n"
                report += "2. Гендерные различия в уровне доходов\n"
                report += "3. Возрастные пики доходов для разных групп\n"
        
        elif graph_type == "Автомобили-Доход-Пол":
            report += "АНАЛИЗ СВЯЗИ МАРКИ АВТОМОБИЛЯ С ДОХОДОМ И ПОЛОМ:\n\n"
            if all(col in self.combined_df.columns for col in ['car_make', 'annual_income', 'gender']):
                car_data = self.combined_df[['car_make', 'annual_income', 'gender']].dropna()
                
                report += "СТАТИСТИКА ПО МАРКАМ АВТОМОБИЛЕЙ:\n"
                top_cars = car_data['car_make'].value_counts().head(10)
                for car, count in top_cars.items():
                    car_stats = car_data[car_data['car_make'] == car]
                    report += f"  • {car}: {count:,} владельцев, "
                    report += f"ср. доход: ${car_stats['annual_income'].mean():,.0f}\n"
                
                report += "\nВЫВОДЫ:\n"
                report += "1. Премиальные марки ассоциированы с высокими доходами\n"
                report += "2. Гендерные предпочтения в выборе автомобилей\n"
                report += "3. Марки-индикаторы социального статуса\n"
        
        elif graph_type == "Преступления по городам":
            report += "АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПРЕСТУПЛЕНИЙ ПО ГОРОДАМ:\n\n"
            if 'crime_scene_report' in self.dataframes:
                crime_df = self.dataframes['crime_scene_report']
                
                report += "СТАТИСТИКА ПРЕСТУПЛЕНИЙ:\n"
                report += f"  • Всего преступлений: {len(crime_df):,}\n"
                report += f"  • Уникальных городов: {crime_df['city'].nunique()}\n"
                report += f"  • Уникальных типов преступлений: {crime_df['type'].nunique()}\n\n"
                
                top_cities = crime_df['city'].value_counts().head(5)
                report += "ТОП-5 ГОРОДОВ ПО ПРЕСТУПЛЕНИЯМ:\n"
                for city, count in top_cities.items():
                    report += f"  • {city}: {count:,} преступлений\n"
                
                report += "\nВЫВОДЫ:\n"
                report += "1. Концентрация преступлений в определенных городах\n"
                report += "2. Города с наибольшей криминальной активностью\n"
                report += "3. Возможные географические паттерны преступности\n"
        
        elif graph_type == "Корреляционная матрица":
            report += "АНАЛИЗ КОРРЕЛЯЦИЙ МЕЖДУ ПРИЗНАКАМИ:\n\n"
            
            numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.combined_df[numeric_cols].corr()
                
                report += "СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.7):\n"
                strong_corrs = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            strong_corrs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_value))
                
                if strong_corrs:
                    for col1, col2, corr in strong_corrs:
                        report += f"  • {col1} ↔ {col2}: {corr:.3f}\n"
                else:
                    report += "  • Сильных корреляций не обнаружено\n"
                
                report += "\nУМЕРЕННЫЕ КОРРЕЛЯЦИИ (0.5 < |r| < 0.7):\n"
                moderate_corrs = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_value = corr_matrix.iloc[i, j]
                        if 0.5 < abs(corr_value) < 0.7:
                            moderate_corrs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_value))
                
                if moderate_corrs:
                    for col1, col2, corr in moderate_corrs[:5]:  # Ограничим вывод
                        report += f"  • {col1} ↔ {col2}: {corr:.3f}\n"
                else:
                    report += "  • Умеренных корреляций не обнаружено\n"
                
                report += "\nВЫВОДЫ:\n"
                report += "1. Наличие сильных линейных зависимостей между признаками\n"
                report += "2. Возможность редукции размерности данных\n"
                report += "3. Выявление ключевых взаимосвязанных факторов\n"
        
        return report
    
    def get_data_info(self):
        """Получение полной информации о данных"""
        if not self.dataframes:
            return "❌ Данные не загружены"
        
        info = "📋 ПОЛНОЕ ОПИСАНИЕ БАЗЫ ДАННЫХ MURDER MYSTERY\n\n"
        info += "="*60 + "\n"
        info += "ОБЩАЯ ИНФОРМАЦИЯ:\n"
        info += f"• Всего таблиц в базе: {len(self.table_names)}\n"
        info += f"• Успешно загружено: {len(self.dataframes)} таблиц\n"
        info += f"• Список таблиц: {', '.join(self.table_names)}\n\n"
        
        info += "СТРУКТУРА БАЗЫ ДАННЫХ:\n"
        for table_name in self.table_names:
            if table_name in self.dataframes:
                df = self.dataframes[table_name]
                info += f"\n📊 {table_name.upper()}:\n"
                info += f"   • Записей: {len(df):,}\n"
                info += f"   • Столбцов: {len(df.columns)}\n"
                info += f"   • Типы данных:\n"
                
                # Сводка по типам данных
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    info += f"     - {dtype}: {count} столбцов\n"
                
                # Пример данных
                info += f"   • Пример столбцов: {', '.join(df.columns[:5])}"
                if len(df.columns) > 5:
                    info += f" ... и еще {len(df.columns)-5}"
                info += "\n"
        
        info += "\n" + "="*60 + "\n"
        info += "ВЫВОДЫ И НАПРАВЛЕНИЯ АНАЛИЗА:\n\n"
        info += "1. КЛЮЧЕВЫЕ ТАБЛИЦЫ:\n"
        info += "   • person - основная таблица с информацией о людях\n"
        info += "   • drivers_license - данные о водительских правах\n"
        info += "   • income - информация о доходах\n"
        info += "   • crime_scene_report - отчеты о преступлениях\n"
        info += "   • interview - транскрипты интервью\n\n"
        
        info += "2. ВОЗМОЖНОСТИ АНАЛИЗА:\n"
        info += "   • Демографический анализ (возраст, пол, рост)\n"
        info += "   • Экономический анализ (доходы, имущество)\n"
        info += "   • Криминальный анализ (преступления, расследования)\n"
        info += "   • Социальный анализ (события, членства)\n\n"
        
        info += "3. ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:\n"
        info += "   • Выявление демографических паттернов\n"
        info += "   • Обнаружение корреляций между признаками\n"
        info += "   • Построение профилей для расследования\n"
        info += "   • Генерация гипотез для детективного анализа\n"
        
        return info


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Настройка UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Инициализация анализатора
        self.analyzer = DatabaseAnalyzer()
        self.analyzer.status_updated.connect(self.update_status)
        self.analyzer.progress_updated.connect(self.update_progress)
        
        # Подключение кнопок
        self.ui.testButton.clicked.connect(self.test_connection)
        self.ui.loadAllButton.clicked.connect(self.load_all_data)
        self.ui.combineButton.clicked.connect(self.combine_data)
        self.ui.plotHistogramButton.clicked.connect(self.plot_histograms)
        self.ui.plotMultivariateButton.clicked.connect(self.plot_multivariate)
        
    def test_connection(self):
        """Тестирование подключения"""
        config = {
            'host': self.ui.hostInput.text(),
            'port': int(self.ui.portInput.text()),
            'database': self.ui.databaseInput.text(),
            'user': self.ui.userInput.text(),
            'password': self.ui.passwordInput.text()
        }
        
        if self.analyzer.test_connection(config):
            self.ui.loadAllButton.setEnabled(True)
    
    def load_all_data(self):
        """Загрузка ВСЕХ данных"""
        self.ui.progressBar.setValue(0)
        
        # Используем таймер для неблокирующей загрузки
        QTimer.singleShot(100, self._load_data_async)
    
    def _load_data_async(self):
        """Асинхронная загрузка данных"""
        if self.analyzer.load_all_data():
            self.ui.combineButton.setEnabled(True)
            
            # Показываем полное описание данных
            info = self.analyzer.get_data_info()
            self.ui.connectionOutput.setText(info)
    
    def combine_data(self):
        """Объединение данных"""
        feature_type = self.ui.featureComboBox.currentText()
        
        df = self.analyzer.combine_data(feature_type)
        if df is not None:
            self.ui.plotHistogramButton.setEnabled(True)
            self.ui.plotMultivariateButton.setEnabled(True)
            
            # Показываем информацию о таблице
            info = f"✅ ОБЪЕДИНЕННАЯ ТАБЛИЦА СОЗДАНА\n\n"
            info += f"Тип объединения: {feature_type}\n"
            info += f"Размер таблицы: {len(df):,} строк × {len(df.columns)} столбцов\n\n"
            info += "СТРУКТУРА ТАБЛИЦЫ:\n"
            
            # Группируем столбцы по типу данных
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            info += f"• Числовые столбцы ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}"
            if len(numeric_cols) > 10:
                info += f" ... и еще {len(numeric_cols)-10}"
            info += "\n"
            
            info += f"• Текстовые столбцы ({len(object_cols)}): {', '.join(object_cols[:10])}"
            if len(object_cols) > 10:
                info += f" ... и еще {len(object_cols)-10}"
            info += "\n"
            
            if datetime_cols:
                info += f"• Дата/время ({len(datetime_cols)}): {', '.join(datetime_cols)}\n"
            
            info += "\nПЕРВЫЕ 5 СТРОК:\n"
            info += df.head().to_string()
            
            self.ui.dataOutput.setText(info)
    
    def plot_histograms(self):
        """Построение гистограмм"""
        feature1 = self.ui.histogram1Combo.currentText()
        feature2 = self.ui.histogram2Combo.currentText()
        bins = self.ui.binsSpinBox.value()
        
        report = self.analyzer.plot_histograms(feature1, feature2, bins)
        if report:
            self.ui.analysisOutput.setText(report)
    
    def plot_multivariate(self):
        """Построение многомерных графиков"""
        graph_type = self.ui.graph1Combo.currentText()
        
        report = self.analyzer.plot_multivariate(graph_type)
        if report:
            self.ui.analysisOutput.setText(report)
    
    def update_status(self, message):
        """Обновление статуса"""
        self.ui.statusbar.showMessage(message)
        print(message)  # Для отладки
        
        # Добавляем сообщение в вывод
        if "✅" in message or "❌" in message or "⚠️" in message or "📊" in message or "📋" in message:
            current_text = self.ui.connectionOutput.toPlainText()
            if len(current_text) > 5000:  # Ограничиваем размер
                current_text = current_text[-4000:]
            self.ui.connectionOutput.setText(current_text + "\n" + message)
    
    def update_progress(self, value):
        """Обновление прогресс-бара"""
        self.ui.progressBar.setValue(value)


# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())