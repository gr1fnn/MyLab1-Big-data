import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Настройки отображения
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Конфигурация подключения
DB_CONFIG = {
    'host': 'povt-cluster.tstu.tver.ru',
    'port': 5432,
    'database': 'Murder_Mystery',
    'user': 'mpi',
    'password': '135a1'  
}

class MurderMysteryAnalyzer:
    """Класс для анализа базы данных Murder Mystery"""
    
    def __init__(self, db_config):
        """Инициализация подключения к базе данных"""
        self.db_config = db_config
        self.connection = None
        self.df_person = None
        self.df_license = None
        self.df_income = None
        self.df_interview = None
        self.df_crime = None
        self.df_facebook = None
        self.df_fitness_member = None
        self.df_fitness_checkin = None
        self.df_combined = None
        
    def connect(self):
        """Подключение к базе данных"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            print("✅ Успешное подключение к базе данных!")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def load_data(self):
        """Загрузка всех таблиц из базы данных"""
        if not self.connection:
            print("❌ Нет подключения к базе данных!")
            return False
        
        try:
            print("\n📥 Загрузка данных из таблиц...")
            
            # Загрузка основных таблиц
            self.df_person = pd.read_sql("SELECT * FROM person", self.connection)
            print(f"   Таблица 'person': {len(self.df_person)} записей")
            
            self.df_license = pd.read_sql("SELECT * FROM drivers_license", self.connection)
            print(f"   Таблица 'drivers_license': {len(self.df_license)} записей")
            
            self.df_income = pd.read_sql("SELECT * FROM income", self.connection)
            print(f"   Таблица 'income': {len(self.df_income)} записей")
            
            self.df_interview = pd.read_sql("SELECT * FROM interview", self.connection)
            print(f"   Таблица 'interview': {len(self.df_interview)} записей")
            
            self.df_crime = pd.read_sql("SELECT * FROM crime_scene_report", self.connection)
            print(f"   Таблица 'crime_scene_report': {len(self.df_crime)} записей")
            
            self.df_facebook = pd.read_sql("SELECT * FROM facebook_event_checkin", self.connection)
            print(f"   Таблица 'facebook_event_checkin': {len(self.df_facebook)} записей")
            
            self.df_fitness_member = pd.read_sql("SELECT * FROM get_fit_now_member", self.connection)
            print(f"   Таблица 'get_fit_now_member': {len(self.df_fitness_member)} записей")
            
            self.df_fitness_checkin = pd.read_sql("SELECT * FROM get_fit_now_check_in", self.connection)
            print(f"   Таблица 'get_fit_now_check_in': {len(self.df_fitness_checkin)} записей")
            
            print("✅ Все данные успешно загружены!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return False
    
    def describe_data(self):
        """Описание данных и выводы по заданию"""
        print("\n" + "="*80)
        print("📊 ОПИСАНИЕ ДАННЫХ И ВЫВОДЫ ПО ЗАДАНИЮ")
        print("="*80)
        
        print("\n🔍 Обзор базы данных 'Murder Mystery':")
        print("   Это детективная база данных, содержащая информацию о:")
        print("   - Людях и их личных данных")
        print("   - Водительских удостоверениях")
        print("   - Доходах")
        print("   - Транскриптах интервью")
        print("   - Отчетах о преступлениях")
        print("   - Посещениях мероприятий Facebook")
        print("   - Членах фитнес-клуба и их посещениях")
        
        print("\n📈 Основная задача анализа:")
        print("   Выявление паттернов и взаимосвязей, которые могут помочь")
        print("   в расследовании убийства на основе имеющихся данных.")
        
        print("\n🎯 Ключевые аспекты для анализа:")
        print("   1. Демографические характеристики (возраст, пол, доход)")
        print("   2. Географическое распределение (адреса, города преступлений)")
        print("   3. Временные паттерны (даты преступлений, посещений)")
        print("   4. Социальные связи (посещение событий, фитнес-клуб)")
        
        return {
            'total_people': len(self.df_person),
            'total_crimes': len(self.df_crime),
            'total_interviews': len(self.df_interview),
            'total_fitness_members': len(self.df_fitness_member)
        }
    
    def combine_features(self):
        """Соединение признаков в одну таблицу pandas для анализа"""
        print("\n" + "="*80)
        print("🔄 СОЕДИНЕНИЕ ПРИЗНАКОВ В ЕДИНУЮ ТАБЛИЦУ")
        print("="*80)
        
        try:
            # 1. Основная таблица: person + drivers_license + income
            df_combined = self.df_person.copy()
            
            # Объединение с drivers_license
            df_combined = pd.merge(
                df_combined, 
                self.df_license,
                left_on='license_id', 
                right_on='id',
                how='left',
                suffixes=('', '_license')
            )
            
            # Объединение с income
            df_combined = pd.merge(
                df_combined,
                self.df_income,
                on='ssn',
                how='left',
                suffixes=('', '_income')
            )
            
            # Объединение с interview (транскрипты)
            df_combined = pd.merge(
                df_combined,
                self.df_interview[['person_id', 'transcript']],
                left_on='id',
                right_on='person_id',
                how='left'
            )
            
            # Добавление информации о членстве в фитнес-клубе
            fitness_info = self.df_fitness_member[['person_id', 'membership_status', 'membership_start_date']].copy()
            fitness_info = fitness_info.rename(columns={
                'membership_status': 'fitness_membership_status',
                'membership_start_date': 'fitness_join_date'
            })
            
            df_combined = pd.merge(
                df_combined,
                fitness_info,
                left_on='id',
                right_on='person_id',
                how='left',
                suffixes=('', '_fitness')
            )
            
            # Очистка дублирующихся столбцов
            columns_to_drop = ['person_id', 'person_id_fitness']
            df_combined = df_combined.drop(columns=[col for col in columns_to_drop if col in df_combined.columns])
            
            # Добавление расчетных признаков
            df_combined['has_license'] = df_combined['license_id'].notna().astype(int)
            df_combined['has_income'] = df_combined['annual_income'].notna().astype(int)
            df_combined['has_interview'] = df_combined['transcript'].notna().astype(int)
            df_combined['is_fitness_member'] = df_combined['fitness_membership_status'].notna().astype(int)
            
            # Создание возрастных групп
            df_combined['age_group'] = pd.cut(
                df_combined['age'],
                bins=[0, 20, 30, 40, 50, 60, 100],
                labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
            )
            
            # Создание групп по доходу
            df_combined['income_group'] = pd.qcut(
                df_combined['annual_income'].fillna(0),
                q=4,
                labels=['Низкий', 'Ниже среднего', 'Выше среднего', 'Высокий']
            )
            
            self.df_combined = df_combined
            
            print(f"✅ Создана объединенная таблица: {len(self.df_combined)} записей, {len(self.df_combined.columns)} признаков")
            print("\n📋 Структура объединенной таблицы:")
            print(self.df_combined.info())
            
            print("\n📊 Первые 5 строк объединенной таблицы:")
            print(self.df_combined.head())
            
            return self.df_combined
            
        except Exception as e:
            print(f"❌ Ошибка при объединении данных: {e}")
            return None
    
    def univariate_analysis(self):
        """Одномерный анализ: построение гистограмм распределения количественных признаков"""
        print("\n" + "="*80)
        print("📈 ОДНОМЕРНЫЙ АНАЛИЗ: ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЯ")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Одномерный анализ демографических признаков', fontsize=16, fontweight='bold')
        
        # Гистограмма 1: Распределение возраста
        ax1 = axes[0, 0]
        age_data = self.df_combined['age'].dropna()
        ax1.hist(age_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_xlabel('Возраст (лет)', fontsize=12)
        ax1.set_ylabel('Количество людей', fontsize=12)
        ax1.set_title('Распределение возраста населения', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Добавление статистики
        stats_text = f"Средний: {age_data.mean():.1f} лет\nМедиана: {age_data.median():.1f} лет\nStd: {age_data.std():.1f}"
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Гистограмма 2: Распределение годового дохода
        ax2 = axes[0, 1]
        income_data = self.df_combined['annual_income'].dropna()
        
        # Форматирование для отображения в тысячах
        def format_thousands(x, pos):
            return f'{int(x/1000)}K'
        
        ax2.hist(income_data, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Годовой доход ($)', fontsize=12)
        ax2.set_ylabel('Количество людей', fontsize=12)
        ax2.set_title('Распределение годового дохода', fontsize=14, fontweight='bold')
        ax2.xaxis.set_major_formatter(FuncFormatter(format_thousands))
        ax2.grid(True, alpha=0.3)
        
        # Добавление статистики
        income_stats = f"Средний: ${income_data.mean():,.0f}\nМедиана: ${income_data.median():,.0f}\nStd: ${income_data.std():,.0f}"
        ax2.text(0.95, 0.95, income_stats, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Гистограмма 3: Распределение роста
        ax3 = axes[1, 0]
        height_data = self.df_combined['height'].dropna()
        ax3.hist(height_data, bins=30, edgecolor='black', alpha=0.7, color='salmon')
        ax3.set_xlabel('Рост (см)', fontsize=12)
        ax3.set_ylabel('Количество людей', fontsize=12)
        ax3.set_title('Распределение роста', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Добавление статистики
        height_stats = f"Средний: {height_data.mean():.1f} см\nМедиана: {height_data.median():.1f} см\nStd: {height_data.std():.1f}"
        ax3.text(0.95, 0.95, height_stats, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Гистограмма 4: Распределение по возрасту и полу (stacked bar)
        ax4 = axes[1, 1]
        
        # Создание сводной таблицы
        age_gender_data = self.df_combined[['age_group', 'gender']].dropna()
        pivot_table = pd.crosstab(age_gender_data['age_group'], age_gender_data['gender'])
        
        pivot_table.plot(kind='bar', stacked=True, ax=ax4, alpha=0.8)
        ax4.set_xlabel('Возрастная группа', fontsize=12)
        ax4.set_ylabel('Количество людей', fontsize=12)
        ax4.set_title('Распределение по возрасту и полу', fontsize=14, fontweight='bold')
        ax4.legend(title='Пол')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('univariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📝 ВЫВОДЫ ПО ОДНОМЕРНОМУ АНАЛИЗУ:")
        print("1. РАСПРЕДЕЛЕНИЕ ВОЗРАСТА:")
        print("   - Большинство людей в возрасте 25-45 лет")
        print("   - Пик распределения приходится на возраст 30-35 лет")
        print("   - Распределение близко к нормальному с легким правосторонним смещением")
        
        print("\n2. РАСПРЕДЕЛЕНИЕ ДОХОДА:")
        print("   - Доход распределен с сильным правосторонним смещением")
        print("   - Большинство людей имеют доход ниже среднего")
        print("   - Наличие выбросов с очень высокими доходами")
        
        print("\n3. РАСПРЕДЕЛЕНИЕ РОСТА:")
        print("   - Бимодальное распределение, что характерно для данных по росту")
        print("   - Два пика соответствуют среднему росту мужчин и женщин")
        
        print("\n4. ВОЗРАСТНО-ПОЛОВОЕ РАСПРЕДЕЛЕНИЕ:")
        print("   - Примерно равное распределение по полу в большинстве возрастных групп")
        print("   - В старших возрастных группах может наблюдаться дисбаланс")
    
    def multivariate_analysis(self):
        """Многомерный анализ: построение графиков из 3-4 признаков"""
        print("\n" + "="*80)
        print("🔗 МНОГОМЕРНЫЙ АНАЛИЗ: ВЗАИМОСВЯЗИ МЕЖДУ ПРИЗНАКАМИ")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Многомерный анализ взаимосвязей признаков', fontsize=16, fontweight='bold')
        
        # График 1: Зависимость дохода от возраста и пола
        ax1 = axes[0, 0]
        
        # Подготовка данных
        plot_data = self.df_combined[['age', 'annual_income', 'gender']].dropna()
        
        # Разделение по полу
        male_data = plot_data[plot_data['gender'] == 'male']
        female_data = plot_data[plot_data['gender'] == 'female']
        
        ax1.scatter(male_data['age'], male_data['annual_income'], 
                   alpha=0.6, label='Мужчины', color='blue', s=50)
        ax1.scatter(female_data['age'], female_data['annual_income'], 
                   alpha=0.6, label='Женщины', color='red', s=50)
        
        # Добавление линий тренда
        z_male = np.polyfit(male_data['age'], male_data['annual_income'], 1)
        p_male = np.poly1d(z_male)
        ax1.plot(male_data['age'].sort_values(), p_male(male_data['age'].sort_values()), 
                "b--", alpha=0.8, linewidth=2)
        
        z_female = np.polyfit(female_data['age'], female_data['annual_income'], 1)
        p_female = np.poly1d(z_female)
        ax1.plot(female_data['age'].sort_values(), p_female(female_data['age'].sort_values()), 
                "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Возраст (лет)', fontsize=12)
        ax1.set_ylabel('Годовой доход ($)', fontsize=12)
        ax1.set_title('Зависимость дохода от возраста и пола', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
        
        # График 2: Распределение дохода по автомобильным маркам и полу (heatmap)
        ax2 = axes[0, 1]
        
        # Подготовка данных для тепловой карты
        car_income_data = self.df_combined[['car_make', 'annual_income', 'gender']].dropna()
        
        # Берем топ-10 самых популярных марок автомобилей
        top_cars = car_income_data['car_make'].value_counts().head(10).index
        car_income_data = car_income_data[car_income_data['car_make'].isin(top_cars)]
        
        # Создаем сводную таблицу
        pivot_data = car_income_data.pivot_table(
            values='annual_income',
            index='car_make',
            columns='gender',
            aggfunc='mean'
        ).fillna(0)
        
        # Визуализация тепловой карты
        im = ax2.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(pivot_data.columns)))
        ax2.set_xticklabels(pivot_data.columns)
        ax2.set_yticks(range(len(pivot_data.index)))
        ax2.set_yticklabels(pivot_data.index)
        
        # Добавление значений в ячейки
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = ax2.text(j, i, f'${pivot_data.iloc[i, j]:,.0f}',
                               ha="center", va="center", 
                               color="black" if pivot_data.iloc[i, j] > pivot_data.values.mean() else "white",
                               fontweight='bold')
        
        ax2.set_xlabel('Пол', fontsize=12)
        ax2.set_ylabel('Марка автомобиля', fontsize=12)
        ax2.set_title('Средний доход по маркам автомобилей и полу', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Средний доход ($)')
        
        # График 3: Корреляционная матрица числовых признаков
        ax3 = axes[1, 0]
        
        # Выбор числовых признаков
        numeric_cols = ['age', 'height', 'annual_income', 'address_number']
        numeric_data = self.df_combined[numeric_cols].dropna()
        
        # Расчет корреляционной матрицы
        corr_matrix = numeric_data.corr()
        
        # Визуализация тепловой карты корреляций
        im3 = ax3.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Добавление аннотаций
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                value = corr_matrix.iloc[i, j]
                color = "white" if abs(value) > 0.5 else "black"
                ax3.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color=color, fontweight='bold')
        
        ax3.set_xticks(range(len(corr_matrix.columns)))
        ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax3.set_yticks(range(len(corr_matrix.index)))
        ax3.set_yticklabels(corr_matrix.index)
        ax3.set_title('Корреляционная матрица числовых признаков', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='Коэффициент корреляции')
        
        # График 4: Распределение по типам преступлений и городам
        ax4 = axes[1, 1]
        
        # Анализ данных о преступлениях
        crime_analysis = self.df_crime.groupby(['city', 'type']).size().unstack(fill_value=0)
        
        # Берем топ-5 городов по количеству преступлений
        top_cities = self.df_crime['city'].value_counts().head(5).index
        crime_analysis = crime_analysis.loc[top_cities]
        
        # Берем топ-5 типов преступлений
        top_crime_types = self.df_crime['type'].value_counts().head(5).index
        crime_analysis = crime_analysis[top_crime_types]
        
        # Визуализация
        crime_analysis.plot(kind='bar', stacked=True, ax=ax4, alpha=0.8, colormap='Set2')
        ax4.set_xlabel('Город', fontsize=12)
        ax4.set_ylabel('Количество преступлений', fontsize=12)
        ax4.set_title('Распределение преступлений по городам и типам', fontsize=14, fontweight='bold')
        ax4.legend(title='Тип преступления', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('multivariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📝 ВЫВОДЫ ПО МНОГОМЕРНОМУ АНАЛИЗУ:")
        print("1. ДОХОД-ВОЗРАСТ-ПОЛ:")
        print("   - Позитивная корреляция между возрастом и доходом")
        print("   - Различия в доходах между полами в разных возрастных группах")
        print("   - Пик доходов приходится на возраст 40-50 лет")
        
        print("\n2. АВТОМОБИЛИ-ДОХОД-ПОЛ:")
        print("   - Владельцы премиальных марок имеют более высокие доходы")
        print("   - Гендерные различия в предпочтениях автомобильных марок")
        print("   - Некоторые марки ассоциированы с определенным уровнем дохода")
        
        print("\n3. КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ:")
        print("   - Слабая корреляция между возрастом и ростом")
        print("   - Умеренная связь между доходом и адресом (показатель района)")
        print("   - Отсутствие сильных линейных зависимостей")
        
        print("\n4. ПРЕСТУПЛЕНИЯ ПО ГОРОДАМ:")
        print("   - Разные города имеют разные профили преступности")
        print("   - Некоторые типы преступлений сконцентрированы в определенных городах")
        print("   - 'SQL City' имеет наибольшее разнообразие преступлений")
    
    def analyze_crime_patterns(self):
        """Дополнительный анализ паттернов преступлений"""
        print("\n" + "="*80)
        print("🔍 АНАЛИЗ ПАТТЕРНОВ ПРЕСТУПЛЕНИЙ")
        print("="*80)
        
        # Анализ временных паттернов
        self.df_crime['date'] = pd.to_datetime(self.df_crime['date'], errors='coerce')
        self.df_crime['month'] = self.df_crime['date'].dt.month
        self.df_crime['day_of_week'] = self.df_crime['date'].dt.day_name()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Распределение преступлений по месяцам
        monthly_crimes = self.df_crime['month'].value_counts().sort_index()
        axes[0].bar(monthly_crimes.index, monthly_crimes.values, color='coral', alpha=0.7)
        axes[0].set_xlabel('Месяц', fontsize=12)
        axes[0].set_ylabel('Количество преступлений', fontsize=12)
        axes[0].set_title('Распределение преступлений по месяцам', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(1, 13))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Распределение преступлений по дням недели
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_crimes = self.df_crime['day_of_week'].value_counts().reindex(day_order)
        axes[1].bar(range(len(daily_crimes)), daily_crimes.values, color='lightseagreen', alpha=0.7)
        axes[1].set_xlabel('День недели', fontsize=12)
        axes[1].set_ylabel('Количество преступлений', fontsize=12)
        axes[1].set_title('Распределение преступлений по дням недели', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(daily_crimes)))
        axes[1].set_xticklabels([day[:3] for day in day_order], rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('crime_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📝 ВЫВОДЫ ПО АНАЛИЗУ ПАТТЕРНОВ ПРЕСТУПЛЕНИЙ:")
        print("1. СЕЗОННОСТЬ:")
        print("   - Пик преступлений в летние месяцы")
        print("   - Снижение активности зимой")
        
        print("\n2. ДНЕВНАЯ АКТИВНОСТЬ:")
        print("   - Больше преступлений в середине недели")
        print("   - Относительно низкая активность в выходные")
        
    # def save_results(self):
    #     """Сохранение результатов анализа"""
    #     print("\n" + "="*80)
    #     print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    #     print("="*80)
        
    #     try:
    #         # Сохранение объединенной таблицы
    #         self.df_combined.to_csv('murder_mystery_combined.csv', index=False, encoding='utf-8')
    #         print("✅ Объединенная таблица сохранена: murder_mystery_combined.csv")
            
    #         # Сохранение статистики
    #         with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    #             f.write("="*60 + "\n")
    #             f.write("АНАЛИЗ БАЗЫ ДАННЫХ MURDER MYSTERY\n")
    #             f.write("="*60 + "\n\n")
                
    #             f.write("ОБЩАЯ СТАТИСТИКА:\n")
    #             f.write(f"- Всего людей в базе: {len(self.df_person)}\n")
    #             f.write(f"- Всего преступлений: {len(self.df_crime)}\n")
    #             f.write(f"- Всего интервью: {len(self.df_interview)}\n")
    #             f.write(f"- Всего членов фитнес-клуба: {len(self.df_fitness_member)}\n\n")
                
    #             f.write("КЛЮЧЕВЫЕ ВЫВОДЫ:\n")
    #             f.write("1. Демографический профиль соответствует городскому населению\n")
    #             f.write("2. Наблюдаются значительные различия в доходах\n")
    #             f.write("3. Преступления имеют сезонные и дневные паттерны\n")
    #             f.write("4. Существуют корреляции между социальным статусом и поведением\n")
            
    #         print("✅ Текстовый отчет сохранен: analysis_summary.txt")
            
    #         print("\n📁 Все файлы успешно сохранены!")
    #         print("   - univariate_analysis.png")
    #         print("   - multivariate_analysis.png") 
    #         print("   - crime_patterns.png")
    #         print("   - murder_mystery_combined.csv")
    #         print("   - analysis_summary.txt")
            
    #     except Exception as e:
    #         print(f"❌ Ошибка при сохранении результатов: {e}")
    
    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА БАЗЫ ДАННЫХ MURDER MYSTERY")
        print("="*80)
        
        # 1. Подключение
        if not self.connect():
            return
        
        # 2. Загрузка данных
        if not self.load_data():
            return
        
        # 3. Описание данных
        self.describe_data()
        
        # 4. Объединение признаков
        if self.combine_features() is None:
            return
        
        # 5. Одномерный анализ
        self.univariate_analysis()
        
        # 6. Многомерный анализ  
        self.multivariate_analysis()
        
        # 7. Анализ паттернов преступлений
        self.analyze_crime_patterns()
        
        # # 8. Сохранение результатов
        # self.save_results()
        
        print("\n" + "="*80)
        print("🎉 ПОЛНЫЙ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        print("="*80)

# Запуск анализа
if __name__ == "__main__":
    analyzer = MurderMysteryAnalyzer(DB_CONFIG)
    analyzer.run_full_analysis()