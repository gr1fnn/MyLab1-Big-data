#!/usr/bin/env python3
"""
Task 2: Анализ данных Murder Mystery с помощью PySpark
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# ==================== НАСТРОЙКА ПУТЕЙ ====================

# 1. Путь к Java
java_path = r"C:\Program Files\Java\jdk-17"
if os.path.exists(java_path):
    os.environ["JAVA_HOME"] = java_path
    print(f"✅ Установлена JAVA_HOME: {java_path}")

# 2. Путь к Hadoop (winutils) - исправлено!
hadoop_path = r"C:\hadoop"  # Используем raw string
if os.path.exists(hadoop_path):
    # Устанавливаем системные переменные
    os.environ['HADOOP_HOME'] = hadoop_path
    os.environ['PATH'] = f"{hadoop_path}\\bin;{os.environ['PATH']}"
    
    # Также устанавливаем hadoop.home.dir
    os.environ['hadoop.home.dir'] = hadoop_path
    
    # Для PySpark
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    print(f"✅ Установлен HADOOP_HOME: {hadoop_path}")
else:
    print(f"⚠️  Hadoop не найден по пути: {hadoop_path}")
    print("Примечание: Spark все равно должен запуститься")

class SparkAnalyzer:
    def __init__(self):
        self.spark = None
        self.dataframes = {}
        
    def initialize_spark(self):
        """Инициализация Spark сессии"""
        print("Инициализация Spark...")
        
        try:
            # Путь к драйверу PostgreSQL
            postgresql_jar = r"C:\Program Files\Java\postgresql-42.7.9.jar"
            
            spark_builder = SparkSession.builder \
                .appName("MurderMysteryAnalysis") \
                .master("local[*]") \
                .config("spark.sql.shuffle.partitions", "4") \
                .config("spark.driver.memory", "2g") \
                .config("spark.executor.memory", "2g")
            
            if os.path.exists(postgresql_jar):
                spark_builder = spark_builder.config("spark.jars", postgresql_jar)
                print(f"✅ Используется драйвер PostgreSQL: {postgresql_jar}")
            else:
                print("⚠️  Драйвер PostgreSQL не найден")
            
            self.spark = spark_builder.getOrCreate()
            
            # Устанавливаем уровень логов
            self.spark.sparkContext.setLogLevel("WARN")
            
            print(f"✅ Spark версии {self.spark.version} запущен")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при запуске Spark: {e}")
            return False
    
    def load_data_from_postgres(self):
        """Загрузка данных из PostgreSQL"""
        print("\nЗагрузка данных из PostgreSQL...")
        
        jdbc_url = "jdbc:postgresql://povt-cluster.tstu.tver.ru:5432/Murder_Mystery"
        
        tables = ['person', 'income', 'drivers_license']
        
        for table in tables:
            print(f"Загрузка {table}...")
            try:
                df = self.spark.read \
                    .format("jdbc") \
                    .option("url", jdbc_url) \
                    .option("dbtable", table) \
                    .option("user", "mpi") \
                    .option("password", "135a1") \
                    .option("driver", "org.postgresql.Driver") \
                    .load()
                
                self.dataframes[table] = df
                print(f"  ✅ Загружено {df.count()} строк")
                
                # Выводим схему для отладки
                print(f"  Столбцы {table}: {', '.join(df.columns)}")
                
            except Exception as e:
                print(f"  ❌ Ошибка загрузки {table}: {e}")
        
        # Проверяем, что все таблицы загружены
        loaded_tables = list(self.dataframes.keys())
        print(f"\n✅ Загружено таблиц: {len(loaded_tables)} из {len(tables)}")
        print(f"Таблицы: {', '.join(loaded_tables)}")
        
        return len(self.dataframes) > 0
    
    def display_results(self, df, description=""):
        """Отображение результатов"""
        print(f"\n{description}")
        print("-" * 50)
        
        if df.count() > 0:
            df.show(10, truncate=False)
            print(f"Всего строк: {df.count()}")
        else:
            print("Нет данных")
    
    def task_2_1(self):
        """Join двух таблиц с сортировкой и агрегацией"""
        print("\nЗадание 2.1: JOIN двух таблиц")
        
        if 'person' not in self.dataframes or 'income' not in self.dataframes:
            print("Необходимые таблицы не загружены")
            return
        
        try:
            result = self.dataframes['person'] \
                .join(self.dataframes['income'], 'ssn') \
                .groupBy('address_street_name') \
                .agg(
                    count('id').alias('person_count'),
                    avg('annual_income').alias('avg_income'),
                    sum('annual_income').alias('total_income')
                ) \
                .orderBy(desc('avg_income'))
            
            self.display_results(result, "Доход по улицам")
        except Exception as e:
            print(f"Ошибка в задании 2.1: {e}")
    
    def task_2_2(self):
        """Join трех таблиц с сортировкой и агрегацией"""
        print("\nЗадание 2.2: JOIN трех таблиц")
        
        required = ['person', 'drivers_license', 'income']
        if not all(t in self.dataframes for t in required):
            print("Не все таблицы загружены")
            return
        
        try:
            # Используем alias для устранения конфликтов имен
            person_alias = self.dataframes['person'].alias('p')
            dl_alias = self.dataframes['drivers_license'].alias('dl')
            income_alias = self.dataframes['income'].alias('i')
            
            result = person_alias \
                .join(dl_alias, col('p.license_id') == col('dl.id')) \
                .join(income_alias, col('p.ssn') == col('i.ssn')) \
                .groupBy('dl.gender', 'dl.car_make') \
                .agg(
                    count('*').alias('total_people'),
                    avg('i.annual_income').alias('avg_income'),
                    avg('dl.age').alias('avg_age')
                ) \
                .orderBy(desc('avg_income'))
            
            self.display_results(result, "Доход по полу и марке авто")
        except Exception as e:
            print(f"Ошибка в задании 2.2: {e}")
    
    def task_2_3(self):
        """Данные по одному объекту по всем таблицам"""
        print("\nЗадание 2.3: Данные об одном человеке")
        
        if 'person' not in self.dataframes:
            print("Таблица person не загружена")
            return
        
        try:
            # Берем первого человека
            person_df = self.dataframes['person']
            person_id = person_df.select(min('id')).collect()[0][0]
            print(f"ID первого человека: {person_id}")
            
            # Получаем данные человека
            person_data = person_df.filter(col('id') == person_id)
            
            # Подготавливаем данные о водительских правах
            if 'drivers_license' in self.dataframes:
                # Используем alias и явно указываем столбцы
                dl_df = self.dataframes['drivers_license']
                # Получаем license_id этого человека
                license_id_val = person_data.select('license_id').collect()[0][0]
                
                if license_id_val:
                    dl_data = dl_df.filter(col('id') == license_id_val)
                    
                    # Выбираем нужные столбцы из drivers_license
                    dl_selected = dl_data.select(
                        col('id').alias('dl_id'),
                        'car_make',
                        'car_model',
                        'car_color',
                        'gender',
                        'age'
                    )
                    
                    # Объединяем с данными человека
                    result = person_data.join(
                        dl_selected,
                        col('license_id') == col('dl_id'),
                        'left'
                    )
                else:
                    result = person_data
                    print("У этого человека нет данных о водительских правах")
            else:
                result = person_data
            
            # Добавляем данные о доходе
            if 'income' in self.dataframes:
                result = result.join(
                    self.dataframes['income'].select('ssn', 'annual_income'),
                    'ssn',
                    'left'
                )
            
            # Выбираем финальные столбцы для отображения
            select_cols = ['id', 'name', 'age', 'address_street_name']
            
            if 'drivers_license' in self.dataframes and 'car_make' in result.columns:
                select_cols.extend(['car_make', 'car_color'])
            
            if 'income' in self.dataframes and 'annual_income' in result.columns:
                select_cols.append('annual_income')
            
            result_final = result.select(*[col for col in select_cols if col in result.columns])
            
            self.display_results(result_final, f"Данные о человеке ID={person_id}")
        except Exception as e:
            print(f"Ошибка в задании 2.3: {e}")
            import traceback
            traceback.print_exc()
    
    def task_2_4(self):
        """Подсчет количества строк по совмещенным данным"""
        print("\nЗадание 2.4: Подсчет строк")
        
        if 'person' not in self.dataframes:
            print("Таблица person не загружена")
            return
        
        try:
            total_person = self.dataframes['person'].count()
            
            # Люди с информацией о доходе
            if 'income' in self.dataframes:
                with_income = self.dataframes['person'] \
                    .join(self.dataframes['income'], 'ssn') \
                    .count()
            else:
                with_income = 0
            
            # Люди с информацией о водительских правах
            if 'drivers_license' in self.dataframes:
                # Используем alias для устранения конфликта имен
                person_alias = self.dataframes['person'].alias('p')
                dl_alias = self.dataframes['drivers_license'].alias('dl')
                
                with_license = person_alias \
                    .join(dl_alias, col('p.license_id') == col('dl.id')) \
                    .count()
            else:
                with_license = 0
            
            # Люди с и доходом, и правами
            if 'income' in self.dataframes and 'drivers_license' in self.dataframes:
                person_alias = self.dataframes['person'].alias('p')
                dl_alias = self.dataframes['drivers_license'].alias('dl')
                income_alias = self.dataframes['income'].alias('i')
                
                with_both = person_alias \
                    .join(dl_alias, col('p.license_id') == col('dl.id')) \
                    .join(income_alias, col('p.ssn') == col('i.ssn')) \
                    .count()
            else:
                with_both = 0
            
            results = [
                ("Всего людей", total_person),
                ("С информацией о доходе", with_income),
                ("С информацией о водительских правах", with_license),
                ("С информацией о доходе и правах", with_both)
            ]
            
            result_df = self.spark.createDataFrame(results, ["Категория", "Количество"])
            self.display_results(result_df, "Статистика")
        except Exception as e:
            print(f"Ошибка в задании 2.4: {e}")
    
    def task_2_5(self):
        """Три сложных анализа с PySpark"""
        print("\nЗадание 2.5: Три анализа")
        
        # Анализ 1: Возрастные группы и доход
        if 'person' in self.dataframes and 'income' in self.dataframes:
            print("\nАнализ 1: Доход по возрасту")
            
            try:
                # Используем возраст из таблицы person
                result1 = self.dataframes['person'].alias('p') \
                    .join(self.dataframes['income'].alias('i'), 'ssn') \
                    .withColumn('age_group',
                               when(col('p.age') < 30, 'До 30')
                              .when((col('p.age') >= 30) & (col('p.age') < 40), '30-39')
                              .when((col('p.age') >= 40) & (col('p.age') < 50), '40-49')
                              .otherwise('50+')) \
                    .groupBy('age_group') \
                    .agg(
                        avg('i.annual_income').alias('avg_income'),
                        count('*').alias('count')
                    ) \
                    .orderBy('age_group')
                
                self.display_results(result1, "Доход по возрастным группам")
            except Exception as e:
                print(f"Ошибка в анализе 1: {e}")
        
        # Анализ 2: Распределение автомобилей по цветам
        if 'drivers_license' in self.dataframes:
            print("\nАнализ 2: Распределение автомобилей по цветам")
            
            try:
                result2 = self.dataframes['drivers_license'] \
                    .groupBy('car_color') \
                    .agg(
                        count('*').alias('количество'),
                        count('car_make').alias('уникальных_марок')  # Просто count, так как groupBy уже группирует
                    ) \
                    .orderBy(desc('количество'))
                
                self.display_results(result2, "Машины по цвету")
            except Exception as e:
                print(f"Ошибка в анализе 2: {e}")
        
        # Анализ 3: Сводная информация о людях
        print("\nАнализ 3: Сводная информация о людях")
        
        try:
            # Начинаем с таблицы person
            base_df = self.dataframes['person'].alias('p')
            
            # Добавляем информацию о доходе
            if 'income' in self.dataframes:
                base_df = base_df.join(
                    self.dataframes['income'].alias('i').select('ssn', 'annual_income'),
                    'ssn',
                    'left'
                )
            
            # Добавляем информацию о водительских правах
            if 'drivers_license' in self.dataframes:
                # Создаем отдельный DataFrame с нужными столбцами
                dl_df = self.dataframes['drivers_license'].alias('dl') \
                    .select(
                        col('id').alias('dl_id'),
                        'car_make',
                        'car_color',
                        'gender'
                    )
                
                base_df = base_df.join(
                    dl_df,
                    col('p.license_id') == col('dl_id'),
                    'left'
                )
            
            # Выбираем столбцы для отображения
            select_cols = ['p.name', 'p.age', 'p.address_street_name']
            
            if 'income' in self.dataframes:
                select_cols.append('annual_income')
            
            if 'drivers_license' in self.dataframes:
                select_cols.extend(['car_make', 'car_color'])
            
            # Фильтруем только существующие столбцы
            existing_cols = [col for col in select_cols if col.split('.')[-1] in base_df.columns]
            
            result3 = base_df.select(*existing_cols) \
                           .orderBy(desc('p.age')) \
                           .limit(10)
            
            self.display_results(result3, "Топ-10 самых старших людей")
        except Exception as e:
            print(f"Ошибка в анализе 3: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self.spark:
            try:
                self.spark.stop()
                print("\n✅ Spark остановлен")
            except:
                pass

def main():
    """Основная функция"""
    print("Анализ данных Murder Mystery с PySpark")
    print("=" * 50)
    
    analyzer = SparkAnalyzer()
    
    if not analyzer.initialize_spark():
        print("\n❌ Не удалось запустить Spark")
        return
    
    if not analyzer.load_data_from_postgres():
        print("\n⚠️  Не все таблицы загружены, но продолжим с тем, что есть...")
    
    print("\n" + "=" * 50)
    print("МЕНЮ:")
    print("1. Все задания (2.1-2.5)")
    print("2. Задание 2.1 (JOIN двух таблиц)")
    print("3. Задание 2.2 (JOIN трех таблиц)")
    print("4. Задание 2.3 (Данные об одном человеке)")
    print("5. Задание 2.4 (Подсчет строк)")
    print("6. Задание 2.5 (Три анализа)")
    print("7. Выход")
    
    while True:
        try:
            choice = input("\nВыберите задание (1-7): ").strip()
            
            if choice == '1':
                analyzer.task_2_1()
                analyzer.task_2_2()
                analyzer.task_2_3()
                analyzer.task_2_4()
                analyzer.task_2_5()
            elif choice == '2':
                analyzer.task_2_1()
            elif choice == '3':
                analyzer.task_2_2()
            elif choice == '4':
                analyzer.task_2_3()
            elif choice == '5':
                analyzer.task_2_4()
            elif choice == '6':
                analyzer.task_2_5()
            elif choice == '7':
                print("Выход...")
                break
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
                
        except KeyboardInterrupt:
            print("\nЗавершение...")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    analyzer.cleanup()
    print("\n✅ Программа завершена")

if __name__ == "__main__":
    main()