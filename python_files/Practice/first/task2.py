#!/usr/bin/env python3
"""
Task 1: SQL запросы к базе данных Murder Mystery с использованием PySpark
Консольное приложение для выполнения SQL запросов с JOIN и агрегацией
"""

import os
import sys
from datetime import datetime
from tabulate import tabulate
import pandas as pd

# Настройка путей для Spark
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
os.environ['SPARK_HOME'] = r'C:\spark\spark-4.1.1-bin-hadoop3'
os.environ['PATH'] = rf'{os.environ["JAVA_HOME"]}\bin;' + os.environ['PATH']
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *


class SparkSQLAnalyzer:
    def __init__(self):
        self.spark = None
        self.connected = False
        self.dataframes = {}
        
    def connect(self, host='povt-cluster.tstu.tver.ru', port=5432, database='Murder_Mystery', 
                user='mpi', password='135a1'):
        """Подключение к базе данных через Spark"""
        try:
            print(f"🚀 Инициализация Spark сессии...")
            
            # Путь к JDBC драйверу
            jdbc_file = r'C:\spark\spark-4.1.1-bin-hadoop3\jars\postgresql-42.7.9.jar'
            
            # Создаем Spark сессию
            self.spark = SparkSession.builder \
                .appName("Murder Mystery Analysis") \
                .master("local[*]") \
                .config("spark.jars", jdbc_file) \
                .config("spark.driver.memory", "2g") \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.driver.host", "localhost") \
                .config("spark.driver.bindAddress", "localhost") \
                .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
                .getOrCreate()
            
            print(f"✅ Spark {self.spark.version} успешно запущен!")
            
            # Настройки подключения к PostgreSQL
            self.jdbc_url = f"jdbc:postgresql://{host}:{port}/{database}"
            self.connection_properties = {
                "user": user,
                "password": password,
                "driver": "org.postgresql.Driver"
            }
            
            print(f"📊 Подключение к базе данных {database} на {host}...")
            
            # Загружаем все таблицы
            tables = ['crime_scene_report', 'drivers_license', 'facebook_event_checkin',
                      'get_fit_now_check_in', 'get_fit_now_member', 'income',
                      'interview', 'person', 'solution']
            
            for table in tables:
                try:
                    print(f"  Загрузка {table}...", end="")
                    df = self.spark.read.jdbc(
                        url=self.jdbc_url,
                        table=table,
                        properties=self.connection_properties
                    )
                    # Кэшируем DataFrame для ускорения
                    df.cache()
                    self.dataframes[table] = df
                    count = df.count()
                    print(f" ✅ {count} строк")
                except Exception as e:
                    print(f"  ⚠️ Ошибка загрузки {table}: {e}")
            
            # Создаем временные представления для SQL запросов
            for table, df in self.dataframes.items():
                df.createOrReplaceTempView(table)
            
            self.connected = True
            print(f"\n✅ Успешно загружено {len(self.dataframes)} таблиц!")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def execute_query(self, query, description="", show_results=True):
        """Выполнение SQL запроса через Spark SQL"""
        if not self.connected:
            print("❌ Нет подключения к Spark!")
            return None
        
        try:
            if description:
                print(f"\n{'='*60}")
                print(f"📝 {description}")
                print(f"{'='*60}")
                print(f"SQL запрос:\n{query}\n")
            
            start_time = datetime.now()
            
            # Выполняем SQL запрос
            result_df = self.spark.sql(query)
            
            # Кэшируем результат
            result_df.cache()
            
            # Получаем количество строк
            result_count = result_df.count()
            
            if show_results:
                print(f"📊 Результат: {result_count} строк")
                
                if result_count > 0:
                    # Конвертируем в pandas для красивого вывода
                    pandas_df = result_df.limit(20).toPandas()
                    print(tabulate(pandas_df, headers='keys', tablefmt='grid', showindex=False))
                    
                    if result_count > 20:
                        print(f"... и еще {result_count - 20} строк")
                else:
                    print("⚠️ Нет данных для отображения")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"⏱️ Время выполнения: {execution_time:.3f} секунд")
            
            return result_df
            
        except Exception as e:
            print(f"❌ Ошибка выполнения запроса: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def task_1_1(self):
        """1. Select c Join по двум таблицам с сортировкой и агрегацией"""
        query = """
        -- Соединение таблиц person и income с агрегацией и сортировкой
        SELECT 
            p.address_street_name,
            COUNT(p.id) as person_count,
            ROUND(AVG(i.annual_income), 2) as avg_income,
            MIN(i.annual_income) as min_income,
            MAX(i.annual_income) as max_income,
            SUM(i.annual_income) as total_income
        FROM person p
        JOIN income i ON p.ssn = i.ssn
        WHERE i.annual_income IS NOT NULL
        GROUP BY p.address_street_name
        HAVING COUNT(p.id) > 1
        ORDER BY avg_income DESC, person_count DESC
        LIMIT 10
        """
        return self.execute_query(query, "1. JOIN двух таблиц с сортировкой и агрегацией")
    
    def task_1_2(self):
        """2. Select c Join по трем таблицам с сортировкой и агрегацией"""
        query = """
        -- Соединение трех таблиц: person, drivers_license, income
        SELECT 
            dl.gender,
            dl.car_make,
            dl.car_model,
            COUNT(*) as total_people,
            ROUND(AVG(i.annual_income), 2) as avg_income,
            ROUND(AVG(dl.age), 1) as avg_age,
            SUM(i.annual_income) as total_group_income
        FROM person p
        JOIN drivers_license dl ON p.license_id = dl.id
        JOIN income i ON p.ssn = i.ssn
        WHERE dl.car_make IS NOT NULL 
          AND i.annual_income IS NOT NULL
          AND dl.age IS NOT NULL
        GROUP BY dl.gender, dl.car_make, dl.car_model
        HAVING COUNT(*) >= 1
        ORDER BY avg_income DESC, total_people DESC
        LIMIT 15
        """
        return self.execute_query(query, "2. JOIN трех таблиц с сортировкой и агрегацией")
    
    def task_1_3(self):
        """3. Данные по одному объекту по всем возможным таблицам"""
        query = """
        -- Все данные о человеке с наименьшим ID из всех связанных таблиц
        SELECT 
            p.id,
            p.name,
            p.address_street_name,
            p.address_number,
            p.ssn,
            dl.age,
            dl.height,
            dl.eye_color,
            dl.hair_color,
            dl.gender,
            dl.plate_number,
            dl.car_make,
            dl.car_model,
            i.annual_income,
            gm.id as membership_id,
            gm.membership_status,
            gm.membership_start_date,
            iv.transcript as interview_transcript,
            fb.event_name,
            fb.date as facebook_checkin_date
        FROM person p
        LEFT JOIN drivers_license dl ON p.license_id = dl.id
        LEFT JOIN income i ON p.ssn = i.ssn
        LEFT JOIN get_fit_now_member gm ON p.id = gm.person_id
        LEFT JOIN interview iv ON p.id = iv.person_id
        LEFT JOIN facebook_event_checkin fb ON p.id = fb.person_id
        WHERE p.id = (SELECT MIN(id) FROM person)
        ORDER BY p.id
        LIMIT 1
        """
        return self.execute_query(query, "3. Все данные об одном человеке (минимальный ID)")
    
    def task_1_4(self):
        """4. Подсчет количества строк по совмещенным данным в 2 таблицах"""
        query = """
        -- Подсчет различных комбинаций
        SELECT 
            'Всего записей в person: ' as description,
            CAST(COUNT(*) AS STRING) as count
        FROM person
        
        UNION ALL
        
        SELECT 
            'Всего записей в income: ',
            CAST(COUNT(*) AS STRING)
        FROM income
        
        UNION ALL
        
        SELECT 
            'Всего записей с доходами (JOIN person и income): ',
            CAST(COUNT(DISTINCT p.id) AS STRING)
        FROM person p
        JOIN income i ON p.ssn = i.ssn
        
        UNION ALL
        
        SELECT 
            'Уникальных людей с водительскими правами: ',
            CAST(COUNT(DISTINCT p.id) AS STRING)
        FROM person p
        JOIN drivers_license dl ON p.license_id = dl.id
        
        UNION ALL
        
        SELECT 
            'Средний доход по улицам (количество улиц): ',
            CAST(COUNT(*) AS STRING)
        FROM (
            SELECT p.address_street_name, AVG(i.annual_income) as avg_income
            FROM person p
            JOIN income i ON p.ssn = i.ssn
            WHERE i.annual_income IS NOT NULL
            GROUP BY p.address_street_name
            HAVING COUNT(p.id) >= 1
        ) as street_incomes
        """
        return self.execute_query(query, "4. Подсчет количества строк по совмещенным данным")
    
    def task_1_5_1(self):
        """5.1 Анализ возрастного распределения с доходами"""
        query = """
        -- Анализ возрастного распределения с доходами
        WITH age_groups AS (
            SELECT 
                CASE 
                    WHEN dl.age < 20 THEN 'До 20 лет'
                    WHEN dl.age BETWEEN 20 AND 29 THEN '20-29 лет'
                    WHEN dl.age BETWEEN 30 AND 39 THEN '30-39 лет'
                    WHEN dl.age BETWEEN 40 AND 49 THEN '40-49 лет'
                    WHEN dl.age >= 50 THEN '50+ лет'
                    ELSE 'Не указан'
                END as age_group,
                dl.gender,
                i.annual_income
            FROM person p
            JOIN drivers_license dl ON p.license_id = dl.id
            JOIN income i ON p.ssn = i.ssn
            WHERE dl.age IS NOT NULL AND i.annual_income IS NOT NULL
        )
        SELECT 
            age_group,
            gender,
            COUNT(*) as person_count,
            ROUND(AVG(annual_income), 2) as avg_income,
            MIN(annual_income) as min_income,
            MAX(annual_income) as max_income,
            SUM(annual_income) as total_income
        FROM age_groups
        GROUP BY age_group, gender
        ORDER BY 
            CASE age_group
                WHEN 'До 20 лет' THEN 1
                WHEN '20-29 лет' THEN 2
                WHEN '30-39 лет' THEN 3
                WHEN '40-49 лет' THEN 4
                WHEN '50+ лет' THEN 5
                ELSE 6
            END,
            gender
        """
        return self.execute_query(query, "5.1 Анализ возрастного распределения с доходами")
    
    def task_1_5_2(self):
        """5.2 Детальный анализ автомобилей по полу и доходу"""
        query = """
        -- Детальный анализ автомобилей по полу и доходу
        WITH car_stats AS (
            SELECT 
                dl.car_make,
                dl.gender,
                i.annual_income,
                dl.age,
                AVG(i.annual_income) OVER (PARTITION BY dl.car_make) as avg_car_income
            FROM person p
            JOIN drivers_license dl ON p.license_id = dl.id
            JOIN income i ON p.ssn = i.ssn
            WHERE dl.car_make IS NOT NULL 
              AND i.annual_income IS NOT NULL
        )
        SELECT 
            car_make,
            gender,
            COUNT(*) as total_owners,
            ROUND(AVG(annual_income), 2) as avg_owner_income,
            MIN(annual_income) as min_owner_income,
            MAX(annual_income) as max_owner_income,
            ROUND(AVG(age), 1) as avg_owner_age,
            ROUND(AVG(avg_car_income), 2) as avg_car_make_income,
            CASE 
                WHEN AVG(annual_income) > AVG(avg_car_income) THEN 'Выше среднего по марке'
                WHEN AVG(annual_income) < AVG(avg_car_income) THEN 'Ниже среднего по марке'
                ELSE 'Соответствует среднему по марке'
            END as income_comparison
        FROM car_stats
        GROUP BY car_make, gender
        HAVING COUNT(*) >= 1
        ORDER BY avg_owner_income DESC, total_owners DESC
        LIMIT 15
        """
        return self.execute_query(query, "5.2 Детальный анализ автомобилей по полу и доходу")
    
    def task_1_5_3(self):
        """5.3 Комплексный анализ членов спортзала"""
        query = """
        -- Комплексный анализ членов спортзала
        WITH gym_members AS (
            SELECT 
                p.name,
                p.id,
                gm.membership_status,
                gm.membership_start_date,
                i.annual_income,
                dl.gender,
                dl.age,
                dl.height,
                dl.car_make,
                ROW_NUMBER() OVER (PARTITION BY gm.membership_status ORDER BY i.annual_income DESC NULLS LAST) as income_rank
            FROM person p
            JOIN get_fit_now_member gm ON p.id = gm.person_id
            LEFT JOIN income i ON p.ssn = i.ssn
            LEFT JOIN drivers_license dl ON p.license_id = dl.id
            WHERE gm.membership_status IS NOT NULL
        ),
        gym_stats AS (
            SELECT 
                membership_status,
                COUNT(*) as total_members,
                COUNT(DISTINCT gender) as gender_count,
                ROUND(AVG(age), 1) as avg_age,
                ROUND(AVG(height), 1) as avg_height,
                ROUND(AVG(annual_income), 2) as avg_income,
                COUNT(DISTINCT car_make) as unique_car_brands,
                CONCAT_WS(', ', COLLECT_SET(COALESCE(car_make, 'Нет авто'))) as car_brands
            FROM gym_members
            GROUP BY membership_status
        )
        SELECT 
            gs.membership_status,
            gs.total_members,
            gs.avg_age,
            gs.avg_height,
            gs.avg_income,
            gs.unique_car_brands,
            gs.car_brands,
            gm_top.name as top_earner_name,
            gm_top.annual_income as top_earner_income
        FROM gym_stats gs
        LEFT JOIN (
            SELECT membership_status, name, annual_income
            FROM gym_members 
            WHERE income_rank = 1
        ) gm_top ON gs.membership_status = gm_top.membership_status
        ORDER BY gs.avg_income DESC NULLS LAST
        """
        return self.execute_query(query, "5.3 Комплексный анализ членов спортзала")
    
    def task_1_5(self):
        """5. Три сложных SELECT запроса"""
        print("\n" + "="*60)
        print("5. ТРИ СЛОЖНЫХ SELECT ЗАПРОСА")
        print("="*60)
        
        self.task_1_5_1()
        self.task_1_5_2()
        self.task_1_5_3()
    
    def show_table_info(self):
        """Показать информацию о таблицах"""
        if not self.connected:
            print("❌ Нет подключения к Spark!")
            return
        
        print("\n" + "="*60)
        print("📊 ИНФОРМАЦИЯ О ТАБЛИЦАХ")
        print("="*60)
        
        for table_name, df in self.dataframes.items():
            print(f"\n📋 Таблица: {table_name}")
            print(f"   Строк: {df.count()}")
            print(f"   Колонки: {', '.join(df.columns)}")
            print(f"   Схема:")
            df.printSchema()
    
    def show_sample_data(self):
        """Показать примеры данных из таблиц"""
        tables = ['crime_scene_report', 'drivers_license', 'person', 'income', 
                  'interview', 'get_fit_now_member', 'get_fit_now_check_in']
        
        print("\n" + "="*60)
        print("📊 ПРИМЕРЫ ДАННЫХ ИЗ ТАБЛИЦ")
        print("="*60)
        
        for table in tables:
            if table in self.dataframes:
                query = f"SELECT * FROM {table} LIMIT 3"
                self.execute_query(query, f"Данные из таблицы {table} (первые 3 строки)")
    
    def run_all_tasks(self):
        """Выполнить все задания"""
        print("🚀 ЗАПУСК ВСЕХ SQL ЗАДАНИЙ (PySpark)\n")
        
        # Показываем информацию о таблицах
        self.show_table_info()
        
        # Выполняем задания
        self.task_1_1()
        self.task_1_2()
        self.task_1_3()
        self.task_1_4()
        self.task_1_5()
        
        print("\n" + "="*60)
        print("✅ ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
        print("="*60)
    
    def close(self):
        """Закрытие соединения"""
        if self.spark:
            self.spark.stop()
        print("\n🔌 Spark сессия закрыта")


def main():
    """Основная функция"""
    print("="*60)
    print("TASK 1: SQL ЗАПРОСЫ К БАЗЕ ДАННЫХ MURDER MYSTERY")
    print("="*60)
    print("🚀 Версия с PySpark\n")
    
    analyzer = SparkSQLAnalyzer()
    
    # Параметры подключения
    connection_params = {
        'host': 'povt-cluster.tstu.tver.ru',
        'port': 5432,
        'database': 'Murder_Mystery',
        'user': 'mpi',
        'password': '135a1'
    }
    
    print(f"Параметры подключения:")
    print(f"  Хост: {connection_params['host']}")
    print(f"  Порт: {connection_params['port']}")
    print(f"  База данных: {connection_params['database']}")
    print(f"  Пользователь: {connection_params['user']}")
    print("-" * 60)
    
    # Подключаемся к базе
    if not analyzer.connect(**connection_params):
        print("\n❌ Не удалось подключиться к базе данных.")
        return
    
    try:
        # Меню выбора
        while True:
            print("\n" + "-"*60)
            print("МЕНЮ SQL ЗАПРОСОВ (PySpark):")
            print("1. Выполнить все задания (1-5)")
            print("2. JOIN двух таблиц с сортировкой и агрегацией")
            print("3. JOIN трех таблиц с сортировкой и агрегацией")
            print("4. Все данные об одном человеке")
            print("5. Подсчет количества строк")
            print("6. Три сложных SELECT запроса")
            print("7. Показать информацию о таблицах")
            print("8. Показать примеры данных из таблиц")
            print("9. Выход")
            
            choice = input("\nВыберите опцию (1-9): ").strip()
            
            if choice == '1':
                analyzer.run_all_tasks()
            elif choice == '2':
                analyzer.task_1_1()
            elif choice == '3':
                analyzer.task_1_2()
            elif choice == '4':
                analyzer.task_1_3()
            elif choice == '5':
                analyzer.task_1_4()
            elif choice == '6':
                analyzer.task_1_5()
            elif choice == '7':
                analyzer.show_table_info()
            elif choice == '8':
                analyzer.show_sample_data()
            elif choice == '9':
                print("\n👋 Завершение работы...")
                break
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
                
    except KeyboardInterrupt:
        print("\n\n⏹️ Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()