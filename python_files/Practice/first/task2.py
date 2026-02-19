import os
import sys
from datetime import datetime
from tabulate import tabulate
import pandas as pd

from sql_queries import (
    QUERY_1_1, QUERY_1_2, QUERY_1_3, QUERY_1_4,
    QUERY_1_5_1, QUERY_1_5_2, QUERY_1_5_3,
    QUERY_DESCRIPTIONS, get_sample_query
)

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
            print(f" Инициализация Spark сессии...")
            
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
            
            print(f"Spark {self.spark.version} успешно запущен!")
            
            # Настройки подключения к PostgreSQL
            self.jdbc_url = f"jdbc:postgresql://{host}:{port}/{database}"
            self.connection_properties = {
                "user": user,
                "password": password,
                "driver": "org.postgresql.Driver"
            }
            
            print(f"Подключение к базе данных {database} на {host}...")
            
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
                    print(f" {count} строк")
                except Exception as e:
                    print(f" Ошибка загрузки {table}: {e}")
            
            # Создаем временные представления для SQL запросов
            for table, df in self.dataframes.items():
                df.createOrReplaceTempView(table)
            
            self.connected = True
            print(f"\n Успешно загружено {len(self.dataframes)} таблиц!")
            
            return True
            
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def execute_query(self, query, description="", show_results=True):
        """Выполнение SQL запроса через Spark SQL"""
        if not self.connected:
            print("Нет подключения к Spark!")
            return None
        
        try:
            if description:
                print(f"{description}")
            
            start_time = datetime.now()
            
            result_df = self.spark.sql(query)
            
            result_df.cache()
            
            result_count = result_df.count()
            
            if show_results:
                print(f"Результат: {result_count} строк")
                
                if result_count > 0:
                    # Конвертируем в pandas для красивого вывода
                    pandas_df = result_df.limit(20).toPandas()
                    print(tabulate(pandas_df, headers='keys', tablefmt='grid', showindex=False))
                    
                    if result_count > 20:
                        print(f"... и еще {result_count - 20} строк")
                else:
                    print("Нет данных для отображения")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"Время выполнения: {execution_time:.3f} секунд")
            
            return result_df
            
        except Exception as e:
            print(f" Ошибка выполнения запроса: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def task_1_1(self):
        """1. Select c Join по двум таблицам с сортировкой и агрегацией"""
        spark_query = QUERY_1_1.replace('::numeric', '')
        return self.execute_query(spark_query, QUERY_DESCRIPTIONS['1.1'])
    
    def task_1_2(self):
        """2. Select c Join по трем таблицам с сортировкой и агрегацией"""
        spark_query = QUERY_1_2.replace('::numeric', '')
        return self.execute_query(spark_query, QUERY_DESCRIPTIONS['1.2'])
    
    def task_1_3(self):
        """3. Данные по одному объекту по всем возможным таблицам"""
        return self.execute_query(QUERY_1_3, QUERY_DESCRIPTIONS['1.3'])
    
    def task_1_4(self):
        """4. Подсчет количества строк по совмещенным данным в 2 таблицах"""
        spark_query = QUERY_1_4.replace('::text', '')
        spark_query = spark_query.replace('COUNT(*)::text', 'CAST(COUNT(*) AS STRING)')
        spark_query = spark_query.replace('COUNT(DISTINCT p.id)::text', 'CAST(COUNT(DISTINCT p.id) AS STRING)')
        return self.execute_query(spark_query, QUERY_DESCRIPTIONS['1.4'])
    
    def task_1_5_1(self):
        """5.1 Анализ возрастного распределения с доходами"""
        spark_query = QUERY_1_5_1.replace('::numeric', '')
        return self.execute_query(spark_query, QUERY_DESCRIPTIONS['1.5.1'])
    
    def task_1_5_2(self):
        """5.2 Детальный анализ автомобилей по полу и доходу"""
        spark_query = QUERY_1_5_2.replace('::numeric', '')
        return self.execute_query(spark_query, QUERY_DESCRIPTIONS['1.5.2'])
    
    def task_1_5_3(self):
        """5.3 Комплексный анализ членов спортзала"""
        spark_query = QUERY_1_5_3
        spark_query = spark_query.replace('STRING_AGG(DISTINCT COALESCE(dl.car_make, \'Нет авто\'), \', \')', 
                                         'CONCAT_WS(\', \', COLLECT_SET(COALESCE(dl.car_make, \'Нет авто\')))')
        spark_query = spark_query.replace('::numeric', '')
        return self.execute_query(spark_query, QUERY_DESCRIPTIONS['1.5.3'])
    
    def task_1_5(self):
        """5. Три сложных SELECT запроса"""
        print("5. ТРИ СЛОЖНЫХ SELECT ЗАПРОСА")
        
        self.task_1_5_1()
        self.task_1_5_2()
        self.task_1_5_3()
    
    def show_table_info(self):
        """Показать информацию о таблицах"""
        if not self.connected:
            print("Нет подключения к Spark!")
            return
        
        print(" ИНФОРМАЦИЯ О ТАБЛИЦАХ")
        
        for table_name, df in self.dataframes.items():
            print(f"\nТаблица: {table_name}")
            print(f"   Строк: {df.count()}")
            print(f"   Колонки: {', '.join(df.columns)}")
            print(f"   Схема:")
            df.printSchema()
    
    def show_sample_data(self):
        """Показать примеры данных из таблиц"""
        tables = ['crime_scene_report', 'drivers_license', 'person', 'income', 
                  'interview', 'get_fit_now_member', 'get_fit_now_check_in']
        
        print(" ПРИМЕРЫ ДАННЫХ ИЗ ТАБЛИЦ")
        
        for table in tables:
            if table in self.dataframes:
                query = get_sample_query(table, 3)
                self.execute_query(query, f"Данные из таблицы {table} (первые 3 строки)")
    
    def run_all_tasks(self):
        """Выполнить все задания"""
        print("ЗАПУСК ВСЕХ SQL ЗАДАНИЙ (PySpark)\n")
        
        self.show_table_info()
        
        self.task_1_1()
        self.task_1_2()
        self.task_1_3()
        self.task_1_4()
        self.task_1_5()
        
        print(" ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
    
    def close(self):
        """Закрытие соединения"""
        if self.spark:
            self.spark.stop()
        print("\nSpark сессия закрыта")


def main():
    """Основная функция"""

    analyzer = SparkSQLAnalyzer()
    
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
    
    if not analyzer.connect(**connection_params):
        print("\n Не удалось подключиться к базе данных.")
        return
    
    try:
        while True:
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
                print("\n Завершение работы...")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")
                
    except KeyboardInterrupt:
        print("\n\n Программа прервана пользователем")
    except Exception as e:
        print(f"\n Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()