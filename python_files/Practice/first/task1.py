import sys
import psycopg2
import pandas as pd
from tabulate import tabulate
from datetime import datetime

from sql_queries import (
    QUERY_1_1, QUERY_1_2, QUERY_1_3, QUERY_1_4,
    QUERY_1_5_1, QUERY_1_5_2, QUERY_1_5_3,
    QUERY_TABLE_INFO, QUERY_DESCRIPTIONS, get_sample_query
)

class SQLAnalyzer:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.connected = False
        
    def connect(self, host='povt-cluster.tstu.tver.ru', port=5432, database='Murder_Mystery', 
                user='mpi', password='135a1'):
        """Подключение к базе данных"""
        try:
            print(f"Подключение к базе данных {database} на {host}...")
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            self.cursor = self.connection.cursor()
            self.connected = True
            print("Подключение успешно!")
            
            self.cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in self.cursor.fetchall()]
            print(f" Найдено таблиц: {len(tables)}")
            if tables:
                print(f" Таблицы: {', '.join(tables)}")
            else:
                print("Таблицы не найдены!")
            return True
            
        except Exception as e:
            print(f" Ошибка подключения: {e}")
            return False
    
    def execute_query(self, query, description="", show_results=True):
        """Выполнение SQL запроса"""
        if not self.connected:
            print(" Нет подключения к базе данных!")
            return None
        
        try:
            self.connection.rollback()
            
            if description:
                print(f" {description}")

                print(f"SQL запрос:\n{query}\n")
            
            start_time = datetime.now()
            self.cursor.execute(query)
            
            if show_results and self.cursor.description:
                # Получаем результаты
                results = self.cursor.fetchall()
                columns = [desc[0] for desc in self.cursor.description]
                
                # Выводим количество строк
                print(f" Результат: {len(results)} строк")
                
                # Выводим результаты в виде таблицы
                if results:
                    # Ограничиваем вывод для больших результатов
                    display_results = results[:20] if len(results) > 20 else results
                    df = pd.DataFrame(display_results, columns=columns)
                    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
                    
                    if len(results) > 20:
                        print(f"... и еще {len(results) - 20} строк")
                else:
                    print(" Нет данных для отображения")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f" Время выполнения: {execution_time:.3f} секунд")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"Ошибка выполнения запроса: {e}")
            self.connection.rollback() 
            return None
    
    def task_1_1(self):
        """1. Select c Join по двум таблицам с сортировкой и агрегацией"""
        return self.execute_query(QUERY_1_1, QUERY_DESCRIPTIONS['1.1'])
    
    def task_1_2(self):
        """2. Select c Join по трем таблицам с сортировкой и агрегацией"""
        return self.execute_query(QUERY_1_2, QUERY_DESCRIPTIONS['1.2'])
    
    def task_1_3(self):
        """3. Данные по одному объекту по всем возможным таблицам"""
        return self.execute_query(QUERY_1_3, QUERY_DESCRIPTIONS['1.3'])
    
    def task_1_4(self):
        """4. Подсчет количества строк по совмещенным данным в 2 таблицах"""
        return self.execute_query(QUERY_1_4, QUERY_DESCRIPTIONS['1.4'])
    
    def task_1_5_1(self):
        """5.1 Анализ возрастного распределения с доходами"""
        return self.execute_query(QUERY_1_5_1, QUERY_DESCRIPTIONS['1.5.1'])
    
    def task_1_5_2(self):
        """5.2 Детальный анализ автомобилей по полу и доходу"""
        return self.execute_query(QUERY_1_5_2, QUERY_DESCRIPTIONS['1.5.2'])
    
    def task_1_5_3(self):
        """5.3 Комплексный анализ членов спортзала"""
        return self.execute_query(QUERY_1_5_3, QUERY_DESCRIPTIONS['1.5.3'])
    
    def task_1_5(self):
        """5. Три сложных SELECT запроса"""
        print("5. ТРИ СЛОЖНЫХ SELECT ЗАПРОСА")
        
        self.task_1_5_1()
        self.task_1_5_2()
        self.task_1_5_3()
    
    def show_table_info(self):
        """Показать информацию о таблицах"""
        if not self.connected:
            print("Нет подключения к базе данных!")
            return
        
        self.execute_query(QUERY_TABLE_INFO, QUERY_DESCRIPTIONS['table_info'])
    
    def show_sample_data(self):
        """Показать примеры данных из таблиц"""
        tables = ['crime_scene_report', 'drivers_license', 'person', 'income', 
                  'interview', 'get_fit_now_member', 'get_fit_now_check_in']
        
        print("ПРИМЕРЫ ДАННЫХ ИЗ ТАБЛИЦ")
        
        for table in tables:
            query = get_sample_query(table, 3)
            self.execute_query(query, f"Данные из таблицы {table} (первые 3 строки)")
    
    def run_all_tasks(self):
        """Выполнить все задания"""
        print("ЗАПУСК ВСЕХ SQL ЗАДАНИЙ\n")
        
        self.show_table_info()
        
        self.task_1_1()
        self.task_1_2()
        self.task_1_3()
        self.task_1_4()
        self.task_1_5()
        
        print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
    
    def close(self):
        """Закрытие соединения"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.commit()
            self.connection.close()
        print("\n🔌 Соединение с базой данных закрыто")


def main():
    """Основная функция"""
    print("TASK 1: SQL ЗАПРОСЫ К БАЗЕ ДАННЫХ MURDER MYSTERY")
    print("Реализация: psycopg2 (прямое подключение к PostgreSQL)")
    print("-" * 60)
    
    analyzer = SQLAnalyzer()
    
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
    
    if not analyzer.connect(**connection_params):
        print("\n Не удалось подключиться к базе данных.")
        return
    
    try:
        while True:
            print("МЕНЮ SQL ЗАПРОСОВ:")
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