#!/usr/bin/env python3
"""
Task 1: SQL запросы к базе данных Murder Mystery
Консольное приложение для выполнения SQL запросов с JOIN и агрегацией
"""

import sys
import psycopg2
import pandas as pd
from tabulate import tabulate
from datetime import datetime

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
            print("✅ Подключение успешно!")
            
            # Получаем список таблиц
            self.cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in self.cursor.fetchall()]
            print(f"📊 Найдено таблиц: {len(tables)}")
            if tables:
                print(f"📋 Таблицы: {', '.join(tables)}")
            else:
                print("⚠️ Таблицы не найдены!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def execute_query(self, query, description="", show_results=True, return_df=False):
        """Выполнение SQL запроса"""
        if not self.connected:
            print("❌ Нет подключения к базе данных!")
            return None
        
        try:
            if description:
                print(f"\n{'='*60}")
                print(f"📝 {description}")
                print(f"{'='*60}")
                print(f"SQL запрос:\n{query}\n")
            
            start_time = datetime.now()
            self.cursor.execute(query)
            
            if show_results:
                # Получаем результаты
                results = self.cursor.fetchall()
                columns = [desc[0] for desc in self.cursor.description]
                
                # Выводим количество строк
                print(f"📊 Результат: {len(results)} строк")
                
                # Выводим результаты в виде таблицы
                if results:
                    # Ограничиваем вывод для больших результатов
                    display_results = results[:20] if len(results) > 20 else results
                    df = pd.DataFrame(display_results, columns=columns)
                    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
                    
                    if len(results) > 20:
                        print(f"... и еще {len(results) - 20} строк")
                else:
                    print("⚠️ Нет данных для отображения")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"⏱️ Время выполнения: {execution_time:.3f} секунд")
            
            if return_df:
                # Возвращаем DataFrame
                results = self.cursor.fetchall()
                columns = [desc[0] for desc in self.cursor.description]
                return pd.DataFrame(results, columns=columns)
                
            return True
            
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
            p.city,
            COUNT(p.id) as person_count,
            AVG(i.annual_income) as avg_income,
            MIN(i.annual_income) as min_income,
            MAX(i.annual_income) as max_income,
            SUM(i.annual_income) as total_income
        FROM person p
        JOIN income i ON p.ssn = i.ssn
        WHERE i.annual_income IS NOT NULL
        GROUP BY p.city
        HAVING COUNT(p.id) > 1
        ORDER BY avg_income DESC, person_count DESC
        LIMIT 10;
        """
        return self.execute_query(query, "1. JOIN двух таблиц с сортировкой и агрегацией")
    
    def task_1_2(self):
        """2. Select c Join по трем таблицам с сортировкой и агрегацией"""
        query = """
        -- Соединение трех таблиц: person, drivers_license, income
        SELECT 
            p.gender,
            dl.car_make,
            dl.car_model,
            COUNT(*) as total_people,
            AVG(i.annual_income) as avg_income,
            AVG(p.age) as avg_age,
            SUM(i.annual_income) as total_group_income
        FROM person p
        JOIN drivers_license dl ON p.license_id = dl.id
        JOIN income i ON p.ssn = i.ssn
        WHERE dl.car_make IS NOT NULL 
          AND i.annual_income IS NOT NULL
          AND p.age IS NOT NULL
        GROUP BY p.gender, dl.car_make, dl.car_model
        HAVING COUNT(*) >= 1
        ORDER BY avg_income DESC, total_people DESC
        LIMIT 15;
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
            p.age,
            p.height,
            p.weight,
            p.eye_color,
            p.hair_color,
            p.gender,
            dl.plate_number,
            dl.car_make,
            dl.car_model,
            dl.car_color,
            dl.gender as license_gender,
            i.annual_income,
            i.salary_frequency,
            gm.membership_status,
            gm.membership_start_date,
            i2.transcript as interview_transcript
        FROM person p
        LEFT JOIN drivers_license dl ON p.license_id = dl.id
        LEFT JOIN income i ON p.ssn = i.ssn
        LEFT JOIN get_fit_now_member gm ON p.id = gm.person_id
        LEFT JOIN interview i2 ON p.id = i2.person_id
        WHERE p.id = (SELECT MIN(id) FROM person)
        ORDER BY p.id;
        """
        return self.execute_query(query, "3. Все данные об одном человеке (минимальный ID)")
    
    def task_1_4(self):
        """4. Подсчет количества строк по совмещенным данным в 2 таблицах"""
        query = """
        -- Подсчет различных комбинаций
        SELECT 
            'Всего записей в person: ' as description,
            COUNT(*) as count
        FROM person
        
        UNION ALL
        
        SELECT 
            'Всего записей в income: ',
            COUNT(*)
        FROM income
        
        UNION ALL
        
        SELECT 
            'Всего записей с доходами (JOIN person и income): ',
            COUNT(DISTINCT p.id)
        FROM person p
        JOIN income i ON p.ssn = i.ssn
        
        UNION ALL
        
        SELECT 
            'Уникальных людей с водительскими правами: ',
            COUNT(DISTINCT p.id)
        FROM person p
        JOIN drivers_license dl ON p.license_id = dl.id
        
        UNION ALL
        
        SELECT 
            'Средний доход по городам (количество городов): ',
            COUNT(*)
        FROM (
            SELECT p.city, AVG(i.annual_income) as avg_income
            FROM person p
            JOIN income i ON p.ssn = i.ssn
            WHERE i.annual_income IS NOT NULL
            GROUP BY p.city
            HAVING COUNT(p.id) >= 1
        ) as city_incomes;
        """
        return self.execute_query(query, "4. Подсчет количества строк по совмещенным данным")
    
    def task_1_5(self):
        """5. Три сложных Select запроса"""
        
        print("\n" + "="*60)
        print("5. ТРИ СЛОЖНЫХ SELECT ЗАПРОСА")
        print("="*60)
        
        # Запрос 5.1: Анализ возрастного распределения с доходами
        query1 = """
        -- Анализ возрастного распределения с доходами
        WITH age_groups AS (
            SELECT 
                CASE 
                    WHEN age < 20 THEN 'До 20 лет'
                    WHEN age BETWEEN 20 AND 29 THEN '20-29 лет'
                    WHEN age BETWEEN 30 AND 39 THEN '30-39 лет'
                    WHEN age BETWEEN 40 AND 49 THEN '40-49 лет'
                    WHEN age >= 50 THEN '50+ лет'
                    ELSE 'Не указан'
                END as age_group,
                gender,
                annual_income
            FROM person p
            JOIN income i ON p.ssn = i.ssn
            WHERE age IS NOT NULL AND annual_income IS NOT NULL
        )
        SELECT 
            age_group,
            gender,
            COUNT(*) as person_count,
            ROUND(AVG(annual_income), 2) as avg_income,
            ROUND(MIN(annual_income), 2) as min_income,
            ROUND(MAX(annual_income), 2) as max_income,
            ROUND(SUM(annual_income), 2) as total_income
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
            gender;
        """
        self.execute_query(query1, "5.1 Анализ возрастного распределения с доходами")
        
        # Запрос 5.2: Анализ автомобилей по полу и доходу
        query2 = """
        -- Детальный анализ автомобилей по полу и доходу
        WITH car_analysis AS (
            SELECT 
                dl.car_make,
                p.gender,
                i.annual_income,
                p.age,
                COUNT(*) OVER (PARTITION BY dl.car_make, p.gender) as car_gender_count,
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
            ROUND(MIN(annual_income), 2) as min_owner_income,
            ROUND(MAX(annual_income), 2) as max_owner_income,
            ROUND(AVG(age), 1) as avg_owner_age,
            ROUND(avg_car_income, 2) as avg_car_make_income,
            CASE 
                WHEN AVG(annual_income) > avg_car_income THEN 'Выше среднего по марке'
                WHEN AVG(annual_income) < avg_car_income THEN 'Ниже среднего по марке'
                ELSE 'Соответствует среднему по марке'
            END as income_comparison
        FROM car_analysis
        GROUP BY car_make, gender, avg_car_income
        HAVING COUNT(*) >= 1
        ORDER BY avg_owner_income DESC, total_owners DESC
        LIMIT 15;
        """
        self.execute_query(query2, "5.2 Детальный анализ автомобилей по полу и доходу")
        
        # Запрос 5.3: Анализ членов спортзала и их характеристик
        query3 = """
        -- Комплексный анализ членов спортзала
        WITH gym_members AS (
            SELECT 
                p.*,
                gm.membership_status,
                gm.membership_start_date,
                i.annual_income,
                dl.car_make,
                dl.car_model,
                ROW_NUMBER() OVER (PARTITION BY gm.membership_status ORDER BY i.annual_income DESC) as income_rank
            FROM person p
            LEFT JOIN get_fit_now_member gm ON p.id = gm.person_id
            LEFT JOIN income i ON p.ssn = i.ssn
            LEFT JOIN drivers_license dl ON p.license_id = dl.id
        ),
        gym_stats AS (
            SELECT 
                membership_status,
                COUNT(*) as total_members,
                COUNT(DISTINCT gender) as gender_count,
                ROUND(AVG(age), 1) as avg_age,
                ROUND(AVG(height), 1) as avg_height,
                ROUND(AVG(weight), 1) as avg_weight,
                ROUND(AVG(annual_income), 2) as avg_income,
                COUNT(DISTINCT car_make) as unique_car_brands,
                STRING_AGG(DISTINCT car_make, ', ') as car_brands
            FROM gym_members
            WHERE membership_status IS NOT NULL
            GROUP BY membership_status
        )
        SELECT 
            gs.membership_status,
            gs.total_members,
            gs.avg_age,
            gs.avg_height,
            gs.avg_weight,
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
        ORDER BY gs.avg_income DESC;
        """
        self.execute_query(query3, "5.3 Комплексный анализ членов спортзала")
    
    def show_table_info(self):
        """Показать информацию о таблицах"""
        if not self.connected:
            print("❌ Нет подключения к базе данных!")
            return
        
        query = """
        SELECT 
            t.table_name,
            COUNT(c.column_name) as column_count,
            STRING_AGG(c.column_name || ' (' || c.data_type || ')', ', ' 
                       ORDER BY c.ordinal_position) as columns_info
        FROM information_schema.tables t
        JOIN information_schema.columns c ON t.table_name = c.table_name
        WHERE t.table_schema = 'public'
        GROUP BY t.table_name
        ORDER BY t.table_name
        LIMIT 10;
        """
        
        self.execute_query(query, "Информация о таблицах базы данных")
    
    def run_all_tasks(self):
        """Выполнить все задания"""
        print("🚀 ЗАПУСК ВСЕХ SQL ЗАДАНИЙ\n")
        
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
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("\n🔌 Соединение с базой данных закрыто")


def main():
    """Основная функция"""
    print("="*60)
    print("TASK 1: SQL ЗАПРОСЫ К БАЗЕ ДАННЫХ MURDER MYSTERY")
    print("="*60)
    
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
    
    # Подключаемся к базе
    if not analyzer.connect(**connection_params):
        print("\n❌ Не удалось подключиться к базе данных.")
        print("Проверьте:")
        print("1. Доступность хоста povt-cluster.tstu.tver.ru")
        print("2. Правильность логина и пароля")
        print("3. Наличие базы данных 'murder_mystery'")
        print("4. Разрешения для пользователя 'mpi'")
        return
    
    try:
        # Меню выбора
        while True:
            print("\n" + "-"*60)
            print("МЕНЮ SQL ЗАПРОСОВ:")
            print("1. Выполнить все задания (1-5)")
            print("2. JOIN двух таблиц с сортировкой и агрегацией")
            print("3. JOIN трех таблиц с сортировкой и агрегацией")
            print("4. Все данные об одном человеке")
            print("5. Подсчет количества строк")
            print("6. Три сложных SELECT запроса")
            print("7. Показать информацию о таблицах")
            print("8. Выход")
            
            choice = input("\nВыберите опцию (1-8): ").strip()
            
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
                print("\n👋 Завершение работы...")
                break
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
                
    except KeyboardInterrupt:
        print("\n\n⏹️ Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()