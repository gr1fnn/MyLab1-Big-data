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
    
    def execute_query(self, query, description="", show_results=True):
        """Выполнение SQL запроса"""
        if not self.connected:
            print("❌ Нет подключения к базе данных!")
            return None
        
        try:
            # Откатываем предыдущую транзакцию, если была ошибка
            self.connection.rollback()
            
            if description:
                print(f"\n{'='*60}")
                print(f"📝 {description}")
                print(f"{'='*60}")
                print(f"SQL запрос:\n{query}\n")
            
            start_time = datetime.now()
            self.cursor.execute(query)
            
            if show_results and self.cursor.description:
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
            
            # Фиксируем транзакцию
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка выполнения запроса: {e}")
            self.connection.rollback()  # Откатываем транзакцию при ошибке
            return None
    
    def task_1_1(self):
        """1. Select c Join по двум таблицам с сортировкой и агрегацией"""
        query = """
        -- Соединение таблиц person и income с агрегацией и сортировкой
        SELECT 
            p.address_street_name,
            COUNT(p.id) as person_count,
            ROUND(AVG(i.annual_income)::numeric, 2) as avg_income,
            MIN(i.annual_income) as min_income,
            MAX(i.annual_income) as max_income,
            SUM(i.annual_income) as total_income
        FROM person p
        JOIN income i ON p.ssn = i.ssn
        WHERE i.annual_income IS NOT NULL
        GROUP BY p.address_street_name
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
            dl.gender,
            dl.car_make,
            dl.car_model,
            COUNT(*) as total_people,
            ROUND(AVG(i.annual_income)::numeric, 2) as avg_income,
            ROUND(AVG(dl.age)::numeric, 1) as avg_age,
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
        LIMIT 1;
        """
        return self.execute_query(query, "3. Все данные об одном человеке (минимальный ID)")
    
    def task_1_4(self):
        """4. Подсчет количества строк по совмещенным данным в 2 таблицах"""
        query = """
        -- Подсчет различных комбинаций
        SELECT 
            'Всего записей в person: ' as description,
            COUNT(*)::text as count
        FROM person
        
        UNION ALL
        
        SELECT 
            'Всего записей в income: ',
            COUNT(*)::text
        FROM income
        
        UNION ALL
        
        SELECT 
            'Всего записей с доходами (JOIN person и income): ',
            COUNT(DISTINCT p.id)::text
        FROM person p
        JOIN income i ON p.ssn = i.ssn
        
        UNION ALL
        
        SELECT 
            'Уникальных людей с водительскими правами: ',
            COUNT(DISTINCT p.id)::text
        FROM person p
        JOIN drivers_license dl ON p.license_id = dl.id
        
        UNION ALL
        
        SELECT 
            'Средний доход по улицам (количество улиц): ',
            COUNT(*)::text
        FROM (
            SELECT p.address_street_name, AVG(i.annual_income) as avg_income
            FROM person p
            JOIN income i ON p.ssn = i.ssn
            WHERE i.annual_income IS NOT NULL
            GROUP BY p.address_street_name
            HAVING COUNT(p.id) >= 1
        ) as street_incomes;
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
            ROUND(AVG(annual_income)::numeric, 2) as avg_income,
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
            gender;
        """
        return self.execute_query(query, "5.1 Анализ возрастного распределения с доходами")
    
    def task_1_5_2(self):
        """5.2 Детальный анализ автомобилей по полу и доходу"""
        query = """
        -- Детальный анализ автомобилей по полу и доходу
        SELECT 
            dl.car_make,
            dl.gender,
            COUNT(*) as total_owners,
            ROUND(AVG(i.annual_income)::numeric, 2) as avg_owner_income,
            MIN(i.annual_income) as min_owner_income,
            MAX(i.annual_income) as max_owner_income,
            ROUND(AVG(dl.age)::numeric, 1) as avg_owner_age,
            ROUND(AVG(AVG(i.annual_income)) OVER (PARTITION BY dl.car_make)::numeric, 2) as avg_car_make_income,
            CASE 
                WHEN AVG(i.annual_income) > AVG(AVG(i.annual_income)) OVER (PARTITION BY dl.car_make) 
                    THEN 'Выше среднего по марке'
                WHEN AVG(i.annual_income) < AVG(AVG(i.annual_income)) OVER (PARTITION BY dl.car_make) 
                    THEN 'Ниже среднего по марке'
                ELSE 'Соответствует среднему по марке'
            END as income_comparison
        FROM person p
        JOIN drivers_license dl ON p.license_id = dl.id
        JOIN income i ON p.ssn = i.ssn
        WHERE dl.car_make IS NOT NULL 
          AND i.annual_income IS NOT NULL
        GROUP BY dl.car_make, dl.gender
        HAVING COUNT(*) >= 1
        ORDER BY avg_owner_income DESC, total_owners DESC
        LIMIT 15;
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
                ROUND(AVG(age)::numeric, 1) as avg_age,
                ROUND(AVG(height)::numeric, 1) as avg_height,
                ROUND(AVG(annual_income)::numeric, 2) as avg_income,
                COUNT(DISTINCT car_make) as unique_car_brands,
                STRING_AGG(DISTINCT COALESCE(car_make, 'Нет авто'), ', ') as car_brands
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
        ORDER BY gs.avg_income DESC NULLS LAST;
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
        ORDER BY t.table_name;
        """
        
        self.execute_query(query, "Информация о таблицах базы данных")
    
    def show_sample_data(self):
        """Показать примеры данных из таблиц"""
        tables = ['crime_scene_report', 'drivers_license', 'person', 'income', 
                  'interview', 'get_fit_now_member', 'get_fit_now_check_in']
        
        print("\n" + "="*60)
        print("📊 ПРИМЕРЫ ДАННЫХ ИЗ ТАБЛИЦ")
        print("="*60)
        
        for table in tables:
            query = f"SELECT * FROM {table} LIMIT 3;"
            self.execute_query(query, f"Данные из таблицы {table} (первые 3 строки)")
    
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
            self.connection.commit()  # Фиксируем все изменения перед закрытием
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
        print("3. Наличие базы данных 'Murder_Mystery'")
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