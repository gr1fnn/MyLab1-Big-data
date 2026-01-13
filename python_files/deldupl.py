import psycopg2

def fix_all_data_problems():
    PASSWORD = "135a1"
    
    try:
        conn = psycopg2.connect(
            host="povt-cluster.tstu.tver.ru",
            port=5432,
            database="Murder_Mystery",
            user="mpi",
            password=PASSWORD
        )
        cursor = conn.cursor()
        
        print("="*60)
        print("ИСПРАВЛЕНИЕ ВСЕХ ПРОБЛЕМ С ДАННЫМИ")
        print("="*60)
        
        # 1. Проверить и исправить отсутствующие license_id
        print("\n1. Проверка отсутствующих license_id...")
        cursor.execute("""
            SELECT COUNT(DISTINCT p.license_id) as missing_licenses,
                   COUNT(*) as affected_records
            FROM person p
            LEFT JOIN drivers_license dl ON p.license_id = dl.id
            WHERE dl.id IS NULL AND p.license_id IS NOT NULL
        """)
        missing_licenses, affected_records = cursor.fetchone()
        
        if missing_licenses > 0:
            print(f"  Найдено {missing_licenses} отсутствующих license_id")
            print(f"  Затронуто {affected_records} записей в person")
            
            print("  Устанавливаю NULL для проблемных записей...")
            cursor.execute("""
                UPDATE person 
                SET license_id = NULL
                WHERE license_id IN (
                    SELECT DISTINCT p.license_id
                    FROM person p
                    LEFT JOIN drivers_license dl ON p.license_id = dl.id
                    WHERE dl.id IS NULL AND p.license_id IS NOT NULL
                )
            """)
            print(f"  Обновлено {cursor.rowcount} записей")
        
        # 2. Проверить отсутствующие person_id в дочерних таблицах
        print("\n2. Проверка ссылочной целостности...")
        
        # interview → person
        cursor.execute("""
            SELECT COUNT(*) as missing
            FROM interview i
            LEFT JOIN person p ON i.person_id = p.id
            WHERE p.id IS NULL
        """)
        missing_interview = cursor.fetchone()[0]
        
        if missing_interview > 0:
            print(f"  В interview найдено {missing_interview} записей с отсутствующими person_id")
            cursor.execute("DELETE FROM interview WHERE person_id NOT IN (SELECT id FROM person)")
            print(f"  Удалено {cursor.rowcount} записей из interview")
        
        # facebook_event_checkin → person
        cursor.execute("""
            SELECT COUNT(*) as missing
            FROM facebook_event_checkin f
            LEFT JOIN person p ON f.person_id = p.id
            WHERE p.id IS NULL
        """)
        missing_fb = cursor.fetchone()[0]
        
        if missing_fb > 0:
            print(f"  В facebook_event_checkin найдено {missing_fb} записей с отсутствующими person_id")
            cursor.execute("DELETE FROM facebook_event_checkin WHERE person_id NOT IN (SELECT id FROM person)")
            print(f"  Удалено {cursor.rowcount} записей из facebook_event_checkin")
        
        # get_fit_now_member → person
        cursor.execute("""
            SELECT COUNT(*) as missing
            FROM get_fit_now_member g
            LEFT JOIN person p ON g.person_id = p.id
            WHERE p.id IS NULL
        """)
        missing_member = cursor.fetchone()[0]
        
        if missing_member > 0:
            print(f"  В get_fit_now_member найдено {missing_member} записей с отсутствующими person_id")
            cursor.execute("DELETE FROM get_fit_now_member WHERE person_id NOT IN (SELECT id FROM person)")
            print(f"  Удалено {cursor.rowcount} записей из get_fit_now_member")
        
        # income → person (по ssn)
        cursor.execute("""
            SELECT COUNT(*) as missing
            FROM income i
            LEFT JOIN person p ON i.ssn = p.ssn
            WHERE p.ssn IS NULL
        """)
        missing_income = cursor.fetchone()[0]
        
        if missing_income > 0:
            print(f"  В income найдено {missing_income} записей с отсутствующими ssn в person")
            cursor.execute("DELETE FROM income WHERE ssn NOT IN (SELECT ssn FROM person)")
            print(f"  Удалено {cursor.rowcount} записей из income")
        
        # get_fit_now_check_in → get_fit_now_member
        cursor.execute("""
            SELECT COUNT(*) as missing
            FROM get_fit_now_check_in c
            LEFT JOIN get_fit_now_member m ON c.membership_id = m.id
            WHERE m.id IS NULL
        """)
        missing_checkin = cursor.fetchone()[0]
        
        if missing_checkin > 0:
            print(f"  В get_fit_now_check_in найдено {missing_checkin} записей с отсутствующими membership_id")
            cursor.execute("DELETE FROM get_fit_now_check_in WHERE membership_id NOT IN (SELECT id FROM get_fit_now_member)")
            print(f"  Удалено {cursor.rowcount} записей из get_fit_now_check_in")
        
        # 3. Создать constraints
        print("\n3. Создание constraints...")
        
        constraints = [
            ("person", "person_ssn_key", "UNIQUE (ssn)"),
            ("income", "income_pkey", "PRIMARY KEY (ssn)"),
            ("interview", "interview_pkey", "PRIMARY KEY (person_id)"),
            ("facebook_event_checkin", "facebook_event_checkin_pkey", "PRIMARY KEY (person_id, event_id, date)"),
            ("get_fit_now_member", "get_fit_now_member_pkey", "PRIMARY KEY (id)"),
            ("get_fit_now_check_in", "get_fit_now_check_in_pkey", "PRIMARY KEY (membership_id, check_in_date, check_in_time)"),
        ]
        
        for table, name, sql in constraints:
            try:
                cursor.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
                cursor.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} {sql}")
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        # FOREIGN KEY constraints с NOT VALID
        print("\n4. Создание FOREIGN KEY constraints...")
        
        fk_constraints = [
            ("person", "person_license_id_fkey", "FOREIGN KEY (license_id) REFERENCES drivers_license(id) NOT VALID"),
            ("income", "income_ssn_fkey", "FOREIGN KEY (ssn) REFERENCES person(ssn) NOT VALID"),
            ("interview", "interview_person_id_fkey", "FOREIGN KEY (person_id) REFERENCES person(id) NOT VALID"),
            ("facebook_event_checkin", "facebook_event_checkin_person_id_fkey", "FOREIGN KEY (person_id) REFERENCES person(id) NOT VALID"),
            ("get_fit_now_member", "get_fit_now_member_person_id_fkey", "FOREIGN KEY (person_id) REFERENCES person(id) NOT VALID"),
            ("get_fit_now_check_in", "get_fit_now_check_in_membership_id_fkey", "FOREIGN KEY (membership_id) REFERENCES get_fit_now_member(id) NOT VALID"),
        ]
        
        for table, name, sql in fk_constraints:
            try:
                cursor.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
                cursor.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} {sql}")
                print(f"  ✅ {name} (с NOT VALID)")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n" + "="*60)
        print("✅ ВСЕ ПРОБЛЕМЫ ИСПРАВЛЕНЫ!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    fix_all_data_problems()