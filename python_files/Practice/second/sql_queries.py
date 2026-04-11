"""
SQL запросы для базы данных Murder Mystery
Общий файл с запросами для использования в разных реализациях (psycopg2 и Spark)
"""

# 1. Select c Join по двум таблицам с сортировкой и агрегацией
QUERY_1_1 = """
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

# 2. Select c Join по трем таблицам с сортировкой и агрегацией
QUERY_1_2 = """
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

# 3. Данные по одному объекту по всем возможным таблицам
QUERY_1_3 = """
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

# 4. Подсчет количества строк по совмещенным данным в 2 таблицах
QUERY_1_4 = """
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

# 5.1 Анализ возрастного распределения с доходами
QUERY_1_5_1 = """
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

# 5.2 Детальный анализ автомобилей по полу и доходу
QUERY_1_5_2 = """
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

# 5.3 Комплексный анализ членов спортзала
QUERY_1_5_3 = """
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
        ROUND(AVG(dl.age)::numeric, 1) as avg_age,
        ROUND(AVG(dl.height)::numeric, 1) as avg_height,
        ROUND(AVG(i.annual_income)::numeric, 2) as avg_income,
        COUNT(DISTINCT dl.car_make) as unique_car_brands,
        STRING_AGG(DISTINCT COALESCE(dl.car_make, 'Нет авто'), ', ') as car_brands
    FROM person p
    JOIN get_fit_now_member gm ON p.id = gm.person_id
    LEFT JOIN income i ON p.ssn = i.ssn
    LEFT JOIN drivers_license dl ON p.license_id = dl.id
    WHERE gm.membership_status IS NOT NULL
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

# Запрос для получения информации о таблицах
QUERY_TABLE_INFO = """
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

# Описания запросов
QUERY_DESCRIPTIONS = {
    '1.1': "1. JOIN двух таблиц с сортировкой и агрегацией",
    '1.2': "2. JOIN трех таблиц с сортировкой и агрегацией",
    '1.3': "3. Все данные об одном человеке (минимальный ID)",
    '1.4': "4. Подсчет количества строк по совмещенным данным",
    '1.5.1': "5.1 Анализ возрастного распределения с доходами",
    '1.5.2': "5.2 Детальный анализ автомобилей по полу и доходу",
    '1.5.3': "5.3 Комплексный анализ членов спортзала",
    'table_info': "Информация о таблицах базы данных"
}

def get_sample_query(table_name, limit=3):
    """Возвращает запрос для получения sample данных из таблицы"""
    return f"SELECT * FROM {table_name} LIMIT {limit};"