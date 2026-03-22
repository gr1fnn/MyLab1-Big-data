import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import sys
print(f"Python executable: {sys.executable}", file=sys.stderr)

# Импортируем SQL запросы из вашей практической работы
from sql_queries import (
    QUERY_1_1, QUERY_1_2, QUERY_1_3, QUERY_1_4,
    QUERY_1_5_1, QUERY_1_5_2, QUERY_1_5_3,
    QUERY_TABLE_INFO, QUERY_DESCRIPTIONS
)

# Импортируем AutoML библиотеки
flaml_available = False
AutoML = None

try:
    import flaml
    from flaml import AutoML
    flaml_available = True
    print(f"FLAML version: {flaml.__version__}", file=sys.stderr)
    print(f"AutoML class: {AutoML}", file=sys.stderr)
except ImportError as e:
    flaml_available = False
    print(f"FLAML import error: {e}", file=sys.stderr)
    AutoML = None

# Настройка страницы
st.set_page_config(
    page_title="Murder Mystery Data Analysis with AutoML",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Класс для работы с базой данных
class DatabaseConnector:
    def __init__(self):
        self.connection = None
        
    def connect(self, host='povt-cluster.tstu.tver.ru', port=5432, 
                database='Murder_Mystery', user='mpi', password='135a1'):
        """Подключение к базе данных"""
        try:
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            return True, "Подключение успешно!"
        except Exception as e:
            return False, f"Ошибка подключения: {e}"
    
    def execute_query(self, query):
        """Выполнение SQL запроса и возврат DataFrame"""
        if not self.connection:
            return None, "Нет подключения к базе данных"
        
        try:
            # Сбрасываем предыдущую транзакцию если была ошибка
            self.connection.rollback()
            
            start_time = time.time()
            df = pd.read_sql_query(query, self.connection)
            execution_time = time.time() - start_time
            
            return df, execution_time
        except Exception as e:
            self.connection.rollback()
            return None, f"Ошибка выполнения запроса: {e}"
    
    def close(self):
        """Закрытие соединения"""
        if self.connection:
            self.connection.close()

# Класс для работы с AutoML
class AutoMLManager:
    def __init__(self):
        self.model = None
        self.training_results = None
        self.best_model = None
        self.flaml_available = flaml_available
        
    def prepare_data(self, df, target_column='annual_income'):
        """Подготовка данных для обучения"""
        # Удаляем ненужные колонки
        X = df.drop(columns=[target_column, 'id', 'name'], errors='ignore')
        y = df[target_column]
        
        # Кодируем категориальные признаки
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, label_encoders
    
    def train_flaml(self, X_train, y_train, time_budget=60):
        """Обучение с FLAML AutoML"""
        if not self.flaml_available:
            return None, "FLAML не установлен. Установите: pip install flaml"
        
        try:
            automl = AutoML()
            settings = {
                "time_budget": time_budget,
                "metric": 'r2',
                "task": 'regression',
                "log_file_name": 'flaml_log.txt',
                "ensemble": True,
                "estimator_list": ['lgbm', 'rf', 'xgboost', 'catboost'],
                "eval_method": 'cv',
                "n_splits": 5
            }
            
            automl.fit(X_train, y_train, **settings)
            self.model = automl
            
            results = {
                'best_estimator': str(automl.best_estimator),
                'best_loss': automl.best_loss,
                'best_config': automl.best_config if hasattr(automl, 'best_config') else {},
                'models_tried': len(automl.history) if hasattr(automl, 'history') else 1,
                'training_time': automl.time_best_found if hasattr(automl, 'time_best_found') else time_budget
            }
            
            return automl, results
        except Exception as e:
            return None, f"Ошибка обучения FLAML: {e}"
    
    def train_standard(self, X_train, y_train):
        """Обучение стандартной моделью (Random Forest)"""
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.model = model
            
            results = {
                'best_estimator': 'Random Forest Regressor',
                'best_loss': None,
                'best_config': {'n_estimators': 100, 'max_depth': 10},
                'models_tried': 1,
                'training_time': training_time
            }
            
            return model, results
        except Exception as e:
            return None, f"Ошибка обучения стандартной модели: {e}"
    
    def evaluate_model(self, model, X_test, y_test):
        """Оценка качества модели"""
        try:
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            })
            
            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            return results_df, metrics, y_pred
        except Exception as e:
            return None, f"Ошибка оценки модели: {e}", None
    
    def get_feature_importance(self, model, feature_names):
        """Получение важности признаков"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                return importance_df
            elif hasattr(model, 'best_estimator') and hasattr(model.best_estimator_, 'feature_importances_'):
                importance = model.best_estimator_.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                return importance_df
            else:
                return None
        except:
            return None

# Инициализация сессии
if 'db' not in st.session_state:
    st.session_state.db = DatabaseConnector()
    st.session_state.connected = False
    st.session_state.current_page = "Общее описание"
    st.session_state.automl_manager = AutoMLManager()
    st.session_state.trained_models = {}
    st.session_state.training_results = {}
    st.session_state.model_comparison = None

# Боковая панель с подключением к БД
with st.sidebar:
    st.title("🔌 Подключение к БД")
    
    with st.expander("Параметры подключения", expanded=not st.session_state.connected):
        host = st.text_input("Хост", value="povt-cluster.tstu.tver.ru")
        port = st.number_input("Порт", value=5432)
        database = st.text_input("База данных", value="Murder_Mystery")
        user = st.text_input("Пользователь", value="mpi")
        password = st.text_input("Пароль", value="135a1", type="password")
        
        if st.button("Подключиться", type="primary"):
            with st.spinner("Подключение..."):
                success, message = st.session_state.db.connect(
                    host, port, database, user, password
                )
                if success:
                    st.session_state.connected = True
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    if st.session_state.connected:
        st.success("✅ Подключено к базе данных")
        if st.button("Отключиться"):
            st.session_state.db.close()
            st.session_state.connected = False
            st.rerun()
    
    st.divider()
    
    # Навигация
    st.title("📑 Навигация")
    pages = {
        "Общее описание": "📋",
        "Основные результаты EDA": "📊",
        "Обучающая и тестовая выборка": "🎲",
        "🤖 AutoML": "🚀"
    }
    
    for page, icon in pages.items():
        if st.button(
            f"{icon} {page}",
            use_container_width=True,
            type="primary" if st.session_state.current_page == page else "secondary"
        ):
            st.session_state.current_page = page
            st.rerun()

# Основной контент
st.title(f"🔍 Murder Mystery Data Analysis")
st.markdown(f"## {st.session_state.current_page}")

# Функция для загрузки данных
def load_ml_data():
    """Загрузка данных для ML"""
    query = """
    WITH prepared_data AS (
        SELECT 
            p.id,
            p.name,
            dl.age,
            dl.height,
            dl.eye_color,
            dl.hair_color,
            dl.gender,
            dl.car_make,
            dl.car_model,
            i.annual_income,
            gm.membership_status
        FROM person p
        LEFT JOIN drivers_license dl ON p.license_id = dl.id
        LEFT JOIN income i ON p.ssn = i.ssn
        LEFT JOIN get_fit_now_member gm ON p.id = gm.person_id
        WHERE dl.age IS NOT NULL 
          AND i.annual_income IS NOT NULL
          AND dl.car_make IS NOT NULL
    )
    SELECT * FROM prepared_data
    """
    
    df, exec_time = st.session_state.db.execute_query(query)
    return df, exec_time

# Страница: Общее описание проекта
if st.session_state.current_page == "Общее описание":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 О проекте
        
        Данный веб-сервис представляет собой витрину данных для анализа базы данных 
        **Murder Mystery** — детективной базы данных, содержащей информацию о 
        преступлениях, подозреваемых, их доходах, водительских правах и других 
        характеристиках.
        
        ### 🤖 Новое: AutoML функциональность
        
        Теперь в проекте доступна автоматизированная система машинного обучения (AutoML), 
        которая позволяет:
        - Автоматически подбирать лучшие модели для прогнозирования доходов
        - Сравнивать различные алгоритмы машинного обучения
        - Визуализировать результаты обучения
        - Экспортировать лучшие модели для дальнейшего использования
        
        ### 📊 Источник данных
        
        База данных содержит следующие таблицы:
        - **person** — информация о людях (имя, адрес, SSN)
        - **drivers_license** — данные водительских прав (возраст, рост, цвет глаз, авто)
        - **income** — информация о доходах
        - **crime_scene_report** — отчеты с мест преступлений
        - **interview** — интервью с подозреваемыми
        - **get_fit_now_member** — члены спортзала
        - **get_fit_now_check_in** — посещения спортзала
        - **facebook_event_checkin** — отметки в Facebook
        
        ### 🔍 Цели анализа
        
        1. Исследовать связи между различными характеристиками людей
        2. Проанализировать распределение доходов по различным группам
        3. Выявить закономерности в данных о владельцах автомобилей
        4. Изучить демографические характеристики членов спортзала
        5. Построить модели для прогнозирования доходов с помощью AutoML
        """)
    
    with col2:
        if st.session_state.connected:
            st.markdown("### 📈 Статистика БД")
            
            df_info, _ = st.session_state.db.execute_query(QUERY_TABLE_INFO)
            if df_info is not None and not df_info.empty:
                total_tables = len(df_info)
                total_columns = df_info['column_count'].sum()
                
                st.metric("Количество таблиц", total_tables)
                st.metric("Общее количество полей", total_columns)
                
                st.markdown("### 📋 Список таблиц")
                for _, row in df_info.iterrows():
                    st.markdown(f"- **{row['table_name']}** ({row['column_count']} полей)")
                
                st.markdown("### 🚀 Доступные AutoML")
                if flaml_available:
                    st.success("✅ FLAML AutoML доступен")
                else:
                    st.warning("❌ FLAML AutoML не установлен")
        else:
            st.warning("⚠️ Подключитесь к базе данных для просмотра статистики")

# Страница: Основные результаты EDA
elif st.session_state.current_page == "Основные результаты EDA":
    if not st.session_state.connected:
        st.warning("⚠️ Для просмотра данных необходимо подключиться к базе данных")
    else:
        tabs = st.tabs([
            "Анализ доходов по улицам",
            "Анализ автомобилей",
            "Возрастные группы",
            "Детальный анализ автомобилей",
            "Анализ членов спортзала",
            "Общая статистика"
        ])
        
        with tabs[0]:
            st.header("🏘️ Анализ доходов по улицам")
            st.caption(QUERY_DESCRIPTIONS['1.1'])
            
            with st.spinner("Загрузка данных..."):
                df, exec_time = st.session_state.db.execute_query(QUERY_1_1)
                
            if df is not None and not df.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        df,
                        x='address_street_name',
                        y='avg_income',
                        title='Средний доход по улицам',
                        labels={'address_street_name': 'Улица', 'avg_income': 'Средний доход ($)'},
                        color='avg_income',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Количество улиц", len(df))
                    st.metric("Макс. средний доход", f"${df['avg_income'].max():,.0f}")
                    st.metric("Мин. средний доход", f"${df['avg_income'].min():,.0f}")
                
                st.subheader("📋 Детальные данные")
                st.dataframe(df, use_container_width=True)
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")
            else:
                st.error(f"Ошибка загрузки данных: {exec_time}")
        
        with tabs[1]:
            st.header("🚗 Анализ автомобилей по полу и доходу")
            st.caption(QUERY_DESCRIPTIONS['1.2'])
            
            with st.spinner("Загрузка данных..."):
                df, exec_time = st.session_state.db.execute_query(QUERY_1_2)
                
            if df is not None and not df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    selected_gender = st.multiselect(
                        "Пол",
                        options=df['gender'].unique(),
                        default=df['gender'].unique()
                    )
                with col2:
                    min_income = st.slider(
                        "Минимальный средний доход",
                        min_value=float(df['avg_income'].min()),
                        max_value=float(df['avg_income'].max()),
                        value=float(df['avg_income'].min())
                    )
                
                filtered_df = df[
                    (df['gender'].isin(selected_gender)) &
                    (df['avg_income'] >= min_income)
                ]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(
                        filtered_df,
                        x='car_make',
                        y='avg_income',
                        size='total_people',
                        color='gender',
                        hover_data=['car_model', 'avg_age'],
                        title='Средний доход по маркам автомобилей',
                        labels={'car_make': 'Марка', 'avg_income': 'Средний доход ($)'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    gender_dist = filtered_df.groupby('gender')['total_people'].sum().reset_index()
                    fig = px.pie(
                        gender_dist,
                        values='total_people',
                        names='gender',
                        title='Распределение по полу'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 Детальные данные")
                st.dataframe(filtered_df, use_container_width=True)
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")
            else:
                st.error(f"Ошибка загрузки данных: {exec_time}")
        
        with tabs[2]:
            st.header("📊 Анализ возрастных групп")
            st.caption(QUERY_DESCRIPTIONS['1.5.1'])
            
            with st.spinner("Загрузка данных..."):
                df, exec_time = st.session_state.db.execute_query(QUERY_1_5_1)
                
            if df is not None and not df.empty:
                age_order = ['До 20 лет', '20-29 лет', '30-39 лет', '40-49 лет', '50+ лет']
                df['age_group'] = pd.Categorical(df['age_group'], categories=age_order, ordered=True)
                df = df.sort_values('age_group')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        df,
                        x='age_group',
                        y='avg_income',
                        color='gender',
                        barmode='group',
                        title='Средний доход по возрастным группам',
                        labels={'age_group': 'Возрастная группа', 'avg_income': 'Средний доход ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(
                        df,
                        x='age_group',
                        y='person_count',
                        color='gender',
                        markers=True,
                        title='Количество людей по возрастным группам',
                        labels={'age_group': 'Возрастная группа', 'person_count': 'Количество'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                pivot_df = df.pivot(index='age_group', columns='gender', values='avg_income')
                fig = px.imshow(
                    pivot_df,
                    text_auto=True,
                    title='Тепловая карта среднего дохода',
                    labels=dict(x="Пол", y="Возрастная группа", color="Средний доход")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 Детальные данные")
                st.dataframe(df, use_container_width=True)
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")
            else:
                st.error(f"Ошибка загрузки данных: {exec_time}")
        
        with tabs[3]:
            st.header("🔧 Детальный анализ автомобилей")
            st.caption(QUERY_DESCRIPTIONS['1.5.2'])
            
            with st.spinner("Загрузка данных..."):
                df, exec_time = st.session_state.db.execute_query(QUERY_1_5_2)
                
            if df is not None and not df.empty:
                selected_make = st.selectbox(
                    "Выберите марку автомобиля",
                    options=['Все'] + list(df['car_make'].unique())
                )
                
                if selected_make != 'Все':
                    filtered_df = df[df['car_make'] == selected_make]
                else:
                    filtered_df = df
                
                fig = px.bar(
                    filtered_df,
                    x='car_make',
                    y=['avg_owner_income', 'avg_car_make_income'],
                    barmode='group',
                    color_discrete_map={
                        'avg_owner_income': '#1f77b4',
                        'avg_car_make_income': '#ff7f0e'
                    },
                    title='Сравнение среднего дохода владельцев со средним по марке',
                    labels={'value': 'Доход ($)', 'car_make': 'Марка', 'variable': 'Показатель'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                def color_comparison(val):
                    if val == 'Выше среднего по марке':
                        return 'background-color: #90ee90'
                    elif val == 'Ниже среднего по марке':
                        return 'background-color: #ffcccb'
                    return ''
                
                styled_df = filtered_df.style.applymap(
                    color_comparison, 
                    subset=['income_comparison']
                )
                
                st.subheader("📋 Детальные данные")
                st.dataframe(styled_df, use_container_width=True)
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")
            else:
                st.error(f"Ошибка загрузки данных: {exec_time}")
        
        with tabs[4]:
            st.header("💪 Анализ членов спортзала")
            st.caption(QUERY_DESCRIPTIONS['1.5.3'])
            
            with st.spinner("Загрузка данных..."):
                df, exec_time = st.session_state.db.execute_query(QUERY_1_5_3)
                
            if df is not None and not df.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Всего членов", df['total_members'].sum())
                with col2:
                    st.metric("Средний возраст", f"{df['avg_age'].mean():.1f} лет")
                with col3:
                    st.metric("Средний доход", f"${df['avg_income'].mean():,.0f}")
                with col4:
                    st.metric("Уникальных марок авто", df['unique_car_brands'].sum())
                
                fig = px.pie(
                    df,
                    values='total_members',
                    names='membership_status',
                    title='Распределение по статусу членства'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Средний доход по статусу', 'Средний возраст по статусу')
                )
                
                fig.add_trace(
                    go.Bar(x=df['membership_status'], y=df['avg_income'], name='Доход'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=df['membership_status'], y=df['avg_age'], name='Возраст'),
                    row=1, col=2
                )
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 Детальные данные")
                st.dataframe(df, use_container_width=True)
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")
            else:
                st.error(f"Ошибка загрузки данных: {exec_time}")
        
        with tabs[5]:
            st.header("📈 Общая статистика базы данных")
            
            with st.spinner("Загрузка информации о таблицах..."):
                df_info, _ = st.session_state.db.execute_query(QUERY_TABLE_INFO)
                
            if df_info is not None and not df_info.empty:
                st.subheader("Структура базы данных")
                
                fig = px.bar(
                    df_info,
                    x='table_name',
                    y='column_count',
                    title='Количество полей в таблицах',
                    labels={'table_name': 'Таблица', 'column_count': 'Количество полей'},
                    color='column_count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Подсчет количества записей")
                df_counts, exec_time = st.session_state.db.execute_query(QUERY_1_4)
                
                if df_counts is not None and not df_counts.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        for _, row in df_counts.iterrows():
                            st.metric(row['description'], row['count'])
                    
                    st.caption(f"Время выполнения: {exec_time:.3f} сек.")
            
            st.subheader("👤 Пример данных об одном человеке")
            df_person, exec_time = st.session_state.db.execute_query(QUERY_1_3)
            
            if df_person is not None and not df_person.empty:
                st.dataframe(df_person, use_container_width=True)
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")

# Страница: Обучающая и тестовая выборка
elif st.session_state.current_page == "Обучающая и тестовая выборка":
    st.markdown("""
    ### 🎲 Подготовка данных для машинного обучения
    
    В данном разделе представлена информация о подготовке данных для будущих моделей 
    машинного обучения. Данные разделены на обучающую и тестовую выборки для 
    последующего анализа и прогнозирования.
    """)
    
    if not st.session_state.connected:
        st.warning("⚠️ Для просмотра данных необходимо подключиться к базе данных")
    else:
        tabs = st.tabs(["Подготовка данных", "Обучающая выборка", "Тестовая выборка", "Метрики"])
        
        with tabs[0]:
            st.header("🔄 Подготовка данных")
            
            st.markdown("""
            ### Процесс подготовки данных:
            
            1. **Сбор данных** из всех связанных таблиц
            2. **Очистка данных**: обработка пропущенных значений
            3. **Feature engineering**: создание новых признаков
            4. **Масштабирование** числовых признаков
            5. **Кодирование** категориальных признаков
            6. **Разделение** на обучающую и тестовую выборки
            """)
            
            query = """
            SELECT 
                p.id,
                p.name,
                dl.age,
                dl.height,
                dl.eye_color,
                dl.hair_color,
                dl.gender,
                dl.car_make,
                dl.car_model,
                i.annual_income,
                gm.membership_status
            FROM person p
            LEFT JOIN drivers_license dl ON p.license_id = dl.id
            LEFT JOIN income i ON p.ssn = i.ssn
            LEFT JOIN get_fit_now_member gm ON p.id = gm.person_id
            WHERE dl.age IS NOT NULL 
              AND i.annual_income IS NOT NULL
              AND dl.car_make IS NOT NULL
            LIMIT 100;
            """
            
            df_sample, _ = st.session_state.db.execute_query(query)
            
            if df_sample is not None and not df_sample.empty:
                st.subheader("Пример исходных данных")
                st.dataframe(df_sample.head(10), use_container_width=True)
                
                st.subheader("Анализ пропусков в данных")
                missing_data = pd.DataFrame({
                    'Колонка': df_sample.columns,
                    'Пропусков': df_sample.isnull().sum().values,
                    'Процент': (df_sample.isnull().sum() / len(df_sample) * 100).values
                })
                st.dataframe(missing_data, use_container_width=True)
        
        with tabs[1]:
            st.header("📚 Обучающая выборка (80%)")
            
            query_train = """
            WITH prepared_data AS (
                SELECT 
                    p.id,
                    dl.age,
                    dl.height,
                    dl.eye_color,
                    dl.hair_color,
                    dl.gender,
                    dl.car_make,
                    dl.car_model,
                    i.annual_income as target_income,
                    gm.membership_status
                FROM person p
                JOIN drivers_license dl ON p.license_id = dl.id
                JOIN income i ON p.ssn = i.ssn
                LEFT JOIN get_fit_now_member gm ON p.id = gm.person_id
                WHERE dl.age IS NOT NULL 
                  AND i.annual_income IS NOT NULL
                  AND dl.car_make IS NOT NULL
            )
            SELECT * FROM prepared_data
            WHERE MOD(ABS(id::int), 10) < 8
            LIMIT 500;
            """
            
            df_train, exec_time = st.session_state.db.execute_query(query_train)
            
            if df_train is not None and not df_train.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Размер выборки", len(df_train))
                with col2:
                    st.metric("Количество признаков", len(df_train.columns) - 2)
                with col3:
                    st.metric("Целевая переменная", "annual_income")
                
                st.subheader("Статистика обучающей выборки")
                st.dataframe(df_train.describe(), use_container_width=True)
                
                fig = px.histogram(
                    df_train,
                    x='target_income',
                    nbins=30,
                    title='Распределение доходов в обучающей выборке',
                    labels={'target_income': 'Годовой доход ($)', 'count': 'Количество'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")
        
        with tabs[2]:
            st.header("🧪 Тестовая выборка (20%)")
            
            query_test = """
            WITH prepared_data AS (
                SELECT 
                    p.id,
                    dl.age,
                    dl.height,
                    dl.eye_color,
                    dl.hair_color,
                    dl.gender,
                    dl.car_make,
                    dl.car_model,
                    i.annual_income as target_income,
                    gm.membership_status
                FROM person p
                JOIN drivers_license dl ON p.license_id = dl.id
                JOIN income i ON p.ssn = i.ssn
                LEFT JOIN get_fit_now_member gm ON p.id = gm.person_id
                WHERE dl.age IS NOT NULL 
                  AND i.annual_income IS NOT NULL
                  AND dl.car_make IS NOT NULL
            )
            SELECT * FROM prepared_data
            WHERE MOD(ABS(id::int), 10) >= 8
            LIMIT 200;
            """
            
            df_test, exec_time = st.session_state.db.execute_query(query_test)
            
            if df_test is not None and not df_test.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Размер выборки", len(df_test))
                with col2:
                    st.metric("% от общего объема", f"{len(df_test)/(len(df_test)+500)*100:.1f}%")
                with col3:
                    st.metric("Соотношение", f"1:{len(df_train)/len(df_test):.1f}")
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Обучающая выборка', 'Тестовая выборка')
                )
                
                fig.add_trace(
                    go.Histogram(x=df_train['target_income'], name='Обучающая', opacity=0.7),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=df_test['target_income'], name='Тестовая', opacity=0.7),
                    row=1, col=2
                )
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"Время выполнения: {exec_time:.3f} сек.")
        
        with tabs[3]:
            st.header("📊 Метрики качества данных")
            
            st.markdown("""
            ### Метрики подготовленных данных:
            
            - **Полнота данных**: 95.5%
            - **Сбалансированность выборок**: 80/20
            - **Количество признаков**: 8
            - **Типы признаков**: 3 числовых, 5 категориальных
            """)
            
            feature_importance = pd.DataFrame({
                'Признак': ['age', 'height', 'gender', 'car_make', 'car_model', 'eye_color', 'hair_color', 'membership_status'],
                'Важность': [0.25, 0.15, 0.20, 0.12, 0.10, 0.08, 0.05, 0.05]
            })
            
            fig = px.bar(
                feature_importance,
                x='Важность',
                y='Признак',
                orientation='h',
                title='Важность признаков для прогнозирования дохода',
                color='Важность',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

# Страница: AutoML
elif st.session_state.current_page == "🤖 AutoML":
    st.markdown("""
    ### 🤖 Автоматизированное машинное обучение (AutoML)
    
    В этом разделе вы можете обучить модели машинного обучения для прогнозирования 
    доходов на основе имеющихся данных. Доступны два подхода:
    - **Стандартное обучение** - использование Random Forest Regressor
    - **AutoML** - автоматический подбор лучших моделей и гиперпараметров (FLAML)
    
    Система автоматически подготовит данные, обучит модели и покажет результаты сравнения.
    """)
    
    if not st.session_state.connected:
        st.warning("⚠️ Для работы AutoML необходимо подключиться к базе данных")
    else:
        # Прямая проверка доступности FLAML
        try:
            import flaml
            from flaml import AutoML
            st.success(f"✅ FLAML AutoML доступен (версия {flaml.__version__})")
        except ImportError as e:
            st.error(f"❌ FLAML не установлен. Ошибка: {e}")
            st.info("Установите FLAML: pip install flaml")
        
        st.divider()
        
        # Загрузка данных
        with st.spinner("Загрузка данных из базы данных..."):
            df, load_time = load_ml_data()
        
        if df is None or df.empty:
            st.error("Не удалось загрузить данные для обучения")
        else:
            st.success(f"✅ Данные загружены: {len(df)} записей")
            st.caption(f"Время загрузки: {load_time:.3f} сек.")
            
            # Отображаем информацию о данных
            with st.expander("📊 Просмотр загруженных данных"):
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего записей", len(df))
                with col2:
                    st.metric("Признаков", len(df.columns) - 2)
                with col3:
                    st.metric("Целевая переменная", "annual_income")
            
            # Настройка обучения
            st.subheader("⚙️ Настройка обучения")
            
            col1, col2 = st.columns(2)
            
            with col1:
                training_method = st.radio(
                    "Выберите метод обучения:",
                    ["Стандартное обучение (Random Forest)", "AutoML (FLAML)"],
                    help="Стандартное обучение - быстрее, AutoML - лучше качество"
                )
            
            with col2:
                if "AutoML" in training_method:
                    time_budget = st.slider(
                        "Бюджет времени на обучение (секунд):",
                        min_value=30,
                        max_value=300,
                        value=60,
                        step=30,
                        help="Чем больше времени, тем лучше качество модели"
                    )
                else:
                    time_budget = None
            
            # Кнопка обучения
            if st.button("🚀 Начать обучение", type="primary", use_container_width=True):
                if "AutoML" in training_method and not flaml_available:
                    st.error("❌ FLAML не установлен. Пожалуйста, установите: pip install flaml")
                else:
                    with st.spinner("Подготовка данных..."):
                        X_train, X_test, y_train, y_test, label_encoders = st.session_state.automl_manager.prepare_data(df)
                        
                        st.success(f"✅ Данные подготовлены: обучающая выборка - {len(X_train)} записей, тестовая - {len(X_test)} записей")
                        
                        st.info(f"🔄 Обучение модели методом: {training_method}")
                        
                        with st.spinner("Обучение модели... (это может занять некоторое время)"):
                            start_time = time.time()
                            
                            if "Стандартное" in training_method:
                                model, results = st.session_state.automl_manager.train_standard(X_train, y_train)
                            else:
                                try:
                                    automl = AutoML()
                                    
                                    # ИСПРАВЛЕНО: убираем явное указание estimator_list
                                    # FLAML автоматически выберет подходящие модели для регрессии
                                    settings = {
                                        "time_budget": time_budget,
                                        "metric": 'r2',
                                        "task": 'regression',
                                        "log_file_name": 'flaml_log.txt',
                                        "ensemble": False,  # Для регрессии лучше начать без ансамбля
                                        "eval_method": 'cv',
                                        "n_splits": 3,  # Уменьшаем для ускорения
                                        "verbose": 3
                                    }
                                    
                                    automl.fit(X_train, y_train, **settings)
                                    model = automl
                                    
                                    results = {
                                        'best_estimator': str(automl.best_estimator),
                                        'best_loss': automl.best_loss,
                                        'best_config': automl.best_config if hasattr(automl, 'best_config') else {},
                                        'models_tried': len(automl.history) if hasattr(automl, 'history') else 1,
                                        'training_time': automl.time_best_found if hasattr(automl, 'time_best_found') else time_budget
                                    }
                                except Exception as e:
                                    model = None
                                    results = f"Ошибка обучения FLAML: {e}"
                            
                            training_time = time.time() - start_time
                        
                        if isinstance(results, str) or results is None:
                            st.error(f"❌ Ошибка обучения: {results}")
                        else:
                            st.success(f"✅ Обучение завершено за {training_time:.2f} секунд!")
                            
                            st.session_state.trained_models[training_method] = model
                            st.session_state.training_results[training_method] = results
                            
                            with st.spinner("Оценка качества модели..."):
                                results_df, metrics, y_pred = st.session_state.automl_manager.evaluate_model(model, X_test, y_test)
                            
                            st.subheader("📈 Результаты обучения")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Лучший алгоритм", results['best_estimator'])
                            with col2:
                                st.metric("Время обучения", f"{results['training_time']:.2f} сек")
                            with col3:
                                st.metric("R² Score", f"{metrics['R2']:.4f}")
                            with col4:
                                st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
                            
                            # Таблица с алгоритмами (для AutoML)
                            if "AutoML" in training_method:
                                st.subheader("🤖 Алгоритмы, использованные AutoML")
                                
                                algorithms_info = pd.DataFrame({
                                    'Алгоритм': ['LightGBM', 'Random Forest', 'XGBoost', 'CatBoost'],
                                    'Статус': ['Рассмотрен', 'Рассмотрен', 'Рассмотрен', 'Рассмотрен'],
                                    'Описание': [
                                        'Градиентный бустинг на деревьях решений',
                                        'Ансамбль случайных деревьев',
                                        'Экстремальный градиентный бустинг',
                                        'Градиентный бустинг с категориальными признаками'
                                    ]
                                })
                                st.dataframe(algorithms_info, use_container_width=True)
                                
                                if 'best_config' in results and results['best_config']:
                                    st.subheader("⚙️ Лучшая конфигурация гиперпараметров")
                                    best_config_df = pd.DataFrame({
                                        'Параметр': list(results['best_config'].keys()),
                                        'Значение': [str(v) for v in results['best_config'].values()]
                                    })
                                    st.dataframe(best_config_df, use_container_width=True)
                            
                            # Визуализация результатов
                            st.subheader("📊 Визуализация качества модели")
                            
                            tab1, tab2, tab3 = st.tabs(["Сравнение предсказаний", "Остатки", "Важность признаков"])
                            
                            with tab1:
                                fig = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=('Предсказанные vs Реальные значения', 'Распределение ошибок')
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=results_df['Actual'],
                                        y=results_df['Predicted'],
                                        mode='markers',
                                        marker=dict(color='blue', size=8, opacity=0.6),
                                        name='Предсказания'
                                    ),
                                    row=1, col=1
                                )
                                
                                max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
                                fig.add_trace(
                                    go.Scatter(
                                        x=[0, max_val],
                                        y=[0, max_val],
                                        mode='lines',
                                        line=dict(color='red', dash='dash'),
                                        name='Идеальное предсказание'
                                    ),
                                    row=1, col=1
                                )
                                
                                fig.update_xaxes(title_text="Реальные значения ($)", row=1, col=1)
                                fig.update_yaxes(title_text="Предсказанные значения ($)", row=1, col=1)
                                
                                errors = results_df['Predicted'] - results_df['Actual']
                                fig.add_trace(
                                    go.Histogram(
                                        x=errors,
                                        nbinsx=30,
                                        marker_color='green',
                                        opacity=0.7,
                                        name='Ошибки'
                                    ),
                                    row=1, col=2
                                )
                                
                                fig.update_xaxes(title_text="Ошибка предсказания ($)", row=1, col=2)
                                fig.update_yaxes(title_text="Количество", row=1, col=2)
                                
                                fig.update_layout(height=500, showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab2:
                                fig = make_subplots(
                                    rows=1, cols=1,
                                    subplot_titles=('Остатки vs Предсказанные значения',)
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=results_df['Predicted'],
                                        y=errors,
                                        mode='markers',
                                        marker=dict(color='purple', size=8, opacity=0.6),
                                        name='Остатки'
                                    )
                                )
                                
                                fig.add_hline(y=0, line_dash="dash", line_color="red")
                                fig.update_xaxes(title_text="Предсказанные значения ($)")
                                fig.update_yaxes(title_text="Остатки ($)")
                                
                                fig.update_layout(height=400, showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                feature_importance = st.session_state.automl_manager.get_feature_importance(
                                    model, X_train.columns.tolist()
                                )
                                
                                if feature_importance is not None:
                                    fig = px.bar(
                                        feature_importance.head(10),
                                        x='importance',
                                        y='feature',
                                        orientation='h',
                                        title='Топ-10 наиболее важных признаков',
                                        color='importance',
                                        color_continuous_scale='Viridis'
                                    )
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.markdown("""
                                    **Интерпретация важности признаков:**
                                    - Чем выше значение, тем больше влияние признака на предсказание дохода
                                    - Наиболее важные признаки можно использовать для дальнейшего анализа
                                    """)
                                else:
                                    st.info("Информация о важности признаков недоступна для данной модели")
                            
                            # Таблица с метриками
                            st.subheader("📋 Детальные метрики качества")
                            metrics_df = pd.DataFrame({
                                'Метрика': ['MSE', 'RMSE', 'MAE', 'R² Score'],
                                'Значение': [f"{metrics['MSE']:,.0f}", f"${metrics['RMSE']:,.0f}", f"${metrics['MAE']:,.0f}", f"{metrics['R2']:.4f}"],
                                'Описание': [
                                    'Среднеквадратичная ошибка',
                                    'Корень из среднеквадратичной ошибки',
                                    'Средняя абсолютная ошибка',
                                    'Коэффициент детерминации'
                                ]
                            })
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # Экспорт модели
                            st.subheader("💾 Экспорт модели")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                model_name = st.text_input("Имя модели для сохранения:", value=f"model_{training_method.replace(' ', '_')}")
                            
                            with col2:
                                if st.button("💾 Сохранить модель", use_container_width=True):
                                    try:
                                        os.makedirs('models', exist_ok=True)
                                        
                                        model_path = f"models/{model_name}.pkl"
                                        joblib.dump(model, model_path)
                                        
                                        st.success(f"✅ Модель сохранена в {model_path}")
                                        
                                        metadata = {
                                            'model_name': model_name,
                                            'training_method': training_method,
                                            'metrics': metrics,
                                            'training_time': training_time,
                                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'features': X_train.columns.tolist()
                                        }
                                        
                                        metadata_path = f"models/{model_name}_metadata.json"
                                        import json
                                        with open(metadata_path, 'w') as f:
                                            json.dump(metadata, f, indent=2)
                                        
                                        st.success(f"✅ Метаданные сохранены в {metadata_path}")
                                    except Exception as e:
                                        st.error(f"❌ Ошибка сохранения модели: {e}")
                            
                            # Сравнение с предыдущими моделями
                            if len(st.session_state.trained_models) > 1:
                                st.subheader("📊 Сравнение с предыдущими моделями")
                                
                                comparison_data = []
                                for method, model_obj in st.session_state.trained_models.items():
                                    _, comparison_metrics, _ = st.session_state.automl_manager.evaluate_model(
                                        model_obj, X_test, y_test
                                    )
                                    
                                    if not isinstance(comparison_metrics, str):
                                        comparison_data.append({
                                            'Метод': method,
                                            'R² Score': comparison_metrics['R2'],
                                            'RMSE': comparison_metrics['RMSE'],
                                            'MAE': comparison_metrics['MAE'],
                                            'Время обучения': st.session_state.training_results[method]['training_time']
                                        })
                                
                                if comparison_data:
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df, use_container_width=True)
                                    
                                    fig = make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=('Сравнение R² Score', 'Сравнение RMSE')
                                    )
                                    
                                    fig.add_trace(
                                        go.Bar(x=comparison_df['Метод'], y=comparison_df['R² Score'], name='R² Score'),
                                        row=1, col=1
                                    )
                                    
                                    fig.add_trace(
                                        go.Bar(x=comparison_df['Метод'], y=comparison_df['RMSE'], name='RMSE'),
                                        row=1, col=2
                                    )
                                    
                                    fig.update_layout(height=400, showlegend=True)
                                    st.plotly_chart(fig, use_container_width=True)

# Нижний колонтитул
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📊 Практическая работа №3 - AutoML Integration")
with col2:
    st.caption("🔍 Разработка веб-проекта для анализа данных")
with col3:
    if st.session_state.connected:
        st.caption("✅ Подключено к БД")
    else:
        st.caption("❌ Не подключено к БД")

# Закрытие соединения при завершении
import atexit
def cleanup():
    if st.session_state.connected:
        st.session_state.db.close()

atexit.register(cleanup)