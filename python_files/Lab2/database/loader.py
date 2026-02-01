import pandas as pd
import numpy as np
from .connector import DatabaseConnector

class DataLoader(DatabaseConnector):
    """Класс для загрузки и обработки данных"""
    
    def __init__(self):
        super().__init__()
        self.dataframes = {}
        self.combined_df = None
    
    def load_all_data(self):
        """Загрузка ВСЕХ таблиц из базы данных"""
        if not self.db_config:
            self.status_updated.emit("❌ Сначала настройте подключение")
            return False
        
        try:
            self.status_updated.emit(f"Загружаем ВСЕ таблицы ({len(self.table_names)} таблиц)...")
            self.connection = self.get_connection()
            
            self.dataframes = {}
            
            total_tables = len(self.table_names)
            for i, table_name in enumerate(self.table_names):
                self.progress_updated.emit(int((i / total_tables) * 100))
                
                try:
                    self.dataframes[table_name] = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
                    self.status_updated.emit(f"✅ Таблица '{table_name}' загружена: {len(self.dataframes[table_name])} строк")
                except Exception as e:
                    self.status_updated.emit(f"⚠️ Ошибка загрузки таблицы '{table_name}': {e}")
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"✅ Все данные загружены! Загружено таблиц: {len(self.dataframes)}")
            
            self.show_data_statistics()
            return True
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка загрузки: {e}")
            return False
    
    def show_data_statistics(self):
        """Показ статистики по загруженным данным"""
        if not self.dataframes:
            return
        
        stats = "📊 СТАТИСТИКА ЗАГРУЖЕННЫХ ДАННЫХ:\n\n"
        
        for table_name, df in self.dataframes.items():
            stats += f"📋 {table_name}:\n"
            stats += f"   • Количество строк: {len(df):,}\n"
            stats += f"   • Количество столбцов: {len(df.columns)}\n"
            
            columns_preview = ", ".join(df.columns[:5])
            if len(df.columns) > 5:
                columns_preview += f" ... и еще {len(df.columns)-5} столбцов"
            stats += f"   • Столбцы: {columns_preview}\n\n"
        
        self.status_updated.emit(stats)
    
    def combine_data(self, feature_type="Основные демографические"):
        """Объединение данных в одну таблицу с разными вариантами"""
        try:
            self.status_updated.emit(f"Объединяем данные (тип: {feature_type})...")
            
            required_tables = ['person']
            missing_tables = [t for t in required_tables if t not in self.dataframes]
            
            if missing_tables:
                self.status_updated.emit(f"❌ Отсутствуют таблицы: {', '.join(missing_tables)}")
                return None
            
            df = self.dataframes['person'].copy()
            
            if feature_type == "Основные демографические":
                df = self._add_basic_features(df)
            elif feature_type == "Полный набор признаков":
                df = self._add_full_features(df)
            elif feature_type == "Демография + Доходы + Авто":
                df = self._add_demographic_income_auto(df)
            elif feature_type == "Все доступные данные":
                df = self._add_all_features(df)
            
            df = self._clean_and_add_features(df)
            self.combined_df = df
            
            self.status_updated.emit(f"✅ Объединенная таблица создана: {len(df):,} строк, {len(df.columns)} столбцов")
            return df
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка объединения: {e}")
            import traceback
            self.status_updated.emit(traceback.format_exc())
            return None
    
    def _add_basic_features(self, df):
        """Добавление базовых признаков"""
        if 'drivers_license' in self.dataframes:
            df = pd.merge(
                df,
                self.dataframes['drivers_license'],
                left_on='license_id',
                right_on='id',
                how='left',
                suffixes=('', '_license')
            )
        
        if 'income' in self.dataframes:
            df = pd.merge(
                df,
                self.dataframes['income'],
                on='ssn',
                how='left'
            )
        return df
    
    def _add_full_features(self, df):
        """Добавление всех возможных признаков"""
        df = self._add_basic_features(df)
        
        if 'interview' in self.dataframes:
            df = pd.merge(
                df,
                self.dataframes['interview'][['person_id', 'transcript']],
                left_on='id',
                right_on='person_id',
                how='left'
            )
        
        if 'get_fit_now_member' in self.dataframes:
            df = pd.merge(
                df,
                self.dataframes['get_fit_now_member'][['person_id', 'membership_status', 'membership_start_date']],
                left_on='id',
                right_on='person_id',
                how='left'
            )
        return df
    
    def _add_demographic_income_auto(self, df):
        """Добавление демографии, доходов и авто"""
        return self._add_basic_features(df)
    
    def _add_all_features(self, df):
        """Добавление всех таблиц"""
        merge_attempts = [
            ('drivers_license', 'license_id', 'id'),
            ('income', 'ssn', 'ssn'),
            ('interview', 'id', 'person_id'),
            ('get_fit_now_member', 'id', 'person_id'),
            ('facebook_event_checkin', 'id', 'person_id'),
        ]
        
        for table_name, left_key, right_key in merge_attempts:
            if table_name in self.dataframes:
                try:
                    if left_key == 'id' and right_key == 'person_id':
                        df = pd.merge(
                            df,
                            self.dataframes[table_name],
                            left_on=left_key,
                            right_on=right_key,
                            how='left',
                            suffixes=('', f'_{table_name}')
                        )
                    else:
                        df = pd.merge(
                            df,
                            self.dataframes[table_name],
                            left_on=left_key,
                            right_on=right_key,
                            how='left',
                            suffixes=('', f'_{table_name}')
                        )
                except Exception as e:
                    self.status_updated.emit(f"⚠️ Не удалось объединить {table_name}: {e}")
        return df
    
    def _clean_and_add_features(self, df):
        """Очистка и добавление расчетных признаков"""
        columns_to_drop = []
        for col in df.columns:
            if col.endswith('_y') or col in ['person_id', 'id_license']:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        if 'license_id' in df.columns:
            df['has_license'] = df['license_id'].notna().astype(int)
        
        if 'annual_income' in df.columns:
            df['has_income'] = df['annual_income'].notna().astype(int)
            df['income_group'] = pd.qcut(
                df['annual_income'].fillna(0),
                q=4,
                labels=['Низкий', 'Ниже среднего', 'Выше среднего', 'Высокий']
            )
        
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'].fillna(df['age'].median()),
                bins=[0, 20, 30, 40, 50, 60, 100],
                labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
            )
        
        return df