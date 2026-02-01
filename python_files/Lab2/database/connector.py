import psycopg2
import pandas as pd
from PySide6.QtCore import QObject, Signal

class DatabaseConnector(QObject):
    """Класс для работы с подключением к PostgreSQL"""
    
    status_updated = Signal(str)
    progress_updated = Signal(int)
    
    def __init__(self):
        super().__init__()
        self.db_config = None
        self.connection = None
        self.table_names = []
    
    def test_connection(self, config):
        """Тест подключения к БД"""
        try:
            self.status_updated.emit("Тестируем подключение...")
            conn = psycopg2.connect(**config)
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            self.table_names = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            self.db_config = config
            self.status_updated.emit(f"✅ Подключение успешно! Найдено таблиц: {len(self.table_names)}")
            return True
            
        except Exception as e:
            self.status_updated.emit(f"❌ Ошибка подключения: {e}")
            return False
    
    def get_connection(self):
        """Получение активного подключения"""
        if self.connection is None or self.connection.closed:
            if self.db_config:
                try:
                    self.connection = psycopg2.connect(**self.db_config)
                except Exception as e:
                    raise ConnectionError(f"Не удалось подключиться к БД: {e}")
        return self.connection