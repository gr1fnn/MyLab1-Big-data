import sys
import os
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QComboBox, QListWidget,
                              QListWidgetItem, QTableWidget, QTableWidgetItem,
                              QLabel, QSplitter, QTabWidget, QGroupBox,
                              QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox,
                              QFileDialog, QTextEdit, QProgressBar, QLineEdit)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QColor, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np

# Импортируем ваш аналитический класс
from murder_mystery_analysis import MurderMysteryAnalyzer

# Конфигурация подключения (такая же как в вашем коде)
DB_CONFIG = {
    'host': 'povt-cluster.tstu.tver.ru',
    'port': 5432,
    'database': 'Murder_Mystery',
    'user': 'mpi',
    'password': '135a1'  # Пароль введет пользователь
}

class AnalysisThread(QThread):
    """Поток для выполнения анализа в фоне"""
    progress_signal = Signal(int, str)
    finished_signal = Signal(pd.DataFrame)
    error_signal = Signal(str)
    
    def __init__(self, analyzer, analysis_type, params=None):
        super().__init__()
        self.analyzer = analyzer
        self.analysis_type = analysis_type
        self.params = params or {}
        
    def run(self):
        try:
            if self.analysis_type == "connect":
                self.progress_signal.emit(10, "Подключение к базе данных...")
                success = self.analyzer.connect()
                if not success:
                    self.error_signal.emit("Ошибка подключения к базе данных")
                    return
                    
            elif self.analysis_type == "load":
                self.progress_signal.emit(20, "Загрузка данных...")
                success = self.analyzer.load_data()
                if not success:
                    self.error_signal.emit("Ошибка загрузки данных")
                    return
                    
            elif self.analysis_type == "combine":
                self.progress_signal.emit(40, "Объединение признаков...")
                df = self.analyzer.combine_features()
                if df is None:
                    self.error_signal.emit("Ошибка объединения данных")
                    return
                self.finished_signal.emit(df)
                
            elif self.analysis_type == "univariate":
                self.progress_signal.emit(60, "Выполнение одномерного анализа...")
                self.analyzer.univariate_analysis(gui_mode=True)
                
            elif self.analysis_type == "multivariate":
                self.progress_signal.emit(80, "Выполнение многомерного анализа...")
                self.analyzer.multivariate_analysis(gui_mode=True)
                
            self.progress_signal.emit(100, "Анализ завершен!")
            
        except Exception as e:
            self.error_signal.emit(str(e))

class MplCanvas(FigureCanvas):
    """Виджет для отображения matplotlib графиков"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Устанавливаем темную тему для графиков с белым текстом
        self.fig.patch.set_facecolor('#2b2b2b')  # Темный фон
        self.axes.set_facecolor('#2b2b2b')  # Темный фон для осей
        
        # Устанавливаем белый цвет для всех текстовых элементов
        self.axes.title.set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.spines['right'].set_color('white')

class MurderMysteryGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.current_data = None
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("Murder Mystery Analyzer - Детективный анализ")
        self.setGeometry(100, 100, 1400, 900)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Верхняя панель - подключение и управление
        top_panel = self.create_top_panel()
        main_layout.addWidget(top_panel)
        
        # Центральная область с разделителем
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - выбор данных и параметров
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Правая панель - отображение результатов
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter)
        
        # Нижняя панель - прогресс и статус
        bottom_panel = self.create_bottom_panel()
        main_layout.addWidget(bottom_panel)
        
        # Применяем стиль
        self.apply_styles()
        
    def create_top_panel(self):
        """Создание верхней панели"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # Заголовок
        title = QLabel("🔍 Murder Mystery Database Analyzer")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #333333;")  # Темный цвет текста
        
        # Кнопка подключения
        self.connect_btn = QPushButton("🔌 Подключиться к БД")
        self.connect_btn.setFixedWidth(200)
        
        # Поле для пароля
        password_label = QLabel("Пароль:")
        password_label.setStyleSheet("color: #333333;")  # Темный цвет текста
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFixedWidth(150)
        self.password_input.setText("135a1")  # Предзаполненный пароль
        
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.connect_btn)
        
        return panel
        
    def create_left_panel(self):
        """Создание левой панели с выбором данных"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Группа "Выбор таблиц"
        tables_group = QGroupBox("📋 Выбор таблиц для анализа")
        tables_layout = QVBoxLayout()
        
        self.tables_list = QListWidget()
        self.tables_list.setSelectionMode(QListWidget.MultiSelection)
        
        tables = [
            "person - Личные данные",
            "drivers_license - Водительские удостоверения", 
            "income - Доходы",
            "interview - Интервью",
            "crime_scene_report - Отчеты о преступлениях",
            "facebook_event_checkin - События Facebook",
            "get_fit_now_member - Члены фитнес-клуба",
            "get_fit_now_check_in - Посещения фитнес-клуба"
        ]
        
        for table in tables:
            item = QListWidgetItem(table)
            item.setCheckState(Qt.Unchecked)
            self.tables_list.addItem(item)
            
        # Выделить все по умолчанию
        for i in range(self.tables_list.count()):
            item = self.tables_list.item(i)
            item.setCheckState(Qt.Checked)
            
        tables_layout.addWidget(self.tables_list)
        
        # Кнопки выбора
        select_all_btn = QPushButton("Выбрать все")
        select_none_btn = QPushButton("Снять все")
        
        select_buttons_layout = QHBoxLayout()
        select_buttons_layout.addWidget(select_all_btn)
        select_buttons_layout.addWidget(select_none_btn)
        tables_layout.addLayout(select_buttons_layout)
        
        tables_group.setLayout(tables_layout)
        
        # Группа "Выбор признаков"
        features_group = QGroupBox("🎯 Выбор признаков для анализа")
        features_layout = QVBoxLayout()
        
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.MultiSelection)
        features_layout.addWidget(self.features_list)
        
        # Кнопка обновления признаков
        self.update_features_btn = QPushButton("🔄 Обновить список признаков")
        features_layout.addWidget(self.update_features_btn)
        
        features_group.setLayout(features_layout)
        
        # Группа "Тип анализа"
        analysis_group = QGroupBox("📊 Тип анализа")
        analysis_layout = QVBoxLayout()
        
        analysis_label = QLabel("Выберите тип анализа:")
        analysis_label.setStyleSheet("color: #333333;")  # Темный цвет текста
        
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Одномерный анализ (гистограммы)",
            "Многомерный анализ (графики 3-4 признаков)",
            "Объединение признаков",
            "Полный анализ"
        ])
        
        analysis_layout.addWidget(analysis_label)
        analysis_layout.addWidget(self.analysis_type_combo)
        
        # Кнопка запуска анализа
        self.analyze_btn = QPushButton("🚀 Запустить анализ")
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)
        
        # Кнопка сохранения результатов
        self.save_btn = QPushButton("💾 Сохранить результаты")
        self.save_btn.setEnabled(False)
        analysis_layout.addWidget(self.save_btn)
        
        analysis_group.setLayout(analysis_layout)
        
        # Добавляем все группы
        layout.addWidget(tables_group)
        layout.addWidget(features_group)
        layout.addWidget(analysis_group)
        layout.addStretch()
        
        # Подключение кнопок выбора
        select_all_btn.clicked.connect(self.select_all_tables)
        select_none_btn.clicked.connect(self.select_none_tables)
        self.update_features_btn.clicked.connect(self.update_features_list)
        
        return panel
        
    def create_right_panel(self):
        """Создание правой панели с отображением результатов"""
        panel = QTabWidget()
        
        # Вкладка "Данные"
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)
        
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        data_layout.addWidget(self.data_table)
        
        # Вкладка "Графики"
        self.graphs_tab = QWidget()
        graphs_layout = QVBoxLayout(self.graphs_tab)
        
        # Контейнер для графиков matplotlib
        self.graph_canvas = MplCanvas(self, width=10, height=8, dpi=100)
        graphs_layout.addWidget(self.graph_canvas)
        
        # Вкладка "Описание"
        self.description_tab = QWidget()
        description_layout = QVBoxLayout(self.description_tab)
        
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #333333;
                font-size: 11pt;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        description_layout.addWidget(self.description_text)
        
        # Вкладка "Статистика"
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.stats_tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #333333;
                font-size: 11pt;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        stats_layout.addWidget(self.stats_text)
        
        # Добавляем вкладки
        panel.addTab(self.data_tab, "📋 Данные")
        panel.addTab(self.graphs_tab, "📈 Графики")
        panel.addTab(self.description_tab, "📝 Описание")
        panel.addTab(self.stats_tab, "📊 Статистика")
        
        return panel
        
    def create_bottom_panel(self):
        """Создание нижней панели"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
                color: #333333;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        
        # Статус
        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("color: #333333; font-weight: bold;")
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        return panel
        
    def setup_connections(self):
        """Настройка соединений сигналов"""
        self.connect_btn.clicked.connect(self.connect_to_db)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.save_btn.clicked.connect(self.save_results)
        
    def apply_styles(self):
        """Применение стилей с темным текстом"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
                color: #333333;
            }
            QListWidget::item {
                color: #333333;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
            QTableWidget {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
                alternate-background-color: #f9f9f9;
                color: #333333;
                gridline-color: #dddddd;
            }
            QTableWidget::item {
                color: #333333;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #cccccc;
                color: #333333;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                color: #333333;
            }
            QTabBar::tab:selected {
                background-color: white;
                font-weight: bold;
                color: #333333;
            }
            QLabel {
                color: #333333;
            }
            QComboBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
                color: #333333;
                min-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #cccccc;
                background-color: white;
                color: #333333;
                selection-background-color: #4CAF50;
                selection-color: white;
            }
            QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
                color: #333333;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
                color: #333333;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        
    def select_all_tables(self):
        """Выбрать все таблицы"""
        for i in range(self.tables_list.count()):
            item = self.tables_list.item(i)
            item.setCheckState(Qt.Checked)
            
    def select_none_tables(self):
        """Снять выбор со всех таблиц"""
        for i in range(self.tables_list.count()):
            item = self.tables_list.item(i)
            item.setCheckState(Qt.Unchecked)
            
    def update_features_list(self):
        """Обновить список признаков"""
        if self.analyzer is None or self.analyzer.df_combined is None:
            QMessageBox.warning(self, "Внимание", 
                              "Сначала загрузите данные и объедините признаки")
            return
            
        self.features_list.clear()
        
        if hasattr(self.analyzer, 'df_combined'):
            columns = self.analyzer.df_combined.columns.tolist()
            for column in columns:
                item = QListWidgetItem(f"{column} ({self.analyzer.df_combined[column].dtype})")
                item.setData(Qt.UserRole, column)
                item.setCheckState(Qt.Unchecked)
                self.features_list.addItem(item)
                
    def connect_to_db(self):
        """Подключение к базе данных"""
        password = self.password_input.text().strip()
        if not password:
            QMessageBox.warning(self, "Внимание", "Введите пароль для подключения")
            return
            
        # Обновляем конфигурацию с введенным паролем
        DB_CONFIG['password'] = password
        
        # Создаем анализатор
        self.analyzer = MurderMysteryAnalyzer(DB_CONFIG)
        
        # Запускаем в отдельном потоке
        self.thread = AnalysisThread(self.analyzer, "connect")
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.on_connect_complete)
        self.thread.start()
        
    def on_connect_complete(self):
        """Действия после успешного подключения"""
        self.analyze_btn.setEnabled(True)
        self.update_features_btn.setEnabled(True)
        QMessageBox.information(self, "Успех", "Подключение к базе данных установлено!")
        
    def start_analysis(self):
        """Запуск анализа"""
        if self.analyzer is None:
            QMessageBox.warning(self, "Внимание", "Сначала подключитесь к базе данных")
            return
            
        analysis_type = self.analysis_type_combo.currentText()
        
        # Получаем выбранные таблицы
        selected_tables = []
        for i in range(self.tables_list.count()):
            item = self.tables_list.item(i)
            if item.checkState() == Qt.Checked:
                table_name = item.text().split(" - ")[0]
                selected_tables.append(table_name)
                
        if not selected_tables:
            QMessageBox.warning(self, "Внимание", "Выберите хотя бы одну таблицу")
            return
            
        # Получаем выбранные признаки для многомерного анализа
        selected_features = []
        if "многомерный" in analysis_type.lower():
            for i in range(self.features_list.count()):
                item = self.features_list.item(i)
                if item.checkState() == Qt.Checked:
                    feature_name = item.data(Qt.UserRole)
                    selected_features.append(feature_name)
                    
            if len(selected_features) < 3:
                QMessageBox.warning(self, "Внимание", 
                                  "Для многомерного анализа выберите минимум 3 признака")
                return
                
        # Определяем тип анализа для потока
        if "одномерный" in analysis_type.lower():
            thread_type = "univariate"
        elif "многомерный" in analysis_type.lower():
            thread_type = "multivariate"
            self.analyzer.selected_features = selected_features[:4]  # Берем первые 4
        elif "объединение" in analysis_type.lower():
            thread_type = "combine"
        else:  # полный анализ
            thread_type = "full"
            
        # Запускаем анализ в потоке
        self.thread = AnalysisThread(self.analyzer, thread_type)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.error_signal.connect(self.show_error)
        
        if thread_type == "combine":
            self.thread.finished_signal.connect(self.display_combined_data)
            
        self.thread.finished.connect(self.on_analysis_complete)
        self.thread.start()
        
    def display_combined_data(self, df):
        """Отображение объединенных данных в таблице"""
        self.current_data = df
        
        # Отображаем в таблице
        self.data_table.setRowCount(len(df))
        self.data_table.setColumnCount(len(df.columns))
        self.data_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        # Заполняем таблицу (ограничиваем количество строк для производительности)
        max_rows = min(100, len(df))
        for i in range(max_rows):
            for j, column in enumerate(df.columns):
                value = df.iloc[i, j]
                item = QTableWidgetItem(str(value) if not pd.isna(value) else "")
                item.setForeground(QColor("#333333"))  # Темный текст
                self.data_table.setItem(i, j, item)
                
        self.data_table.resizeColumnsToContents()
        
        # Показываем статистику
        stats_text = f"<h3>Объединенная таблица</h3>"
        stats_text += f"<p><b>Записей:</b> {len(df):,}</p>"
        stats_text += f"<p><b>Признаков:</b> {len(df.columns)}</p>"
        stats_text += f"<p><b>Пропущенных значений:</b> {df.isnull().sum().sum():,}</p>"
        
        stats_text += "<h4>Типы данных:</h4><ul>"
        for column in df.columns[:20]:  # Ограничиваем вывод
            dtype = str(df[column].dtype)
            stats_text += f"<li><b>{column}:</b> {dtype}</li>"
        
        if len(df.columns) > 20:
            stats_text += f"<li>... и еще {len(df.columns) - 20} признаков</li>"
        
        stats_text += "</ul>"
        
        self.stats_text.setHtml(stats_text)
        
        # Активируем кнопку сохранения
        self.save_btn.setEnabled(True)
        
    def display_graph(self, fig):
        """Отображение графика из matplotlib"""
        if fig is None:
            return
            
        try:
            # Очищаем текущий график
            self.graph_canvas.axes.clear()
            
            # Копируем фигуру
            self.graph_canvas.fig.clf()
            
            # Создаем новый subplot с темной темой
            ax = self.graph_canvas.fig.add_subplot(111)
            
            # Устанавливаем темную тему
            ax.set_facecolor('#2b2b2b')
            self.graph_canvas.fig.patch.set_facecolor('#2b2b2b')
            
            # Устанавливаем белый цвет для всех текстовых элементов
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
            # Переносим графики из полученной фигуры
            for src_ax in fig.axes:
                # Копируем линии
                for line in src_ax.lines:
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    color = line.get_color()
                    label = line.get_label()
                    linewidth = line.get_linewidth()
                    linestyle = line.get_linestyle()
                    
                    ax.plot(xdata, ydata, color=color, label=label if label != '_nolegend_' else None,
                           linewidth=linewidth, linestyle=linestyle)
                
                # Копируем scatter точки
                for collection in src_ax.collections:
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        colors = collection.get_facecolors()
                        if len(colors) > 0:
                            color = colors[0]
                            ax.scatter(offsets[:, 0], offsets[:, 1], color=color, alpha=0.6, s=50)
                
                # Копируем гистограммы (патчи)
                for patch in src_ax.patches:
                    # Создаем прямоугольник с такими же свойствами
                    rect = plt.Rectangle((patch.get_x(), patch.get_y()), 
                                       patch.get_width(), patch.get_height(),
                                       facecolor=patch.get_facecolor(),
                                       edgecolor=patch.get_edgecolor(),
                                       alpha=patch.get_alpha())
                    ax.add_patch(rect)
            
            # Копируем заголовки и метки
            if src_ax.get_title():
                ax.set_title(src_ax.get_title(), color='white')
            
            if src_ax.get_xlabel():
                ax.set_xlabel(src_ax.get_xlabel(), color='white')
            
            if src_ax.get_ylabel():
                ax.set_ylabel(src_ax.get_ylabel(), color='white')
            
            # Копируем легенду
            if src_ax.get_legend():
                ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            
            # Копируем пределы осей
            ax.set_xlim(src_ax.get_xlim())
            ax.set_ylim(src_ax.get_ylim())
            
            # Обновляем канвас
            self.graph_canvas.draw()
            
            # Закрываем исходную фигуру
            plt.close(fig)
            
        except Exception as e:
            print(f"Ошибка отображения графика: {e}")
            # Показываем сообщение об ошибке
            self.graph_canvas.axes.clear()
            self.graph_canvas.axes.text(0.5, 0.5, 'Ошибка отображения графика', 
                                      ha='center', va='center', color='white', fontsize=14,
                                      transform=self.graph_canvas.axes.transAxes)
            self.graph_canvas.draw()
        
    def update_progress(self, value, message):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def show_error(self, error_message):
        """Показ ошибки"""
        QMessageBox.critical(self, "Ошибка", error_message)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ошибка")
        
    def on_analysis_complete(self):
        """Действия после завершения анализа"""
        self.status_label.setText("Анализ завершен")
        
        # Обновляем описание
        if hasattr(self.analyzer, 'df_combined') and self.analyzer.df_combined is not None:
            description = self.generate_description()
            self.description_text.setHtml(description)
            
        # Показываем сообщение об успехе
        QMessageBox.information(self, "Успех", "Анализ успешно завершен!")
        
    def generate_description(self):
        """Генерация описания данных"""
        if not hasattr(self.analyzer, 'df_combined') or self.analyzer.df_combined is None:
            return "<p style='color: #333333;'>Данные не загружены</p>"
            
        df = self.analyzer.df_combined
        
        description = "<h2 style='color: #333333;'>📊 ОПИСАНИЕ БАЗЫ ДАННЫХ MURDER MYSTERY</h2>"
        description += "<hr>"
        
        description += "<h3 style='color: #333333;'>🔍 Общая информация:</h3>"
        description += f"<ul style='color: #333333;'>"
        description += f"<li><b>Всего записей:</b> {len(df):,}</li>"
        description += f"<li><b>Количество признаков:</b> {len(df.columns)}</li>"
        description += "</ul>"
        
        description += "<h3 style='color: #333333;'>🎯 Ключевые признаки:</h3>"
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        description += f"<p style='color: #333333;'><b>Числовые признаки ({len(numeric_cols)}):</b><br>"
        description += f"{', '.join(numeric_cols[:10])}"
        if len(numeric_cols) > 10:
            description += f" ... и еще {len(numeric_cols) - 10}"
        description += "</p>"
        
        description += f"<p style='color: #333333;'><b>Категориальные признаки ({len(categorical_cols)}):</b><br>"
        description += f"{', '.join(categorical_cols[:10])}"
        if len(categorical_cols) > 10:
            description += f" ... и еще {len(categorical_cols) - 10}"
        description += "</p>"
        
        description += "<h3 style='color: #333333;'>📈 Статистика по числовым признакам:</h3>"
        description += "<table border='1' style='border-collapse: collapse; color: #333333;'>"
        description += "<tr><th>Признак</th><th>Среднее</th><th>Медиана</th><th>Min</th><th>Max</th></tr>"
        
        for col in numeric_cols[:8]:  # Показываем первые 8
            if col in df.columns and df[col].notna().any():
                description += f"<tr>"
                description += f"<td><b>{col}</b></td>"
                description += f"<td>{df[col].mean():.2f}</td>"
                description += f"<td>{df[col].median():.2f}</td>"
                description += f"<td>{df[col].min():.2f}</td>"
                description += f"<td>{df[col].max():.2f}</td>"
                description += f"</tr>"
                
        description += "</table>"
        
        description += "<h3 style='color: #333333;'>🔍 Выводы по анализу:</h3>"
        description += "<ul style='color: #333333;'>"
        description += "<li>База данных содержит детективную информацию о преступлениях</li>"
        description += "<li>Имеются демографические данные, информация о доходах и транспортных средствах</li>"
        description += "<li>Данные могут использоваться для выявления паттернов и взаимосвязей</li>"
        description += "<li>Присутствуют как числовые, так и категориальные признаки для комплексного анализа</li>"
        description += "</ul>"
        
        return description
        
    def save_results(self):
        """Сохранение результатов анализа"""
        if self.current_data is None:
            QMessageBox.warning(self, "Внимание", "Нет данных для сохранения")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "", 
            "CSV файлы (*.csv);;Excel файлы (*.xlsx);;Все файлы (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.current_data.to_csv(file_path, index=False, encoding='utf-8')
                elif file_path.endswith('.xlsx'):
                    self.current_data.to_excel(file_path, index=False)
                else:
                    file_path += '.csv'
                    self.current_data.to_csv(file_path, index=False, encoding='utf-8')
                    
                QMessageBox.information(self, "Успех", f"Данные сохранены в {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {str(e)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MurderMysteryGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()