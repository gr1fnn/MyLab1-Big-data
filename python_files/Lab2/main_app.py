import sys
import warnings
warnings.filterwarnings('ignore')

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtCore import QTimer
import matplotlib.pyplot as plt

# Импорт UI
from murder_mystery_ui import Ui_MainWindow

# Импорт модулей
from database.loader import DataLoader
from analysis.univariate import UnivariateAnalysis
from analysis.multivariate import MultivariateAnalysis
from analysis.mpg_analysis import MPGAnalysis
from analysis.advanced_analysis import AdvancedAnalysis

class MurderMysteryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Инициализация UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Инициализация анализатора
        self.analyzer = DataLoader()
        self.mpg_analyzer = MPGAnalysis()
        
        # Подключение сигналов
        self.analyzer.status_updated.connect(self.update_status)
        self.analyzer.progress_updated.connect(self.update_progress)
        
        # Подключение кнопок
        self.ui.testButton.clicked.connect(self.test_connection)
        self.ui.loadAllButton.clicked.connect(self.load_all_data)
        self.ui.combineButton.clicked.connect(self.combine_data)
        self.ui.plotHistogramButton.clicked.connect(self.plot_histograms)
        self.ui.plotMultivariateButton.clicked.connect(self.plot_multivariate)
        
        # Кнопки общей части
        self.ui.analyzeMpgButton.clicked.connect(self.analyze_mpg)
        self.ui.hypothesisTestButton.clicked.connect(self.test_hypotheses_mpg)
        self.ui.gradientDescentButton.clicked.connect(self.gradient_descent_mpg)
        
        # Кнопки самостоятельной части
        self.ui.exploratoryAnalysisButton.clicked.connect(self.exploratory_analysis)
        self.ui.correlationAnalysisButton.clicked.connect(self.correlation_analysis)
        
        # Показываем окно
        self.show()
    
    def update_status(self, message):
        """Обновление статуса в UI"""
        current_tab = self.ui.tabWidget.currentIndex()
        
        if current_tab == 0:  # Вкладка подключения
            self.ui.connectionOutput.append(message)
        elif current_tab == 1:  # Вкладка данных
            self.ui.dataOutput.append(message)
        elif current_tab == 2:  # Вкладка анализа
            self.ui.analysisOutput.append(message)
        elif current_tab == 3:  # Вкладка расширенного анализа
            self.ui.advancedOutput.append(message)
        
        # Прокрутка вниз
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def update_progress(self, value):
        """Обновление прогресс-бара"""
        self.ui.progressBar.setValue(value)
    
    def scroll_to_bottom(self):
        """Прокрутка текстовых полей вниз"""
        current_tab = self.ui.tabWidget.currentIndex()
        
        if current_tab == 0:
            self.ui.connectionOutput.verticalScrollBar().setValue(
                self.ui.connectionOutput.verticalScrollBar().maximum()
            )
        elif current_tab == 1:
            self.ui.dataOutput.verticalScrollBar().setValue(
                self.ui.dataOutput.verticalScrollBar().maximum()
            )
        elif current_tab == 2:
            self.ui.analysisOutput.verticalScrollBar().setValue(
                self.ui.analysisOutput.verticalScrollBar().maximum()
            )
        elif current_tab == 3:
            self.ui.advancedOutput.verticalScrollBar().setValue(
                self.ui.advancedOutput.verticalScrollBar().maximum()
            )
    
    def test_connection(self):
        """Тест подключения к БД"""
        config = {
            'host': self.ui.hostInput.text(),
            'port': self.ui.portInput.text(),
            'database': self.ui.databaseInput.text(),
            'user': self.ui.userInput.text(),
            'password': self.ui.passwordInput.text()
        }
        
        self.analyzer.test_connection(config)
    
    def load_all_data(self):
        """Загрузка всех данных"""
        self.ui.loadAllButton.setEnabled(False)
        self.ui.combineButton.setEnabled(False)
        
        success = self.analyzer.load_all_data()
        
        if success:
            self.ui.combineButton.setEnabled(True)
        
        self.ui.loadAllButton.setEnabled(True)
    
    def combine_data(self):
        """Объединение данных"""
        feature_type = self.ui.featureComboBox.currentText()
        
        combined_df = self.analyzer.combine_data(feature_type)
        
        if combined_df is not None:
            self.ui.combineButton.setText("✅ Данные объединены")
            self.ui.plotHistogramButton.setEnabled(True)
            self.ui.plotMultivariateButton.setEnabled(True)
            self.ui.exploratoryAnalysisButton.setEnabled(True)
            self.ui.correlationAnalysisButton.setEnabled(True)
            
            # Показываем сводку по объединенным данным
            summary = f"📋 СВОДКА ПО ОБЪЕДИНЕННЫМ ДАННЫМ:\n"
            summary += f"• Количество строк: {len(combined_df):,}\n"
            summary += f"• Количество столбцов: {len(combined_df.columns)}\n"
            summary += f"• Тип объединения: {feature_type}\n"
            
            self.ui.dataOutput.append(summary)
    
    def plot_histograms(self):
        """Построение гистограмм"""
        feature1 = self.ui.histogram1Combo.currentText()
        feature2 = self.ui.histogram2Combo.currentText()
        bins = self.ui.binsSpinBox.value()
        
        univariate = UnivariateAnalysis(self.analyzer.combined_df)
        report, fig = univariate.plot_histograms(feature1, feature2, bins)
        
        if report:
            self.ui.analysisOutput.append(report)
        
        if fig:
            plt.show()
    
    def plot_multivariate(self):
        """Построение многомерных графиков"""
        graph_type = self.ui.graph1Combo.currentText()
        
        multivariate = MultivariateAnalysis(
            self.analyzer.combined_df,
            self.analyzer.dataframes
        )
        report, fig = multivariate.plot_multivariate(graph_type)
        
        if report:
            self.ui.analysisOutput.append(report)
        
        if fig:
            plt.show()
    
    def analyze_mpg(self):
        """Анализ данных mpg"""
        report = self.mpg_analyzer.load_and_analyze_mpg()
        self.ui.advancedOutput.append(report)
        
        if self.mpg_analyzer.mpg_df is not None:
            self.ui.hypothesisTestButton.setEnabled(True)
            self.ui.gradientDescentButton.setEnabled(True)
            self.ui.analyzeMpgButton.setText("✅ Данные mpg загружены")
    
    def test_hypotheses_mpg(self):
        """Проверка гипотез для mpg"""
        report = self.mpg_analyzer.test_hypotheses_mpg()
        self.ui.advancedOutput.append(report)
    
    def gradient_descent_mpg(self):
        """Градиентный спуск для mpg"""
        report = self.mpg_analyzer.gradient_descent_mpg()
        self.ui.advancedOutput.append(report)
    
    def exploratory_analysis(self):
        """Разведочный анализ данных Murder Mystery"""
        if self.analyzer.combined_df is None:
            self.ui.advancedOutput.append("❌ Сначала объедините данные")
            return
        
        advanced = AdvancedAnalysis(self.analyzer.combined_df)
        report = advanced.exploratory_analysis()
        self.ui.advancedOutput.append(report)
     
    def correlation_analysis(self):
        """Корреляционный анализ данных Murder Mystery"""
        if self.analyzer.combined_df is None:
            self.ui.advancedOutput.append("❌ Сначала объедините данные")
            return
        
        advanced = AdvancedAnalysis(self.analyzer.combined_df)
        report = advanced.correlation_analysis()
        self.ui.advancedOutput.append(report)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MurderMysteryApp()
    window.setWindowTitle("Murder Mystery Analyzer")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()