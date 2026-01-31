# -*- coding: utf-8 -*-

from PySide6.QtCore import (QCoreApplication, QMetaObject, Qt)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QSpinBox, QStatusBar, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget, QProgressBar)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1100, 750)
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        
        # Заголовок
        self.titleLabel = QLabel(self.centralwidget)
        self.titleLabel.setObjectName(u"titleLabel")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.verticalLayout.addWidget(self.titleLabel)
        
        # Вкладки
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        
        # Вкладка 1: Подключение
        self.tabConnection = QWidget()
        self.tabConnection.setObjectName(u"tabConnection")
        self.verticalLayout_2 = QVBoxLayout(self.tabConnection)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        
        # Группа подключения
        self.groupBoxConnection = QGroupBox(self.tabConnection)
        self.groupBoxConnection.setObjectName(u"groupBoxConnection")
        
        self.gridLayout = QVBoxLayout(self.groupBoxConnection)
        
        # Поля ввода
        self.hostLayout = QHBoxLayout()
        self.hostLabel = QLabel(self.groupBoxConnection)
        self.hostLabel.setObjectName(u"hostLabel")
        self.hostLabel.setMinimumWidth(100)
        self.hostLayout.addWidget(self.hostLabel)
        
        self.hostInput = QLineEdit(self.groupBoxConnection)
        self.hostInput.setObjectName(u"hostInput")
        self.hostInput.setText("povt-cluster.tstu.tver.ru")
        self.hostLayout.addWidget(self.hostInput)
        self.gridLayout.addLayout(self.hostLayout)
        
        self.portLayout = QHBoxLayout()
        self.portLabel = QLabel(self.groupBoxConnection)
        self.portLabel.setObjectName(u"portLabel")
        self.portLabel.setMinimumWidth(100)
        self.portLayout.addWidget(self.portLabel)
        
        self.portInput = QLineEdit(self.groupBoxConnection)
        self.portInput.setObjectName(u"portInput")
        self.portInput.setText("5432")
        self.portLayout.addWidget(self.portInput)
        self.gridLayout.addLayout(self.portLayout)
        
        self.databaseLayout = QHBoxLayout()
        self.databaseLabel = QLabel(self.groupBoxConnection)
        self.databaseLabel.setObjectName(u"databaseLabel")
        self.databaseLabel.setMinimumWidth(100)
        self.databaseLayout.addWidget(self.databaseLabel)
        
        self.databaseInput = QLineEdit(self.groupBoxConnection)
        self.databaseInput.setObjectName(u"databaseInput")
        self.databaseInput.setText("Murder_Mystery")
        self.databaseLayout.addWidget(self.databaseInput)
        self.gridLayout.addLayout(self.databaseLayout)
        
        self.userLayout = QHBoxLayout()
        self.userLabel = QLabel(self.groupBoxConnection)
        self.userLabel.setObjectName(u"userLabel")
        self.userLabel.setMinimumWidth(100)
        self.userLayout.addWidget(self.userLabel)
        
        self.userInput = QLineEdit(self.groupBoxConnection)
        self.userInput.setObjectName(u"userInput")
        self.userInput.setText("mpi")
        self.userLayout.addWidget(self.userInput)
        self.gridLayout.addLayout(self.userLayout)
        
        self.passwordLayout = QHBoxLayout()
        self.passwordLabel = QLabel(self.groupBoxConnection)
        self.passwordLabel.setObjectName(u"passwordLabel")
        self.passwordLabel.setMinimumWidth(100)
        self.passwordLayout.addWidget(self.passwordLabel)
        
        self.passwordInput = QLineEdit(self.groupBoxConnection)
        self.passwordInput.setObjectName(u"passwordInput")
        self.passwordInput.setText("135a1")
        self.passwordInput.setEchoMode(QLineEdit.Password)
        self.passwordLayout.addWidget(self.passwordInput)
        self.gridLayout.addLayout(self.passwordLayout)
        
        # Кнопки
        self.testButton = QPushButton(self.groupBoxConnection)
        self.testButton.setObjectName(u"testButton")
        self.gridLayout.addWidget(self.testButton)
        
        self.loadAllButton = QPushButton(self.groupBoxConnection)
        self.loadAllButton.setObjectName(u"loadAllButton")
        self.gridLayout.addWidget(self.loadAllButton)
        
        self.verticalLayout_2.addWidget(self.groupBoxConnection)
        
        # Прогресс бар
        self.progressBar = QProgressBar(self.tabConnection)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)
        self.verticalLayout_2.addWidget(self.progressBar)
        
        # Вывод информации
        self.connectionOutput = QTextEdit(self.tabConnection)
        self.connectionOutput.setObjectName(u"connectionOutput")
        self.connectionOutput.setReadOnly(True)
        self.verticalLayout_2.addWidget(self.connectionOutput)
        
        self.tabWidget.addTab(self.tabConnection, "")
        
        # Вкладка 2: Выбор данных
        self.tabData = QWidget()
        self.tabData.setObjectName(u"tabData")
        self.verticalLayout_3 = QVBoxLayout(self.tabData)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        
        # Группа выбора признаков
        self.groupBoxFeatures = QGroupBox(self.tabData)
        self.groupBoxFeatures.setObjectName(u"groupBoxFeatures")
        
        self.featuresLayout = QVBoxLayout(self.groupBoxFeatures)
        
        self.featuresLabel = QLabel(self.groupBoxFeatures)
        self.featuresLabel.setObjectName(u"featuresLabel")
        self.featuresLayout.addWidget(self.featuresLabel)
        
        self.featureComboBox = QComboBox(self.groupBoxFeatures)
        self.featureComboBox.setObjectName(u"featureComboBox")
        self.featureComboBox.addItems(["Основные демографические", 
                                      "Полный набор признаков", 
                                      "Демография + Доходы + Авто",
                                      "Все доступные данные"])
        self.featuresLayout.addWidget(self.featureComboBox)
        
        self.verticalLayout_3.addWidget(self.groupBoxFeatures)
        
        # Кнопки
        self.combineButton = QPushButton(self.tabData)
        self.combineButton.setObjectName(u"combineButton")
        self.combineButton.setEnabled(False)
        self.verticalLayout_3.addWidget(self.combineButton)
        
        # Вывод объединенных данных
        self.dataOutput = QTextEdit(self.tabData)
        self.dataOutput.setObjectName(u"dataOutput")
        self.dataOutput.setReadOnly(True)
        self.verticalLayout_3.addWidget(self.dataOutput)
        
        self.tabWidget.addTab(self.tabData, "")
        
        # Вкладка 3: Анализ
        self.tabAnalysis = QWidget()
        self.tabAnalysis.setObjectName(u"tabAnalysis")
        self.verticalLayout_4 = QVBoxLayout(self.tabAnalysis)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        
        # Одномерный анализ
        self.groupBoxUnivariate = QGroupBox(self.tabAnalysis)
        self.groupBoxUnivariate.setObjectName(u"groupBoxUnivariate")
        
        self.univariateLayout = QVBoxLayout(self.groupBoxUnivariate)
        
        self.histogramLayout1 = QHBoxLayout()
        self.histogram1Label = QLabel(self.groupBoxUnivariate)
        self.histogram1Label.setObjectName(u"histogram1Label")
        self.histogramLayout1.addWidget(self.histogram1Label)
        
        self.histogram1Combo = QComboBox(self.groupBoxUnivariate)
        self.histogram1Combo.setObjectName(u"histogram1Combo")
        self.histogram1Combo.addItems(["Возраст", "Доход", "Рост", "Вес"])
        self.histogramLayout1.addWidget(self.histogram1Combo)
        self.univariateLayout.addLayout(self.histogramLayout1)
        
        self.histogramLayout2 = QHBoxLayout()
        self.histogram2Label = QLabel(self.groupBoxUnivariate)
        self.histogram2Label.setObjectName(u"histogram2Label")
        self.histogramLayout2.addWidget(self.histogram2Label)
        
        self.histogram2Combo = QComboBox(self.groupBoxUnivariate)
        self.histogram2Combo.setObjectName(u"histogram2Combo")
        self.histogram2Combo.addItems(["Доход", "Возраст", "Рост", "Вес"])
        self.histogram2Combo.setCurrentIndex(1)
        self.histogramLayout2.addWidget(self.histogram2Combo)
        self.univariateLayout.addLayout(self.histogramLayout2)
        
        # Настройки гистограмм
        self.binsLayout = QHBoxLayout()
        self.binsLabel = QLabel(self.groupBoxUnivariate)
        self.binsLabel.setObjectName(u"binsLabel")
        self.binsLayout.addWidget(self.binsLabel)
        
        self.binsSpinBox = QSpinBox(self.groupBoxUnivariate)
        self.binsSpinBox.setObjectName(u"binsSpinBox")
        self.binsSpinBox.setMinimum(10)
        self.binsSpinBox.setMaximum(100)
        self.binsSpinBox.setValue(30)
        self.binsLayout.addWidget(self.binsSpinBox)
        self.univariateLayout.addLayout(self.binsLayout)
        
        self.plotHistogramButton = QPushButton(self.groupBoxUnivariate)
        self.plotHistogramButton.setObjectName(u"plotHistogramButton")
        self.plotHistogramButton.setEnabled(False)
        self.univariateLayout.addWidget(self.plotHistogramButton)
        
        self.verticalLayout_4.addWidget(self.groupBoxUnivariate)
        
        # Многомерный анализ
        self.groupBoxMultivariate = QGroupBox(self.tabAnalysis)
        self.groupBoxMultivariate.setObjectName(u"groupBoxMultivariate")
        
        self.multivariateLayout = QVBoxLayout(self.groupBoxMultivariate)
        
        self.multivariateLabel = QLabel(self.groupBoxMultivariate)
        self.multivariateLabel.setObjectName(u"multivariateLabel")
        self.multivariateLayout.addWidget(self.multivariateLabel)
        
        self.graph1Layout = QHBoxLayout()
        self.graph1Label = QLabel(self.groupBoxMultivariate)
        self.graph1Label.setObjectName(u"graph1Label")
        self.graph1Layout.addWidget(self.graph1Label)
        
        self.graph1Combo = QComboBox(self.groupBoxMultivariate)
        self.graph1Combo.setObjectName(u"graph1Combo")
        self.graph1Combo.addItems(["Доход-Возраст-Пол", 
                                  "Автомобили-Доход-Пол", 
                                  "Преступления по городам",
                                  "Корреляционная матрица"])
        self.graph1Layout.addWidget(self.graph1Combo)
        self.multivariateLayout.addLayout(self.graph1Layout)
        
        self.plotMultivariateButton = QPushButton(self.groupBoxMultivariate)
        self.plotMultivariateButton.setObjectName(u"plotMultivariateButton")
        self.plotMultivariateButton.setEnabled(False)
        self.multivariateLayout.addWidget(self.plotMultivariateButton)
        
        self.verticalLayout_4.addWidget(self.groupBoxMultivariate)
        
        # Вывод анализа
        self.analysisOutput = QTextEdit(self.tabAnalysis)
        self.analysisOutput.setObjectName(u"analysisOutput")
        self.analysisOutput.setReadOnly(True)
        self.verticalLayout_4.addWidget(self.analysisOutput)
        
        self.tabWidget.addTab(self.tabAnalysis, "")
        
        self.verticalLayout.addWidget(self.tabWidget)
        
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Анализатор Murder Mystery", None))
        self.titleLabel.setText(QCoreApplication.translate("MainWindow", u"Анализ базы данных Murder Mystery", None))
        
        # Вкладка 1
        self.groupBoxConnection.setTitle(QCoreApplication.translate("MainWindow", u"Подключение к PostgreSQL", None))
        self.hostLabel.setText(QCoreApplication.translate("MainWindow", u"Хост:", None))
        self.portLabel.setText(QCoreApplication.translate("MainWindow", u"Порт:", None))
        self.databaseLabel.setText(QCoreApplication.translate("MainWindow", u"База:", None))
        self.userLabel.setText(QCoreApplication.translate("MainWindow", u"Пользователь:", None))
        self.passwordLabel.setText(QCoreApplication.translate("MainWindow", u"Пароль:", None))
        self.testButton.setText(QCoreApplication.translate("MainWindow", u"Тест подключения", None))
        self.loadAllButton.setText(QCoreApplication.translate("MainWindow", u"Загрузить ВСЕ данные", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabConnection), QCoreApplication.translate("MainWindow", u"Подключение", None))
        
        # Вкладка 2
        self.groupBoxFeatures.setTitle(QCoreApplication.translate("MainWindow", u"Настройка объединения данных", None))
        self.featuresLabel.setText(QCoreApplication.translate("MainWindow", u"Тип объединения признаков:", None))
        self.combineButton.setText(QCoreApplication.translate("MainWindow", u"Объединить данные", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabData), QCoreApplication.translate("MainWindow", u"Данные", None))
        
        # Вкладка 3
        self.groupBoxUnivariate.setTitle(QCoreApplication.translate("MainWindow", u"Одномерный анализ (2 гистограммы)", None))
        self.histogram1Label.setText(QCoreApplication.translate("MainWindow", u"Гистограмма 1:", None))
        self.histogram2Label.setText(QCoreApplication.translate("MainWindow", u"Гистограмма 2:", None))
        self.binsLabel.setText(QCoreApplication.translate("MainWindow", u"Количество бинов:", None))
        self.plotHistogramButton.setText(QCoreApplication.translate("MainWindow", u"Построить гистограммы", None))
        self.groupBoxMultivariate.setTitle(QCoreApplication.translate("MainWindow", u"Многомерный анализ (2 графика)", None))
        self.multivariateLabel.setText(QCoreApplication.translate("MainWindow", u"Выбор типа графика:", None))
        self.graph1Label.setText(QCoreApplication.translate("MainWindow", u"График 1:", None))
        self.plotMultivariateButton.setText(QCoreApplication.translate("MainWindow", u"Построить график", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabAnalysis), QCoreApplication.translate("MainWindow", u"Анализ", None))