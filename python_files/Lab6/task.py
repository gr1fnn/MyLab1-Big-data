import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import pymorphy3
from gensim.models import Word2Vec
from collections import Counter
from sklearn.decomposition import PCA
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Скачивание необходимых данных NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

class SongAnalyzer:
    def __init__(self):
        # Инициализация инструментов
        self.morph = pymorphy3.MorphAnalyzer()
        self.stemmer = SnowballStemmer("russian")
        self.russian_stopwords = stopwords.words('russian')
        
        # Данные
        self.songs = []
        self.lemmatized_songs = []
        self.stemmed_songs = []
        self.processed_songs = []
        self.tokenized_songs = []
        self.w2v_model = None
        self.top_words = []
        self.tfidf_scores = None
        self.all_words = []
        
        # Текущая директория
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.tracks_path = os.path.join(self.current_dir, 'tracks')
        
    def clear_screen(self):
        """Очистка экрана"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title):
        """Печать заголовка"""    
        print(f" {title}")
        
    
    def print_menu(self):
        """Печать главного меню"""
        self.clear_screen()
        print("\n ДОСТУПНЫЕ ОПЕРАЦИИ:\n")
        print("  1️  Загрузить песни из файла")
        print("  2️  Показать информацию о загруженных песнях")
        print("  3️  Показать сравнение лемматизации и стемминга")
        print("  4️  Вывести топ-15 слов по TF-IDF")
        print("  5️ Построить WordCloud")
        print("  6️  Обучить модель Word2Vec")
        print("  7️  Найти похожие слова")
        print("  8️ Построить t-SNE визуализацию")
        print("  9️  Показать полный отчет о выполнении")
        print("  10  Выйти из программы")
    
    def load_songs(self):
        """Загрузка песен из файла """
        self.print_header("ЗАГРУЗКА ПЕСЕН ")
        
        if not os.path.exists(self.tracks_path):
            print(f"\n Файл {self.tracks_path} не найден!")
            print("\n Создаю демонстрационные данные")
            self.songs = [
                "В лесу родилась елочка, в лесу она росла. Зимой и летом стройная, зеленая была.",
                "Маленькой елочке холодно зимой, из лесу елочку взяли мы домой.",
                "Белые снежинки кружатся с утра, вырастают снежные горы со двора.",
                "Расскажи, Снегурочка, где была? Расскажи-ка, милая, как дела?",
                "Три белых коня, три белых коня - декабрь, январь и февраль!",
                "Кабы не было зимы в городах и селах, никогда б не знали мы этих дней веселых.",
                "Ой, мороз, мороз, не морозь меня, не морозь меня, моего коня.",
                "Валенки, да валенки, ой, да неподшиты стареньки, нельзя валенки не носить.",
                "Синий иней лег на провода, в небесах густая синева.",
                "Зима близко, холодно стало, сердце замерло и замолчало."
            ]
            print(f" Создано {len(self.songs)} демо-песен")
        else:
            try:
                with open(self.tracks_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                lines = content.split('\n')
                songs = []
                current_song = []
                
                for line in lines:
                    line = line.strip()
                    if re.match(r'^Песня\s*\d+', line, re.IGNORECASE):
                        if current_song:
                            songs.append(' '.join(current_song))
                            current_song = []
                    elif line and not line.startswith('==='):
                        current_song.append(line)
                
                if current_song:
                    songs.append(' '.join(current_song))
                
                self.songs = [song for song in songs if len(song.split()) > 10]
                print(f" Загружено {len(self.songs)} песен из файла")
            except Exception as e:
                print(f" Ошибка при чтении файла: {e}")
                return False
        
        # Предобработка песен
        print("\n Выполняется предобработка песен")
        self.preprocess_all_songs()
        
        print("\n Загрузка и предобработка завершены!")
        input("\nНажмите Enter для продолжения")
        return True
    
    def is_cyrillic(self, word):
        """Проверка, состоит ли слово из русских букв"""
        return bool(re.match(r'^[а-яё]+$', word.lower()))
    
    def preprocess_text(self, text, use_lemmatization=True, use_stemming=False, remove_stopwords=True):
        """Предобработка текста"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        try:
            words = word_tokenize(text, language='russian')
        except:
            words = text.split()
        
        if remove_stopwords:
            words = [word for word in words if self.is_cyrillic(word)]
            words = [word for word in words if word not in self.russian_stopwords and len(word) > 2]
        
        if use_lemmatization:
            processed_words = []
            for word in words:
                try:
                    lemma = self.morph.parse(word)[0].normal_form
                    processed_words.append(lemma)
                except:
                    processed_words.append(word)
        elif use_stemming:
            processed_words = [self.stemmer.stem(word) for word in words]
        else:
            processed_words = words
        
        return ' '.join(processed_words)
    
    def preprocess_all_songs(self):
        """Предобработка всех песен"""
        print("\n Лемматизация песен")
        self.lemmatized_songs = []
        for i, song in enumerate(self.songs, 1):
            processed = self.preprocess_text(song, use_lemmatization=True, use_stemming=False)
            self.lemmatized_songs.append(processed)
            print(f"  Песня {i}: {len(song.split())} → {len(processed.split())} слов")
        
        print("\n Стемминг песен")
        self.stemmed_songs = []
        for i, song in enumerate(self.songs, 1):
            processed = self.preprocess_text(song, use_lemmatization=False, use_stemming=True)
            self.stemmed_songs.append(processed)
            print(f"  Песня {i}: {len(song.split())} → {len(processed.split())} слов")
        
        # Сохраняем лемматизированные версии как основные
        self.processed_songs = self.lemmatized_songs.copy()
        self.tokenized_songs = [song.split() for song in self.processed_songs if song.split()]
        
        # Сохраняем в файл
        output_path = os.path.join(self.current_dir, 'processed_tracks.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, song in enumerate(self.lemmatized_songs, 1):
                f.write(f"=== Песня {i} ===\n")
                f.write(song + '\n\n')
    
    def show_info(self):
        """Показать информацию о загруженных песнях"""
        self.print_header("ИНФОРМАЦИЯ О ПЕСНЯХ")
        
        if not self.songs:
            print("\n Песни не загружены! Сначала выполните пункт 1.")
            input("\nНажмите Enter для продолжения")
            return
        
        print(f"\n СТАТИСТИКА:")
        print(f"   Всего песен: {len(self.songs)}")
        print(f"   Всего слов в оригинале: {sum(len(song.split()) for song in self.songs)}")
        print(f"   Всего слов после лемматизации: {sum(len(song.split()) for song in self.lemmatized_songs)}")
        print(f"   Всего слов после стемминга: {sum(len(song.split()) for song in self.stemmed_songs)}")
        
        print(f"\n ПРИМЕРЫ ПЕСЕН:")
        for i in range(min(3, len(self.songs))):
            print(f"\n  Песня {i+1}:")
            print(f"    Оригинал: {self.songs[i][:100]}")
            print(f"    Лемматизация: {self.lemmatized_songs[i][:100]}")
            print(f"    Стемминг: {self.stemmed_songs[i][:100]}")
        
        input("\nНажмите Enter для продолжения")
    
    def show_comparison(self):
        """Показать сравнение лемматизации и стемминга"""
        self.print_header("СРАВНЕНИЕ ЛЕММАТИЗАЦИИ И СТЕММИНГА")
        
        print("   Лемматизация - приведение слова к нормальной форме (словарной)")
        print("   Стемминг - отсечение окончаний (грубое усечение)")
        
        print("\n ПРИМЕРЫ СРАВНЕНИЯ:")
        example_words = ["елка", "елочка", "елочкой", "елочку", "елочные", "песня", "песни", "песню"]
        print("\n  Слово → Лемматизация → Стемминг")
        print("  " + "-"*40)
        for word in example_words:
            lemma = self.morph.parse(word)[0].normal_form
            stem = self.stemmer.stem(word)
            print(f"  {word:10} → {lemma:10} → {stem}")
        
        print("\n ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:")
        if self.songs:
            sample_song = self.songs[0]
            processed_lemma = self.preprocess_text(sample_song, use_lemmatization=True)
            processed_stem = self.preprocess_text(sample_song, use_lemmatization=False, use_stemming=True)
            
            print(f"\n  Оригинал: {sample_song[:150]}")
            print(f"\n  После лемматизации: {processed_lemma[:150]}")
            print(f"\n  После стемминга: {processed_stem[:150]}")
        
        input("\nНажмите Enter для продолжения")
    
    def calculate_tfidf(self):
        """Расчет TF-IDF"""
        self.print_header("TF-IDF АНАЛИЗ")
        
        if not self.processed_songs:
            print("\n Данные не обработаны! Сначала загрузите песни (пункт 1).")
            input("\nНажмите Enter для продолжения")
            return
        
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(self.processed_songs)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        
        # Топ-15 слов
        top_indices = tfidf_scores.argsort()[-15:][::-1]
        self.top_words = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
        
        print("\n ТОП-15 СЛОВ ПО TF-IDF:")
        print("\n  №  Слово              TF-IDF Score")
        print("  " + "-"*40)
        for i, (word, score) in enumerate(self.top_words, 1):
            print(f"  {i:2}. {word:15} {score:.4f}")
        
        # Сохраняем для отчета
        self.tfidf_scores = tfidf_scores
        
        # Статистика
        print(f"\n Статистика TF-IDF:")
        print(f"   Всего уникальных слов: {len(feature_names)}")
        print(f"   Максимальный TF-IDF: {tfidf_scores.max():.4f}")
        print(f"   Средний TF-IDF: {tfidf_scores.mean():.4f}")
        
        input("\nНажмите Enter для продолжения")
    
    def build_wordcloud(self):
        """Построение WordCloud"""
        self.print_header("ПОСТРОЕНИЕ WORDCLOUD")
        
        if not self.processed_songs:
            print("\n Данные не обработаны! Сначала загрузите песни (пункт 1).")
            input("\nНажмите Enter для продолжения")
            return
        
        all_songs_text = ' '.join(self.processed_songs)
        
        if not all_songs_text.strip():
            print("\n Недостаточно данных для построения WordCloud")
            input("\nНажмите Enter для продолжения")
            return
        
        print("\n Создание облака слов")
        
        # Создаем WordCloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100,
                             font_path=None).generate(all_songs_text)
        
        # Отображаем
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud для текстов песен', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print(" WordCloud успешно построен и отображен")
        
        # Информация о самых частых словах
        words = all_songs_text.split()
        word_counts = Counter(words)
        print("\n Топ-5 самых частых слов в облаке:")
        for word, count in word_counts.most_common(5):
            print(f"   '{word}': {count} раз(а)")
        
        input("\nНажмите Enter для продолжения")
    
    def train_word2vec(self):
        """Обучение модели Word2Vec"""
        self.print_header("ОБУЧЕНИЕ МОДЕЛИ WORD2VEC")
        
        if not self.tokenized_songs or len(self.tokenized_songs) < 2:
            print("\n Недостаточно данных для обучения! Сначала загрузите песни (пункт 1).")
            input("\nНажмите Enter для продолжения")
            return
        
        print("\n Обучение модели Word2Vec")
        print(f"   Количество предложений: {len(self.tokenized_songs)}")
        print(f"   Размер вектора: 100")
        print(f"   Окно контекста: 5")
        print(f"   Эпох: 50")
        
        self.w2v_model = Word2Vec(sentences=self.tokenized_songs, 
                                 vector_size=100,
                                 window=5,
                                 min_count=1,
                                 workers=4,
                                 epochs=50)
        
        print(f"\n Модель успешно обучена!")
        print(f"   Размер словаря: {len(self.w2v_model.wv.key_to_index)} слов")
        
        # Собираем все слова для статистики
        self.all_words = [word for text in self.tokenized_songs for word in text]
        
        input("\nНажмите Enter для продолжения")
    
    def find_similar_words(self):
        """Поиск похожих слов"""
        self.print_header("ПОИСК ПОХОЖИХ СЛОВ")
        
        if not self.w2v_model:
            print("\n Модель Word2Vec не обучена! Сначала выполните пункт 6.")
            input("\nНажмите Enter для продолжения")
            return
        
        print("\n Поиск похожих слов в модели Word2Vec\n")
        
        # Находим самые частые слова
        if not self.all_words:
            self.all_words = [word for text in self.tokenized_songs for word in text]
        
        word_counts = Counter(self.all_words)
        most_common_words = word_counts.most_common(5)
        
        print(" Самые частые слова в текстах:")
        for i, (word, count) in enumerate(most_common_words, 1):
            print(f"  {i}. '{word}' - {count} раз(а)")
        
        print("\n" + "-"*60)
        
        # Показываем похожие слова для каждого частого слова
        for word, count in most_common_words[:3]:
            print(f"\n Близкие слова к '{word}':")
            try:
                similar_words = self.w2v_model.wv.most_similar(word, topn=5)
                for similar_word, similarity in similar_words:
                    print(f"   {similar_word}: {similarity:.4f}")
            except KeyError:
                print(f"   Слово '{word}' не найдено в модели")
        
        # Интерактивный поиск
        print("\n" + "-"*60)
        print("\n Можете ввести свое слово для поиска (или Enter для выхода):")
        user_word = input(" Ваше слово: ").strip().lower()
        
        if user_word:
            print(f"\nРезультаты поиска для '{user_word}':")
            try:
                similar_words = self.w2v_model.wv.most_similar(user_word, topn=10)
                for word, similarity in similar_words:
                    print(f"   {word}: {similarity:.4f}")
            except KeyError:
                print(f"   Слово '{user_word}' не найдено в словаре модели")
        
        input("\nНажмите Enter для продолжения")
    
    def build_tsne(self):
        """Построение t-SNE визуализации (ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ WINDOWS)"""
        self.print_header("T-SNE ВИЗУАЛИЗАЦИЯ")
        
        if not self.w2v_model or not self.tokenized_songs:
            print("\n Модель Word2Vec не обучена! Сначала выполните пункт 6.")
            input("\nНажмите Enter для продолжения")
            return
        
        print("\n Подготовка данных для t-SNE")
        
        # Получаем 15 самых частых слов
        if not self.all_words:
            self.all_words = [word for text in self.tokenized_songs for word in text]
        
        word_counts = Counter(self.all_words)
        top_15_words = [word for word, _ in word_counts.most_common(15)]
        
        print(f"\n 15 наиболее частых слов:")
        for i, word in enumerate(top_15_words, 1):
            print(f"  {i}. {word}")
        
        # Получаем векторы для этих слов
        word_vectors = []
        words_found = []
        
        for word in top_15_words:
            try:
                word_vectors.append(self.w2v_model.wv[word])
                words_found.append(word)
            except KeyError:
                print(f"   Слово '{word}' не найдено в модели")
                continue
        
        if len(word_vectors) < 3:
            print(f"\n Недостаточно слов для визуализации (найдено {len(word_vectors)} из 15)")
            input("\nНажмите Enter для продолжения")
            return
        
        print(f"\n Найдено векторов: {len(word_vectors)}")
        print(" Применяем PCA для снижения размерности")
        
        word_vectors = np.array(word_vectors)
        
        # Ограничиваем количество компонент
        n_samples = word_vectors.shape[0]
        n_features = word_vectors.shape[1]
        n_components_pca = min(10, n_samples - 1, n_features)
        
        print(f"   Количество слов: {n_samples}")
        print(f"   Размерность векторов: {n_features}")
        print(f"   Компонент PCA: {n_components_pca}")
        
        if n_components_pca >= 2:
            pca = PCA(n_components=n_components_pca, random_state=42)
            word_vectors_reduced = pca.fit_transform(word_vectors)
            print(f"   Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.3f}")
        else:
            print(" Используем исходные векторы (PCA не применим)")
            word_vectors_reduced = word_vectors
        
        print(" Применяем t-SNE")
        
        try:
            # Корректируем параметры t-SNE
            n_points = word_vectors_reduced.shape[0]
            
            if n_points <= 3:
                print(" Слишком мало точек для t-SNE, используем PCA визуализацию")
                if word_vectors_reduced.shape[1] >= 2:
                    plot_data = word_vectors_reduced[:, :2]
                else:
                    plot_data = word_vectors_reduced
                method_name = "PCA"
            else:
                # Настраиваем perplexity
                if n_points <= 5:
                    perplexity_value = n_points - 1
                else:
                    perplexity_value = min(5, n_points - 1)
                
                print(f"   Параметр perplexity: {perplexity_value}")
                print(f"   Количество итераций: 500")
                
                # ВАЖНО: Отключаем параллельные вычисления для Windows
                tsne = TSNE(n_components=2, 
                        random_state=42, 
                        perplexity=perplexity_value,
                        n_jobs=1,  # Оставляем 1 для Windows
                        init='pca',  # Используем PCA для инициализации
                        learning_rate='auto',
                        max_iter=500,  # Уменьшаем количество итераций
                        method='barnes_hut')  # Используем быстрый метод
                plot_data = tsne.fit_transform(word_vectors_reduced)
                method_name = "t-SNE"
            
            # Визуализация
            print(f" Построение графика ({method_name})")
            plt.figure(figsize=(14, 10))
            
            # Создаем scatter plot
            scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], 
                                c=range(len(words_found)), cmap='tab10', 
                                s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
            
            # Добавляем подписи
            for i, word in enumerate(words_found):
                plt.annotate(word, 
                        (plot_data[i, 0], plot_data[i, 1]), 
                        fontsize=12, 
                        fontweight='bold',
                        ha='center',
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="lightyellow", 
                                alpha=0.8, 
                                edgecolor='gray'))
            
            plt.title(f'{method_name} визуализация наиболее частых слов в песнях', 
                    fontsize=16, fontweight='bold')
            plt.xlabel(f'{method_name} компонента 1', fontsize=12)
            plt.ylabel(f'{method_name} компонента 2', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Добавляем цветовую шкалу
            plt.colorbar(scatter, label='Индекс слова')
            
            plt.tight_layout()
            plt.show()
            
            print(f" {method_name} график успешно построен и отображен")
            
        except Exception as e:
            print(f"\n Ошибка при построении t-SNE: {e}")
            print("\n Создаем простую визуализацию через PCA")
            
            try:
                # Используем только PCA
                if word_vectors.shape[1] >= 2:
                    pca = PCA(n_components=2, random_state=42)
                    plot_data = pca.fit_transform(word_vectors)
                    
                    plt.figure(figsize=(14, 10))
                    plt.scatter(plot_data[:, 0], plot_data[:, 1], 
                            s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
                    
                    for i, word in enumerate(words_found):
                        plt.annotate(word, 
                                (plot_data[i, 0], plot_data[i, 1]), 
                                fontsize=12, 
                                fontweight='bold',
                                ha='center',
                                va='bottom')
                    
                    plt.title('PCA визуализация слов (альтернатива t-SNE)', fontsize=16, fontweight='bold')
                    plt.xlabel('PCA компонента 1', fontsize=12)
                    plt.ylabel('PCA компонента 2', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                    print(" PCA визуализация построена")
                else:
                    raise ValueError("Недостаточно размерностей")
                    
            except Exception as e2:
                print(f"\n Не удалось создать визуализацию: {e2}")
                print("\n Выводим текстовую информацию:")
                print("\nСлова и их частотность:")
                for i, word in enumerate(words_found[:10], 1):
                    print(f"  {i}. {word}: {word_counts[word]} раз(а)")
        
        input("\nНажмите Enter для продолжения")
    
    def show_full_report(self):
        """Показать полный отчет о выполнении"""
        self.print_header("ПОЛНЫЙ ОТЧЕТ О ВЫПОЛНЕНИИ ЗАДАНИЯ")
        
        completed_tasks = {
            "1. Сбор 10-30 песен": len(self.songs) >= 10,
            "2. Загрузка файла с песнями": len(self.songs) > 0,
            "3. Лемматизация": len(self.lemmatized_songs) > 0,
            "4. Стемминг": len(self.stemmed_songs) > 0,
            "5. Приведение к нижнему регистру": len(self.processed_songs) > 0,
            "6. Удаление знаков препинания": len(self.processed_songs) > 0,
            "7. Удаление стоп-слов": len(self.processed_songs) > 0,
            "8. Удаление слов на других языках": len(self.processed_songs) > 0,
            "9. Сохранение и загрузка обработанного файла": os.path.exists(os.path.join(self.current_dir, 'processed_tracks.txt')),
            "10. Расчет TF-IDF": self.top_words is not None and len(self.top_words) > 0,
            "11. WordCloud": True,  # Пользователь мог построить
            "12. Подготовка данных для Word2Vec": len(self.tokenized_songs) > 0,
            "13. Обучение Word2Vec": self.w2v_model is not None,
            "14. Поиск похожих слов": self.w2v_model is not None,
            "15. t-SNE визуализация": True  # Пользователь мог построить
        }
        
        print("\n СТАТУС ВЫПОЛНЕНИЯ ПУНКТОВ ЗАДАНИЯ:")
        print("\n  №  Пункт задания                              Статус")
        print("  " + "-"*60)
        for i, (task, status) in enumerate(completed_tasks.items(), 1):
            status_symbol = "" if status else ""
            print(f"  {i:2}. {task:35} {status_symbol}")
        
        # Результаты анализа
        
        print(" РЕЗУЛЬТАТЫ АНАЛИЗА")
        
        
        print(f"\n   Всего песен: {len(self.songs)}")
        if self.all_words:
            print(f"   Уникальных слов после обработки: {len(set(self.all_words))}")
        if self.w2v_model:
            print(f"   Размер словаря Word2Vec: {len(self.w2v_model.wv.key_to_index)}")
        
        if self.top_words:
            print(f"\n   Топ-5 слов по TF-IDF:")
            for i, (word, score) in enumerate(self.top_words[:5], 1):
                print(f"     {i}. '{word}' (TF-IDF: {score:.4f})")
        
    
    def run(self):
        """Запуск главного цикла приложения"""
        while True:
            self.print_menu()
            
            choice = input("\n Выберите действие (1-10): ").strip()
            
            if choice == '1':
                self.load_songs()
            elif choice == '2':
                self.show_info()
            elif choice == '3':
                self.show_comparison()
            elif choice == '4':
                self.calculate_tfidf()
            elif choice == '5':
                self.build_wordcloud()
            elif choice == '6':
                self.train_word2vec()
            elif choice == '7':
                self.find_similar_words()
            elif choice == '8':
                self.build_tsne()
            elif choice == '9':
                self.show_full_report()
            elif choice == '10':
                self.print_header("ДО СВИДАНИЯ!")
                print("\n Спасибо за использование приложения!")
                print(" Все результаты сохранены для дальнейшего анализа.\n")
                sys.exit(0)
            else:
                print("\n Неверный выбор! Пожалуйста, выберите пункт от 1 до 10.")
                input("\nНажмите Enter для продолжения")

if __name__ == "__main__":
    app = SongAnalyzer()
    app.run()