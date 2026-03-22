import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pymorphy3
from gensim.models import Word2Vec
import warnings
import os
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Инициализация морфологического анализатора
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = stopwords.words('russian')

print("ЗАДАНИЕ 1: Анализ текстов песен")

# Получаем текущую директорию
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\nТекущая директория: {current_dir}")

# Пути к файлам
tracks_path = os.path.join(current_dir, 'tracks')
poems_path = os.path.join(current_dir, 'poems')

print(f"Путь к файлу с песнями: {tracks_path}")
print(f"Путь к файлу со стихами: {poems_path}")

def parse_text_file(filepath):
    """
    Парсинг текстового файла с разделителями "Песня X" или названиями стихов
    Возвращает список текстов
    """
    if not os.path.exists(filepath):
        print(f"Файл {filepath} не найден!")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return []
    
    print(f"Файл успешно прочитан. Размер: {len(content)} символов")
    
    # Разделяем по строкам
    lines = content.split('\n')
    
    texts = []
    current_text = []
    in_text = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Пропускаем пустые строки в начале
        if not line and not in_text:
            continue
        
        # Проверяем, является ли строка заголовком
        is_title = False
        
        # Проверка на заголовки песен
        if re.match(r'^Песня\s*\d+', line, re.IGNORECASE) or re.match(r'^Песнь\s*\d+', line, re.IGNORECASE):
            is_title = True
            print(f"  Найден заголовок песни: {line}")
        
        # Проверка на заголовки стихов (короткие строки без знаков препинания)
        elif line and len(line.split()) <= 4 and not re.search(r'[,.!?;:]', line):
            # Распространенные названия стихов
            common_poem_titles = ['птичка', 'узник', 'сказка', 'туча', 'зима', 'гонимы', 'опрятней', 
                                 'унылая', 'и.и.', 'пущину', 'холмах', 'грузии', 'евгений', 'онегин']
            if any(title in line.lower() for title in common_poem_titles) or (line and line[0].isupper() and len(line) < 30):
                is_title = True
                print(f"  Найден заголовок стиха: {line}")
        
        if is_title:
            # Если уже собирали текст, сохраняем его
            if current_text:
                texts.append(' '.join(current_text))
                print(f"    Сохранен текст длиной {len(current_text)} строк")
                current_text = []
            in_text = True
        elif line:
            # Добавляем строку к текущему тексту
            current_text.append(line)
        elif in_text and current_text:
            # Пустая строка - разделитель между текстами
            texts.append(' '.join(current_text))
            print(f"    Сохранен текст длиной {len(current_text)} строк (разделитель)")
            current_text = []
            in_text = False
    
    # Добавляем последний текст
    if current_text:
        texts.append(' '.join(current_text))
        print(f"    Сохранен последний текст длиной {len(current_text)} строк")
    
    # Фильтруем пустые тексты
    texts = [text for text in texts if len(text.split()) > 5]
    
    return texts

# Чтение файлов
print("\n1. Загрузка данных...")
tracks = parse_text_file(tracks_path)
poems = parse_text_file(poems_path)

print(f"\nЗагружено песен: {len(tracks)}")
for i, track in enumerate(tracks[:3], 1):
    print(f"  Песня {i}: {track[:100]}...")

print(f"\nЗагружено стихов: {len(poems)}")
for i, poem in enumerate(poems[:3], 1):
    print(f"  Стих {i}: {poem[:100]}...")

# Проверка, что данные загружены
if len(tracks) == 0 or len(poems) == 0:
    print("ВНИМАНИЕ: Не удалось загрузить данные!")
    exit()

# Функция предобработки текста
def preprocess_text(text, remove_stopwords=True):
    """
    Предобработка текста:
    - приведение к нижнему регистру
    - удаление знаков препинания
    - удаление стоп-слов
    - лемматизация
    """
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление знаков препинания и цифр
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Токенизация
    try:
        words = word_tokenize(text, language='russian')
    except:
        words = text.split()
    
    # Удаление стоп-слов и слов на других языках
    if remove_stopwords:
        # Проверка на русские слова
        words = [word for word in words if all('а' <= char <= 'я' for char in word) or word in ['fuck', 'bitch', 'doobie', 'la']]
        words = [word for word in words if word not in russian_stopwords and len(word) > 2]
    
    # Лемматизация
    lemmatized_words = []
    for word in words:
        try:
            lemma = morph.parse(word)[0].normal_form
            lemmatized_words.append(lemma)
        except:
            lemmatized_words.append(word)
    
    return ' '.join(lemmatized_words)

# Предобработка текстов песен
print("\n2. Проведение препроцессинга текстов песен...")
processed_tracks = []
for i, track in enumerate(tracks):
    if track.strip():
        processed = preprocess_text(track)
        processed_tracks.append(processed)
        print(f"  Песня {i+1}: {len(track.split())} слов -> {len(processed.split())} слов после обработки")

# Сохранение обработанных песен
with open(os.path.join(current_dir, 'processed_tracks.txt'), 'w', encoding='utf-8') as f:
    for i, track in enumerate(processed_tracks):
        f.write(f"=== Песня {i+1} ===\n")
        f.write(track + '\n\n')

print("\n3. Расчет TF-IDF для песен...")
if processed_tracks and any(processed_tracks):
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(processed_tracks)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # Топ-10 слов по TF-IDF
    top_indices = tfidf_scores.argsort()[-10:][::-1]
    top_words = [(feature_names[i], tfidf_scores[i]) for i in top_indices]

    print("\nТоп-10 слов по TF-IDF в песнях:")
    for word, score in top_words:
        print(f"  {word}: {score:.4f}")

    # WordCloud для песен
    print("\n4. Построение WordCloud для песен...")
    all_tracks_text = ' '.join(processed_tracks)
    if all_tracks_text.strip():
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100).generate(all_tracks_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud для текстов песен', fontsize=16)
        plt.show()
        print("WordCloud отображен на экране")

    # Word2Vec для песен
    print("\n5. Обучение модели Word2Vec на песнях...")
    tokenized_tracks = [text.split() for text in processed_tracks if text.split()]
    
    if len(tokenized_tracks) >= 2:
        w2v_model = Word2Vec(sentences=tokenized_tracks, 
                             vector_size=100,
                             window=5,
                             min_count=1,
                             workers=4,
                             epochs=50)

        # Поиск похожих слов
        if tokenized_tracks:
            # Находим самое частое слово для демонстрации
            all_words = [word for text in tokenized_tracks for word in text]
            from collections import Counter
            word_counts = Counter(all_words)
            most_common_word = word_counts.most_common(1)[0][0]
            
            print(f"\n6. Проверка близких слов к '{most_common_word}':")
            try:
                similar_words = w2v_model.wv.most_similar(most_common_word, topn=5)
                for word, similarity in similar_words:
                    print(f"  {word}: {similarity:.4f}")
            except KeyError:
                print(f"  Слово '{most_common_word}' не найдено в модели")
                # Берем первое слово из словаря
                if w2v_model.wv.key_to_index:
                    vocab_word = list(w2v_model.wv.key_to_index.keys())[0]
                    print(f"\nБлизкие слова к '{vocab_word}':")
                    similar_words = w2v_model.wv.most_similar(vocab_word, topn=5)
                    for word, similarity in similar_words:
                        print(f"  {word}: {similarity:.4f}")

        # t-SNE визуализация
        print("\n7. Построение графика t-SNE для 15 наиболее частых слов...")

try:
    # Получаем 15 самых частых слов
    all_words = [word for text in tokenized_tracks for word in text]
    word_counts = Counter(all_words)
    top_15_words = [word for word, _ in word_counts.most_common(15)]

    # Получаем векторы для этих слов
    word_vectors = []
    words_found = []
    for word in top_15_words:
        try:
            word_vectors.append(w2v_model.wv[word])
            words_found.append(word)
        except KeyError:
            continue

    if len(word_vectors) >= 5:
        word_vectors = np.array(word_vectors)
        
        # Используем PCA для предварительного снижения размерности
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, len(word_vectors[0])))
        word_vectors_pca = pca.fit_transform(word_vectors)
        
        # Применяем t-SNE с n_jobs=1
        tsne = TSNE(n_components=2, random_state=42, 
                    perplexity=min(5, len(word_vectors)-1),
                    n_jobs=1,
                    init='random',
                    learning_rate='auto')
        word_vectors_tsne = tsne.fit_transform(word_vectors_pca)
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        plt.scatter(word_vectors_tsne[:, 0], word_vectors_tsne[:, 1], c='red', s=100)
        
        for i, word in enumerate(words_found):
            plt.annotate(word, (word_vectors_tsne[i, 0], word_vectors_tsne[i, 1]), 
                        fontsize=12, fontweight='bold')
        
        plt.title('t-SNE визуализация 15 наиболее частых слов в песнях', fontsize=16)
        plt.xlabel('t-SNE компонента 1')
        plt.ylabel('t-SNE компонента 2')
        plt.grid(True, alpha=0.3)
        plt.show()
        print("t-SNE график отображен на экране")
    else:
        print("Недостаточно слов для t-SNE визуализации")
        
except Exception as e:
    print(f"Ошибка при построении t-SNE: {e}")
    print("Пропускаем t-SNE визуализацию")

print("ЗАДАНИЕ 2: Классификация текстов (песни vs стихи)")

# Предобработка стихов
print("\n1. Предобработка стихов...")
processed_poems = []
for i, poem in enumerate(poems):
    if poem.strip():
        processed = preprocess_text(poem)
        processed_poems.append(processed)
        print(f"  Стих {i+1}: {len(poem.split())} слов -> {len(processed.split())} слов после обработки")

# Создание DataFrame
print("\n2. Создание DataFrame с данными...")
# Создаем метки: 1 - песни, 0 - стихи
tracks_df = pd.DataFrame({
    'text': processed_tracks,
    'label': 1,  # песни
    'title': [f'Песня {i+1}' for i in range(len(processed_tracks))]
})

poems_df = pd.DataFrame({
    'text': processed_poems,
    'label': 0,  # стихи
    'title': [f'Стих {i+1}' for i in range(len(processed_poems))]
})

# Объединяем данные
df = pd.concat([tracks_df, poems_df], ignore_index=True)
# Перемешиваем данные
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Размер датасета: {len(df)} строк")
print(f"Песен: {len(df[df['label']==1])}")
print(f"Стихов: {len(df[df['label']==0])}")

print("\nПример данных:")
for i in range(min(3, len(df))):
    print(f"\n{df.iloc[i]['title']} ({'песня' if df.iloc[i]['label']==1 else 'стих'}):")
    print(f"  Текст: {df.iloc[i]['text'][:150]}...")

# TF-IDF векторизация для классификации
print("\n3. Векторизация текстов для классификации...")

# Проверяем, что есть данные для векторизации
if len(df) > 0 and df['text'].str.len().sum() > 0:
    vectorizer_clf = TfidfVectorizer(max_features=500, 
                                     min_df=1,
                                     max_df=0.9)
    X = vectorizer_clf.fit_transform(df['text'])
    y = df['label']

    print(f"Размерность матрицы признаков: {X.shape}")
    print(f"Количество признаков (слов): {len(vectorizer_clf.get_feature_names_out())}")

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=42,
                                                        stratify=y)

    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")

    # Обучение моделей
    print("\n4. Обучение моделей классификации...")

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVC': SVC(kernel='linear', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}

    for name, model in models.items():
        print(f"\nОбучение модели {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"  Точность (accuracy): {accuracy:.4f}")
        print(f"  Отчет по классификации:")
        print(classification_report(y_test, y_pred, target_names=['Стихи', 'Песни'], zero_division=0))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Стихи', 'Песни'],
                    yticklabels=['Стихи', 'Песни'])
        plt.title(f'Матрица ошибок - {name}')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.show()

    # Сравнение моделей
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")

    results_df = pd.DataFrame(list(results.items()), columns=['Модель', 'Точность'])
    results_df = results_df.sort_values('Точность', ascending=False)
    print("\nСравнение моделей по точности:")
    print(results_df.to_string(index=False))

    # Визуализация сравнения моделей
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df['Модель'], results_df['Точность'], color=['gold', 'silver', 'lightgreen', 'lightblue'])
    plt.title('Сравнение точности моделей классификации', fontsize=16)
    plt.xlabel('Модель')
    plt.ylabel('Точность')
    plt.ylim(0, 1.1)

    # Добавление значений на столбцы
    for bar, value in zip(bars, results_df['Точность']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.3f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("ВЫВОД ПО ИССЛЕДОВАНИЮ")

    best_model = results_df.iloc[0]['Модель']
    best_accuracy = results_df.iloc[0]['Точность']

    # Получаем топ-слова из песен для вывода
    top_words_str = "недостаточно данных"
    if processed_tracks:
        all_words = [word for text in tokenized_tracks for word in text]
        if all_words:
            word_counts = Counter(all_words)
            top_words_str = ', '.join([word for word, _ in word_counts.most_common(5)])