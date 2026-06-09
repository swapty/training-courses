import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. НАСТРОЙКА И ЗАГРУЗКА ДАННЫХ
sns.set(style="whitegrid")

try:
    # Загрузка таблиц (укажи свои правильные пути к файлам)
    ratings = pd.read_csv(r'/ratings.csv')
    movies = pd.read_csv('/movies.csv')
    print("Данные успешно загружены!")
    print(f"Размер таблицы рейтингов: {ratings.shape}")
    print(f"Размер таблицы фильмов: {movies.shape}\n")
except FileNotFoundError:
    print("Ошибка: Файлы не найдены. Проверьте путь к папке с данными.")
    exit()


# 2. ВИЗУАЛИЗАЦИЯ И АНАЛИЗ (EDA)

# График 1: Распределение оценок
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=ratings, palette='viridis')
plt.title('Распределение пользовательских оценок')
plt.xlabel('Оценка')
plt.ylabel('Количество оценок')
plt.show()

# График 2: Топ-10 самых популярных фильмов
# Считаем количество оценок для каждого movieId и сразу даем колонке имя
movie_counts = ratings['movieId'].value_counts().reset_index(name='count')

# Объединяем полученную статистику со справочников фильмов, чтобы узнать названия
top_movies_df = pd.merge(movie_counts.head(10), movies, on='movieId')

plt.figure(figsize=(12, 8))
sns.barplot(y='title', x='count', data=top_movies_df, palette='coolwarm')
plt.title('Топ-10 самых оцениваемых фильмов')
plt.xlabel('Количество оценок')
plt.ylabel('Фильм')
plt.tight_layout()
plt.show()


# 3. АЛГОРИТМ РЕКОМЕНДАЦИЙ (CONTENT-BASED)


print("--- Шаг 1: Расчёт метрик для каждого фильма ---")
# Считаем среднюю оценку и общее количество оценок для каждого фильма
movie_stats = ratings.groupby('movieId').agg(
    mean_rating=('rating', 'mean'),
    count_ratings=('rating', 'count')
).reset_index()

# Объединяем метрики с названиями фильмов и их жанрами
full_movies_df = pd.merge(movie_stats, movies, on='movieId')

print("--- Шаг 2: Фильтрация аномалий и шума ---")
# Оставляем только те фильмы, у которых больше 50 оценок.
# Это защита от ситуации, когда фильм с 1 оценкой "5.0" незаслуженно попадает в топ.
min_ratings_threshold = 50
popular_movies = full_movies_df[full_movies_df['count_ratings'] > min_ratings_threshold]
print(f"Всего фильмов: {len(full_movies_df)}, после фильтрации осталось: {len(popular_movies)}\n")

print("--- Шаг 3: Функция генерации рекомендаций ---")
def get_recommendations_by_genre(genre_name, top_n=5):
    """
    Функция ищет популярные фильмы по заданному жанру 
    и сортирует их по среднему рейтингу от самого высокого.
    """
    # Фильтруем датафрейм: проверяем, содержится ли строка с жанром в колонке genres
    genre_df = popular_movies[popular_movies['genres'].str.contains(genre_name, case=False)]
    
    # Сортируем по среднему рейтингу в порядке убывания
    top_genre_movies = genre_df.sort_values(by='mean_rating', ascending=False)
    
    # Возвращаем красивую таблицу с топ-N результатами
    return top_genre_movies[['title', 'genres', 'mean_rating']].head(top_n)


# 4. ТЕСТИРОВАНИЕ СИСТЕМЫ

# Проверим, как система порекомендует фильмы для любителя Боевиков (Action)
target_genre = 'Action'
recommendations = get_recommendations_by_genre(target_genre, top_n=5)

print(f"Топ-5 рекомендаций для жанра '{target_genre}':")
print(recommendations.to_string(index=False))