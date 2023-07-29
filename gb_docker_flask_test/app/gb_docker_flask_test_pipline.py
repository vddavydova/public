import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Загрузка данных
data = pd.read_csv('gb_docker_flask_test/fake_job_postings.csv')

# Определение признаков и целевой переменной
features = ['description', 'company_profile', 'benefits']
target = 'fraudulent'

X = data[features]
y = data[target]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Преобразование текстовых признаков с помощью TF-IDF
    ('logreg', LogisticRegression())  # Модель логистической регрессии
])

# Обучение модели на обучающей выборке
pipeline.fit(X_train, y_train)

# Оценка точности модели на тестовой выборке
accuracy = pipeline.score(X_test, y_test)
print(f'Точность модели: {accuracy:.4f}')

# Сохранение обученного pipeline на диск
joblib.dump(pipeline, 'fake_job_postings_pipeline.pkl')
