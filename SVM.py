import pandas as pd
import numpy as np
import nltk # type: ignore
from textblob import TextBlob# type: ignore
import re
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords# type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from imblearn.over_sampling import SMOTE
from textstat import flesch_kincaid_grade# type: ignore

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | set(ENGLISH_STOP_WORDS)
def preprocess_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    text = text.encode('ascii', 'ignore').decode('ascii') 
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def compute_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)

def compute_word_count(text):
    return len(text.split())

def compute_complexity(text):
    return flesch_kincaid_grade(text)

df = pd.read_csv(r"C:\Users\prachi\Desktop\combined_dataset.csv")

# Preprocesses the output column
df['Answer Quality'] = df['Answer Quality'].replace('bad', 'poor')
df['Answer Quality'] = df['Answer Quality'].str.strip().str.lower()
df = df[df['Answer Quality'] != 'bad']
df = df[df['Answer Quality'] != 'good ']

quality_mapping = {
    'good': 1,
    'average': 0,
    'poor': -1
}

df['Answer Quality'] = df['Answer Quality'].map(quality_mapping)
df = df[pd.to_numeric(df['Answer Quality'], errors='coerce').notnull()]
df['Answer Quality'] = df['Answer Quality'].astype(int)

y=df[['Answer Quality']]

df['Cleaned_Answer'] = df['Answer'].apply(preprocess_text)
df['Word_Count'] = df['Cleaned_Answer'].apply(compute_word_count)
df['Sentiment'] = df['Cleaned_Answer'].apply(compute_sentiment)
df['Text_Complexity'] = df['Cleaned_Answer'].apply(compute_complexity)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Cleaned_Answer'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[['Word_Count', 'Sentiment','Text_Complexity','Sentence_Structure_Variability']])
scaled_df = pd.DataFrame(scaled_df)
X = pd.concat([scaled_df, tfidf_df], axis=1)
X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

low_importance_features = feature_importances[feature_importances['importance'] < 0.05].index
X_train_reduced = X_train.drop(columns=low_importance_features)
X_test_reduced = X_test.drop(columns=low_importance_features)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_reduced, y_train)

svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_resampled, y_resampled)

y_pred = svm_model.predict(X_test_reduced)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))