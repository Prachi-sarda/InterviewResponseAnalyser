from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import spacy
nlp = spacy.load("en_core_web_sm")
import pandas as pd 
import nltk # type: ignore
from textblob import TextBlob # type: ignore
import re
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from textstat import flesch_kincaid_grade # type: ignore


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove characters outside of ASCII range
    text = text.encode('ascii', 'ignore').decode('ascii')  # Further clean encoding artifacts

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

def sentence_structure(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return len(set(pos_tags))  # unique POS

df = pd.read_csv(r"C:\Users\prachi\Desktop\combined_dataset.csv")

df['Answer Quality'] = df['Answer Quality'].replace('bad', 'poor')
# Remove classes with a single instance, or combine them with similar categories if needed
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
y = df['Answer Quality'].astype(int)

df['Cleaned_Answer'] = df['Answer'].apply(preprocess_text)
df['Word_Count'] = df['Cleaned_Answer'].apply(compute_word_count)
df['Sentiment'] = df['Cleaned_Answer'].apply(compute_sentiment)
df['Sentence_Structure_Variability'] = df['Cleaned_Answer'].apply(sentence_structure)
df['Text_Complexity'] = df['Cleaned_Answer'].apply(compute_complexity)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Cleaned_Answer'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[['Word_Count', 'Sentiment','Text_Complexity','Sentence_Structure_Variability']])
scaled_df = pd.DataFrame(scaled_df)
new_df = pd.concat([scaled_df, df[['Cleaned_Answer']]], axis=1)

new_df = pd.concat([scaled_df, tfidf_df], axis=1)

new_df.columns = new_df.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.2, random_state=42)
# Train a decision tree classifier
clf = DecisionTreeClassifier(class_weight='balanced')  # Automatically handles imbalance
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))