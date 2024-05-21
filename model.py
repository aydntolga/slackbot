import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_excel("InventDatasetNew.xlsx")

with pd.ExcelFile("InventDatasetNew.xlsx") as xls:
    df1 = pd.read_excel(xls, sheet_name="Fails")
    df2 = pd.read_excel(xls, sheet_name="Sheet2")

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip()
        stop_words = set(stopwords.words('turkish'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        
        stemmer = PorterStemmer()
        stemmed_text = [stemmer.stem(word) for word in filtered_text]
       
        preprocessed_text = ' '.join(stemmed_text)
        return preprocessed_text
    else:
        return text

df2['FailSummary'] = df2['FailSummary'].apply(preprocess_text)
df2['FailType'] = df2['FailType'].apply(preprocess_text)
df2['Customer'] = df2['Customer'].apply(preprocess_text)
df2['Source'] = df2['Source'].apply(preprocess_text)
df2['SourceType'] = df2['SourceType'].apply(preprocess_text)
df2['Solution'] = df2['Solution'].apply(preprocess_text)

df2.fillna(0, inplace=True)
df2 = df2.astype(str)

X = df2[['Customer', 'Source', 'SourceType', 'FailType', 'FailSummary']]
y = df2['Solution']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

logger.info("Orijinal Solution değerleri: %s", y.unique())
logger.info("Kodlanmış Solution değerleri: %s", y_encoded)
logger.info("Label Encoding eşlemeleri: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

def build_bow_model(text_data):
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    bow_matrix = vectorizer.fit_transform(text_data)
    return vectorizer, bow_matrix

def update_pipeline(pipeline, bow_model):
    new_transformers = [
        ('encoder', OneHotEncoder(handle_unknown='ignore'), ['Customer', 'FailType']),
        ('bow_solution', bow_model[0], 'FailSummary'),  
        ('vectorizer_sourceType', TfidfVectorizer(preprocessor=preprocess_text), 'SourceType'),
        ('vectorizer_source', TfidfVectorizer(preprocessor=preprocess_text), 'Source')
    ]
    pipeline.steps[0] = ('preprocessor', ColumnTransformer(transformers=new_transformers, remainder='passthrough'))
    return pipeline

def retrain_model(X_train, y_train, pipeline):
    pipeline.fit(X_train, y_train)
    return pipeline

bow_model = build_bow_model(X_train['FailSummary'])  

pipelineMaps = Pipeline([
    ('preprocessor', None),  
    ('classifier', SVC(kernel='linear', probability=True))  
])

pipelineMaps = update_pipeline(pipelineMaps, bow_model)

pipelineMaps = retrain_model(X_train, y_train, pipelineMaps)

accuracy = pipelineMaps.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

try:
    joblib.dump(pipelineMaps, 'maps_updated.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Güncellenmiş Model ve Label Encoder başarıyla kaydedildi.")
except Exception as e:
    print(f"Model kaydedilirken bir hata oluştu: {e}")

