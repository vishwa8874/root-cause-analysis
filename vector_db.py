import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import os

VECTOR_STORE_PATH = './uploads/index.pkl'

def create_vector_store(filepath):
    df = pd.read_csv(filepath)
    if 'Description' not in df.columns:
        raise ValueError('CSV file must contain a "Description" column.')
    
    descriptions = df['Description'].fillna("").tolist()
    
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(descriptions)
    
    
    with open(VECTOR_STORE_PATH, 'wb') as f:
        pickle.dump((vectorizer, vectors, df), f)

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError('Vector store file does not exist.')
    
    with open(VECTOR_STORE_PATH, 'rb') as f:
        vectorizer, vectors, df = pickle.load(f)
    
    return VectorStore(vectorizer, vectors, df)

class VectorStore:
    def __init__(self, vectorizer, vectors, df):
        self.vectorizer = vectorizer
        self.vectors = vectors
        self.df = df

    def query(self, summary):
        summary_vector = self.vectorizer.transform([summary])
        cosine_similarities = linear_kernel(summary_vector, self.vectors).flatten()
        indices = cosine_similarities.argsort()[-5:][::-1]  
        results = [{'Description': self.df.iloc[i]['Description']} for i in indices]
        return results
