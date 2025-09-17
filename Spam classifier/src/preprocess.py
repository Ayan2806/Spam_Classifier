import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text'):
        df['cleaned_text'] = df[text_column].apply(self.preprocess_text)
        return df