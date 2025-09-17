import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import TextPreprocessor

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    return df

def train_models():
    # Load and preprocess data
    df = load_data('../data/spam.csv')
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    
    # Convert labels to binary
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label_binary'], test_size=0.2, random_state=42, stratify=df['label_binary']
    )
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Multinomial NB': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(report)
        
        # Create images directory if it doesn't exist
        os.makedirs('../images', exist_ok=True)
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'../images/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save best model and vectorizer
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    joblib.dump(best_model, '../models/spam_classifier.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    
    print(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    train_models()