import joblib
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import TextPreprocessor

class SpamClassifier:
    def __init__(self):
        # Load models
        self.model = joblib.load('../models/spam_classifier.pkl')
        self.vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
        self.preprocessor = joblib.load('../models/preprocessor.pkl')
    
    def predict(self, text):
        # Preprocess text
        cleaned_text = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)
        probability = self.model.predict_proba(text_vectorized)
        
        return {
            'prediction': 'spam' if prediction[0] == 1 else 'ham',
            'spam_probability': probability[0][1],
            'ham_probability': probability[0][0]
        }

# Example usage
if __name__ == "__main__":
    classifier = SpamClassifier()
    
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim your prize.",
        "Hey, are we still meeting for lunch tomorrow?",
        "Your bank account has been suspended. Verify your identity at http://bank-security.com",
        "Can you pick up some milk on your way home?"
    ]
    
    for msg in test_messages:
        result = classifier.predict(msg)
        print(f"Message: {msg}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Spam probability: {result['spam_probability']:.4f}")
        print("---")