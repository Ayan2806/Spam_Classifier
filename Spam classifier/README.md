# SMS Spam Classifier

A machine learning model that classifies SMS messages as spam or ham with high accuracy and strong precision–recall performance.

## Project Structure

```
spam-classifier/
├── data/
│   └── spam.csv                 # Dataset
├── src/
│   ├── preprocess.py            # Text cleaning
│   ├── train.py                 # Model training
│   └── predict.py               # Prediction
├── models/                      # Trained models
├── requirements.txt             # Dependencies
└── README.md
```

## Installation

1. **Clone repository**
```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

4. **Add dataset**  
Download [spam.csv](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) to `data/` folder

## Usage

1. **Train model**
```bash
cd src
python train.py
```

2. **Make predictions**
```bash
python predict.py
```

## Example Output

```
Message: Win a free iPhone today!
Prediction: SPAM (98.7% confidence)

Message: Meeting tomorrow at 3pm
Prediction: HAM (99.9% confidence)
```

## Model Performance

- **Accuracy**: 98.2%
- **Precision**: 97.5% 
- **Recall**: 86.0%
- **F1-Score**: 91.3%

## Dependencies

```
numpy==1.26.0
pandas==2.1.1
scikit-learn==1.3.2
nltk==3.8.1
```
