import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from unidecode import unidecode

# Assicurati di avere le risorse necessarie di NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')

    def _clean_text(self, text: str) -> str:
        text = unidecode(text)
        text = self.punctuation_pattern.sub('', text)
        text = text.lower()
        text = re.sub(r'[0-9]', '', text) # Rimuovi i numeri
        return text

    def _remove_stopwords(self, text: str) -> str:
        words = nltk.word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def _stem_text(self, text: str) -> str:
        words = nltk.word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def preprocess(self, text: str) -> str:
        text = self._clean_text(text)
        text = self._remove_stopwords(text)
        text = self._stem_text(text)
        return text

