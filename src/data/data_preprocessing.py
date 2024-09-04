import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import spmatrix

# Assicurati di avere le risorse necessarie di NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')

    def _clean_text(self, text: str) -> str:
        text = self.punctuation_pattern.sub('', text)
        text = text.lower()
        text = re.sub(r'\d+', '', text)
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

