from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import nltk
# print(nltk.downloaded())
import re
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

DATA_DIR = Path().resolve().joinpath("app/data") or Path().resolve().resolve().joinpath("app/data")

GLOVE_DIR = Path().resolve().joinpath("app/input") or Path().resolve().resolve().joinpath("app/input")

LABEL_ENCODERS_FILE_PATH = Path().resolve().joinpath("app/output/label_encoder.pkl") or Path().resolve().resolve().joinpath("app/output/label_encoder.pkl")

TOKENIZER_PATH = Path().resolve().joinpath("app/output/tokenizer.pkl") or Path().resolve().resolve().joinpath("app/output/tokenizer.pkl")

class DataPreparationService:
    def __init__(self):
        self.le = LabelEncoder()
        if not LABEL_ENCODERS_FILE_PATH.exists():
            print('label encoder file not found, downloading nltk data')
            nltk.download('omw-1.4')
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('all')
        else:
            self.le = joblib.load(LABEL_ENCODERS_FILE_PATH)
        
        self.file_path = f'{DATA_DIR}/dataset.csv'
        self.df: pd.DataFrame = None
        self.lemm = WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.x = None
        self.y = None
        self.glove_pathfile = f'{GLOVE_DIR}/glove.6B.200d.txt'
        self.tokenizer_path = TOKENIZER_PATH
        self.tokenizer = None
        # Cargar tokenizer si existe, sino crear uno nuevo
        if self.tokenizer_path.exists():
            with open(self.tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            print("Tokenizer cargado exitosamente")
        else:
            self.tokenizer = Tokenizer(oov_token='<OOV>')
            print("Nuevo tokenizer creado")
    
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['new_Sentence'] = self.df['Sentence'].apply(lambda x:x.lower())
        # self.df['new_Sentence'] = self.df['new_Sentence'].apply(lambda x: ' '.join([self.lemm.lemmatize(word) for word in x.split(' ')]))
        self.df['new_Sentence'] = self.df['new_Sentence'].apply(self.remove_stop_words)
        self.df['new_Sentence'] = self.df['new_Sentence'].apply(self.clean_text)
        self.df['new_Sentence'] = self.df['new_Sentence'].apply(self.remove_punctuation)
        self.df = self.tokenize_text(self.df)
        #return self.df
    
    def remove_stop_words(self, text):
        no_stop_words = [word for word in text.split(' ') if word not in self.stop_words]
        return ' '.join(no_stop_words)

    def clean_text(self, text):
        # return ' '.join([self.lemm.lemmatize(word) for word in text.split(' ')])
        text = re.sub(r'[^a-zA-Z ]', '', text)
        return text

    def remove_punctuation(self, text):
        #return re.sub(r'[^a-zA-Z0-9]', '', text)
        return re.sub(r'[^\w\s]', '', text)
    
    def tokenize_text(self, df:pd.DataFrame, input_col = 'new_Sentence', output_col = 'cleaned_word_list'):
        """
        takes a dataset and name of a column
        then tokenizes the text in the column of that dataset
        """
        df.loc[:, output_col] = df.loc[:, input_col].apply(lambda t:word_tokenize(t, language='english'))
        return df
    
    def prepare_data(self):
        self.load_data()
        self.x = self.df['new_Sentence']
        self.y = self.df['Type'] #VAK type
        self.y = self.le.fit_transform(self.y)
        self.y = to_categorical(self.y)
        
        text_train, text_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        X_train, X_test = self.tokenize_data(text_train, text_test, self.tokenizer)
        vocabSize = len(self.tokenizer.index_word) + 1
        embeddings_matrix, embeddings_matrix, hits, misses = self.read_GloVE_embeddings(self.glove_pathfile, vocabSize, self.tokenizer)
        return X_train, X_test, y_train, y_test, embeddings_matrix, hits, misses, vocabSize

    def save_label_encoder(self):
        joblib.dump(self.le, LABEL_ENCODERS_FILE_PATH)
    
    def tokenize_data (self, text_train, text_test, tokenizer):
        tokenizer.fit_on_texts(text_train)
        # Guardar tokenizer después de entrenar
        with open(self.tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sequences_train = tokenizer.texts_to_sequences(text_train)
        sequences_test = tokenizer.texts_to_sequences(text_test)
        X_train = pad_sequences(sequences_train, maxlen=48, truncating='pre')
        X_test = pad_sequences(sequences_test, maxlen=48, truncating='pre')
        
        return X_train, X_test
    
    def read_GloVE_embeddings(self, file_path, vocabsize, tokenizer):
        num_tokens = vocabsize
        embedding_dim = 200
        hits = 0
        misses = 0
        embeddings_index = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
        
        embeddings_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
    
        return embeddings_matrix, embeddings_matrix, hits, misses