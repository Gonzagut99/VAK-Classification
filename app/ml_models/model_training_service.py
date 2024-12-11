from pathlib import Path

import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import joblib
from app.utils.data_preparation import DataPreparationService
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, Dense, LSTM, Bidirectional
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model

class VAKClassificationModelTraining:
    def __init__(self):
        self.data_service = DataPreparationService()
        self.model = None
        self.model_path = Path().resolve().joinpath("app/output/vak_model.keras") or Path().resolve().resolve().joinpath("app/ml_models/output/vak_model.keras")
        self.label_encoder_path = Path().resolve().joinpath("app/output/label_encoder.pkl") or Path().resolve().resolve().joinpath("app/ml_models/output/label_encoder.pkl")
        if self.model_path.exists() and self.label_encoder_path.exists():
            self.load_model()
        else:
            self.build_neural_network()
        
    def build_neural_network(self):
        X_train, X_test, y_train, y_test, embeddings_matrix, hits, misses, vocabSize=self.data_service.prepare_data()
        self.model = Sequential()
        self.model.add(Embedding(
            vocabSize,
            200,
            input_length=X_train.shape[1],
            weights=[embeddings_matrix],
            trainable=False
        ))
        self.model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(3, activation='softmax'))
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        #model.summary()
        history = self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=64,
            validation_data=(X_test, y_test),
            verbose=1
        )
        self.visualize_loss_accuracy(history)
        self.data_service.save_label_encoder()
        self.save_model()
        
    def visualize_loss_accuracy(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'g', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'g', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
        
    async def classify(self, sentence):
        
        sentence = self.data_service.clean_text(sentence)
        #print('Después de clean_text:', sentence)
        sentence = self.data_service.remove_stop_words(sentence)
        #print('Después de remove_stop_words:', sentence)
        sentence = self.data_service.remove_punctuation(sentence)
        #print('Después de remove_punctuation:', sentence)
        sentence = self.data_service.tokenizer.texts_to_sequences([sentence])
        #print('Secuencia tokenizada:', sentence)
        sentence = pad_sequences(sentence, maxlen=48, truncating='pre')
        print('Secuencia con padding:', sentence)
        
        prediction = self.model.predict(sentence)
        maxarg = np.argmax(prediction, axis=-1)
        result = self.data_service.le.inverse_transform(maxarg)[0]
        print('prediction', prediction)
        # print('maxarg', maxarg)
        # print('result array', self.data_service.le.inverse_transform(maxarg))
        print('result', result)
        # result = self.data_service.le.inverse_transform([np.argmax(self.model.predict(sentence), axis=-1)])[0]
        return result
    
    def save_model(self):
        self.model.save(self.model_path)    
        
    def load_model(self):
        self.model = load_model(self.model_path)
    
# DATA_DIR = Path().resolve().joinpath("app/data") or Path().resolve().resolve().joinpath("app/data")

if __name__ == "__main__":
    # data_file = DATA_DIR.joinpath("student_scores.csv")
    model_service = VAKClassificationModelTraining()
    model_service.build_neural_network()
    result = model_service.classify("I like watching movies")
    print(f'The VAK type is {result}')