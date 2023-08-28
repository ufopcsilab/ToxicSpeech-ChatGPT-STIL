#!/usr/bin/python
# -*- encoding: utf-8 -*-

#!pip install simpletransformers

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch
from scipy.special import softmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

def main(seed_value=42,
         train_file='',
         test_file='',
         train_data_column=0,
         train_label_column=1,
         test_data_column=0,
         test_label_column=1,
         embedding_dimension=16,
         model_args_num_train_epochs=3,
         model_args_train_batch_size=8,
         model_args_eval_batch_size=8,
         model_args_overwrite_output_dir=True,
         model_args_save_steps=-1,
         model_args_save_model_every_epoch=False,
         model_args_learning_rate=3e-5,
         model_args_fp16=True,
         using_model='bert',
         pre_trainned_model=''
         ):
    
    if pre_trainned_model == 'neuralmind/bert-large-portuguese-cased' or pre_trainned_model == 'adalbertojunior/distilbert-portuguese-cased' or pre_trainned_model == 'pierreguillou/bert-large-cased-squad-v1.1-portuguese':
        # Set the seed for PyTorch
        torch.manual_seed(seed_value)

        # If you're using CUDA or cuDNN, also set the seed for those
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set the seed for NumPy
        np.random.seed(seed_value)

        # Set the seed for Python's random module
        random.seed(seed_value)


        # Load training and test data
        arquivo_csv_treino = train_file
        arquivo_csv_teste = test_file

        dados_treino = pd.read_csv(arquivo_csv_treino)
        dados_teste = pd.read_csv(arquivo_csv_teste)

        train_df = pd.DataFrame({'text': dados_treino.iloc[:, train_data_column], 'labels': dados_treino.iloc[:, train_label_column]})
        eval_df = pd.DataFrame({'text': dados_teste.iloc[:, test_data_column], 'labels': dados_teste.iloc[:, test_label_column]})

        print(train_df.head())

        print(eval_df['labels'].unique())
        print(eval_df['labels'].unique())

        # Create the ranking model
        model_args = {
            'num_train_epochs': model_args_num_train_epochs,
            'train_batch_size': model_args_train_batch_size,
            'eval_batch_size': model_args_eval_batch_size,
            'overwrite_output_dir': model_args_overwrite_output_dir,
            'save_steps': model_args_save_steps,
            'save_model_every_epoch': model_args_save_model_every_epoch,
            'learning_rate': model_args_learning_rate,
            'fp16': model_args_fp16,
        }

        model = ClassificationModel(
            using_model,
            pre_trainned_model,
            num_labels=2,
            args=model_args,
            use_cuda=True,  # If using a GPU, you can change this to True
        )

        # Train the model on training data
        model.train_model(train_df)

        # Evaluate the model on test data
        predictions, raw_outputs = model.predict(eval_df['text'].tolist())

        probs = softmax(raw_outputs, axis=1)

        # Add probability of predictions to test set
        # We use index 1 to select the column corresponding to the toxic class
        dados_teste['predict_proba'] = probs[:, 1]
        dados_teste.to_csv('data/new_dataset_prob.csv', index=False)

        predicted_labels = np.argmax(raw_outputs, axis=1)
        print("Rótulos únicos em predicted_labels:", np.unique(predicted_labels))

        report = classification_report(eval_df['labels'], predicted_labels)
        conf_matrix = confusion_matrix(eval_df['labels'], predicted_labels)

        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)
        pass

    elif pre_trainned_model == 'naive':
        # Load data
        arquivo_csv_treino = train_file
        dados_treino = pd.read_csv(arquivo_csv_treino)

        # Pre-processing of training data
        X = dados_treino.iloc[:, train_data_column]  # data column
        y = dados_treino.iloc[:, train_label_column]  # column of labels

        # Label encoding for 0 and 1
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        # Tokenization and sequencing of texts
        max_length = 280  # maximum string size
        vocab_size = 100000  # vocabulary size

        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

        # Set the model
        embedding_dim = embedding_dimension

        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = model_args_num_train_epochs
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=2)

        # Model evaluation on test data
        loss, accuracy = model.evaluate(X_val, y_val, verbose=2)
        print(f"Perda nos dados de teste: {loss:.4f}")
        print(f"Acurácia nos dados de teste: {accuracy:.4f}")
        pass

    else:
        print("Invalid technique.")
        return

if __name__ == "__main__":
    main(seed_value=42,
         train_file='../datas/train.csv',
         test_file='../datas/test.csv',
         train_data_column=0,
         train_label_column=1,
         test_data_column=0,
         test_label_column=1,
         embedding_dimension=16,
         model_args_num_train_epochs=3,
         model_args_train_batch_size=8,
         model_args_eval_batch_size=8,
         model_args_overwrite_output_dir=True,
         model_args_save_steps=-1,
         model_args_save_model_every_epoch=False,
         model_args_learning_rate=3e-5,
         model_args_fp16=True,
         using_model='bert',
         pre_trainned_model='neuralmind/bert-large-portuguese-cased'
    )
