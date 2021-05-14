import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model
import numpy as np
import wget
import os 

dataset_path = f"{os.getcwd()}\\dataset_csv.csv"
if not os.path.exists(dataset_path):
	wget.download("https://raw.githubusercontent.com/surajr/SarcasmDetection/master/Data/dataset_csv.csv")

sentences = []
labels = []

with open('dataset_csv.csv', 'r',encoding='utf8') as file:
    reader = csv.DictReader(file, delimiter = '\t')
    for row in reader:
      sentences.append(row['tweets'])
      labels.append(row['label'])

model = load_model('sarcasm_weights')
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "Massive Undetified Terrestrial Word"
sentences = sentences[0:1500]
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

#fitting the data to the tokenizer
tokenizer.fit_on_texts(sentences)

def Detect(input_sequence):
  sequences = tokenizer.texts_to_sequences(input_sequence)
  padded = pad_sequences(sequences, maxlen= max_length, padding=padding_type,truncating=trunc_type)
  result = model.predict(padded)
  numpytolist = np.round(result).tolist()
  sarcastic = ''.join(str(numpytolist[0][0]))
  if sarcastic=='1.0':
    return f'This Sentence is Sarcastic \nProbabilty:{np.amax(result)}'
  else:
    return f'This Sentence is not Sarcastic \nProbabilty:{np.amax(result)}'

while True:
	try:
		User = input("You: ")
		print(Detect(str(User)))
	except KeyboardInterrupt:
		print("\nBye")
		quit()