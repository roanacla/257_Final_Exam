# -*- coding: utf-8 -*-
"""Seekers_Spam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cfklm_-uVY2HzMJdmRk2sxMoxm0cHpTt
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import cloudpickle as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
np.random.seed(2018)
import nltk
# nltk.download('wordnet')
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import pickle
# nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize 
# nltk.download('punkt')
import nltk
from scipy import sparse
from urllib.request import urlopen
# nltk.download('averaged_perceptron_tagger')
from string import punctuation
class Seekers_Spam():
    def remove_stop_and_short_words(self,text):
      text = [word.lower() for word in text.split() if (word.lower() not in sw) and (len(word)>3)]
      return " ".join(text)
  
    def lemmatize_stemming(self,text):
      return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
  
    def remove_punctuation(self,text):
      translator = str.maketrans('', '', string.punctuation)
      return text.translate(translator)
    def tokenization(self,text):
      lst=text.split()
      return lst
  
    def process_data(self,text):
    
      return loaded_tdIdfModel.transform(text)

    def predict(self, text):
      dfrme = pd.DataFrame(index=[0], columns=['text'])
      dfrme['text'] = text
      predict=dfrme['text'].apply(self.tokenization)
      predict = predict.apply(lambda x: ''.join(i+' ' for i in x))
      loaded_tdIdfModel = cp.load(urlopen("https://github.com/alekhyaved/AlternusVera_ML/blob/main/TFidfvectorizer.sav?raw=true"))
      text = loaded_tdIdfModel.transform(predict)
      loaded_model = cp.load(urlopen("https://github.com/alekhyaved/AlternusVera_ML/blob/main/final_SpamModel.sav?raw=true"))
    # processedText = self.process_data(dfrme['text'])
      result = loaded_model.predict_proba(text.toarray())[:,1][0]
      # print(result)
      return result