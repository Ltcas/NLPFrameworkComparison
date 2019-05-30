#Tokenizes the training and test data with Tensorflow
#@author Chance Simmons
#@version October 2018
import tensorflow as tf
import numpy as np
import csv
#from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

MAX_NUM = 10000
LINE_LENGTH = 250
complete_data = []
TRAIN_DATA = "Data/training.csv"
TEST_DATA = "Data/test.csv"
VAL_DATA = "Data/val.csv"      
TRAIN_TOKEN = "Data/train_token.csv"
TEST_TOKEN = "Data/test_token.csv"
VAL_TOKEN = "Data/val_token.csv"

def get_data(file):
    """
    Gets the data from the csv file and returns an array of the data
    
    Parameters:
        file: the file to read data from
    
    Return:
        data: list of lines from data file
    """
    data = []
    with open(file,'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            data.append(str(line))
            complete_data.append(str(line))
    return data

def write_data(file,data):
    """
    Writes the data to the tokenized version of the file
    
    Parameters:
        file(str): the file to write to
        data(str[]): the tokenized data to write to the file
    """
    with open(file,'a',newline='') as csv_file:
        writer = csv.writer(csv_file)
        for line in data:
            writer.writerow(line)
            
def main():
    train_data = get_data(TRAIN_DATA)
    test_data = get_data(TEST_DATA)
    val_data = get_data(VAL_DATA)
   
    tokenizer = Tokenizer(num_words=MAX_NUM, lower=True, split= " ")
    tokenizer.fit_on_texts(complete_data)
    
    train_tokenized = tokenizer.texts_to_sequences(train_data)
    test_tokenized = tokenizer.texts_to_sequences(test_data)
    val_tokenized = tokenizer.texts_to_sequences(val_data)
    
    train_padded = pad_sequences(train_tokenized,maxlen = LINE_LENGTH,
                                 padding = 'post')
    test_padded = pad_sequences(test_tokenized,maxlen = LINE_LENGTH,
                                padding = 'post')
    val_padded = pad_sequences(val_tokenized,maxlen = LINE_LENGTH,
                               padding = 'post')
    
    write_data(TRAIN_TOKEN,train_padded)
    write_data(TEST_TOKEN,test_padded)
    write_data(VAL_TOKEN,val_padded)
    
main()