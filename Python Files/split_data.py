#Create .csv File of Amazon Review Data
#@author Chance Simmons
#@version September 2018
import csv
import os
import json

TRAINING = 'Data/training.csv'
TEST = 'Data/test.csv'
TRAINING_LABEL = 'Data/train_label.csv'
TEST_LABEL = 'Data/test_label.csv'
VAL = 'Data/val.csv'
VAL_LABEL = 'Data/val_label.csv'
JSON = 'Data/reviews.json' 
MIN_DATA = 23000
MAX_DATA = 46000
MAX_VAL = 56000

def write_data(rating,text,text_file,label_file):
    """
    Writes the data to a csv file from the json file
    """
    with open(text_file,"a",newline='') as text_file, open(label_file,'a',newline='') as label:
        data_writer = csv.writer(text_file)
        label_writer = csv.writer(label)
        data = [text]
        label_list = [rating]
        data_writer.writerow(data)
        label_writer.writerow(label_list)
        
def sort_data():
    """
    Sorts data based off of postive or negative reviews
    """
    with open(JSON,'r') as json_file:
        total_negative = 0
        total_positive = 0
    
        for element in json_file:
            rating = int(json.loads(element)['overall'])
      
            if rating == 1 or rating == 2 or rating == 3:
                if total_negative < MIN_DATA:
                    write_data(0,json.loads(element)['reviewText'],
                               TRAINING,TRAINING_LABEL)
                    total_negative = total_negative + 1
                elif total_negative < MAX_DATA:
                    write_data(0,json.loads(element)['reviewText'],
                               TEST,TEST_LABEL)
                    total_negative = total_negative + 1
                elif total_negative < MAX_VAL:
                    write_data(0,json.loads(element)['reviewText'],
                               VAL,VAL_LABEL)
                    total_negative = total_negative + 1
            else:
                if total_positive < MIN_DATA:
                    write_data(1,json.loads(element)['reviewText'],
                               TRAINING,TRAINING_LABEL)
                    total_positive = total_positive + 1
                elif total_positive < MAX_DATA:
                    write_data(1,json.loads(element)['reviewText'],
                               TEST,TEST_LABEL)
                    total_positive = total_positive + 1 
                elif total_positive < MAX_VAL:
                    write_data(1,json.loads(element)['reviewText'],
                               VAL,VAL_LABEL)
                    total_positive = total_positive + 1 
                    
def main():
    sort_data()

main()