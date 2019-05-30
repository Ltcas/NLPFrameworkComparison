#Average number of words per line in csv files
#@author Chance Simmons
#@version September 2018
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_data_train():
    """
    Gets the training data from the training csv
    
    Returns:
        list of strings for the data of each line
    """
    elements = []
    with open("Data/training.csv",'r') as csv_file:
        file_reader = csv.reader(csv_file)
        for line in file_reader:
            elements.append(line)
    return elements

def get_line_lengths_train():
    """
    Gets the line lengths from the training data
    
    Returns:
        list of lengths for each line
    """
    elements = get_data_train()
    line_lengths = []
    for element in elements:
        line = element[0]
        split_line = line.split(" ")
        line_lengths.append(len(split_line))
    return line_lengths

def get_data_test():
    """
    Gets the test data by line and returns the array of lines
    
    Return:
        list of strings for data of each line
    """
    elements = []
    with open("Data/test.csv",'r') as csv_file:
        file_reader = csv.reader(csv_file)
        for line in file_reader:
            elements.append(line)
    return elements

def get_line_lengths_test():
    """
    Gets the line lengths for the test data
    
    Return: 
        list of lengths for each line
    """
    elements = get_data_test()
    line_lengths = []
    for element in elements:
        line = element[0]
        split_line = line.split(" ")
        line_lengths.append(len(split_line))
    return line_lengths

def plot_train():
    """
    Plots the training data number of words per line
    """
    train_line = get_line_lengths_train()
    training_plot = sns.distplot(train_line)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title("Training Data")
    plt.show(training_plot)
    
def plot_test():
    """
    Plots the test data number of words per line
    """
    test_line = get_line_lengths_test()
    test_plot = sns.distplot(test_line)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title("Test Data")
    plt.show(test_plot)
    
def main():
    line_lengths = get_line_lengths_train()
    print("TRAIN: Average number of words for a review: " + 
          str(np.mean(line_lengths)))
    print("TRAIN: Max Number: " + str(np.amax(line_lengths)))
    print("TRAIN: Min Number: " + str(np.amin(line_lengths)))
    
    line_lengths = get_line_lengths_test()
    print("\nTEST: Average number of words for a review: " + 
          str(np.mean(line_lengths)))
    print("TEST: Max Number: " + str(np.amax(line_lengths)))
    print("TEST: Min Number: " + str(np.amin(line_lengths)))
    
    plot_train()
    plot_test()

main()