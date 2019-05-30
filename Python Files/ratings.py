#Analyzes the Amazon reviews data
#@author Chance Simmons
#@version September 2018
import json
import matplotlib.pyplot as plt
import numpy as np

def calculate_totals():
    """
    Calculates the total rankings from 1-5
    
    Returns:
         list of integers with the totals for each rank
    """
    totals = [0,0,0,0,0]
    with open('Data/reviews.json','r') as json_file:
        for element in json_file:
            index = int(json.loads(element)['overall']) - 1
            totals[index] = totals[index] + 1
    return totals
 
def calculate_pos_neg():
    """
    Calculates the total negative and positive reviews.
    
    A ranking of 1,2,or 3 will be a negative review and a ranking of 4 or 5
    will be a positive review.
    
    Returns:
        list of integers with the total positve and negative reviews
    """
    totals = [0,0]
    with open('Data/reviews.json','r') as json_file:
        for element in json_file:
            data = int(json.loads(element)['overall'])
            if data == 1 or data == 2 or data == 3:
                totals[0] += 1
            else:
                totals[1] += 1
    return totals

def plot_bar(labels,values):
    """
    Plots the ranking totals from 1 to 5
    
    Parameters:
        labels: list of strings with values 1-5
        values: list that holds the totals for each ranking
    """
    index = np.arange(len(labels))
    plt.bar(index, values)
    plt.xlabel('Review Ranking', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks(index, labels, fontsize=12)
    plt.title('Spread of Amazon Review Ratings')
    plt.show()

def plot_pos_neg(labels,values):
    """
    Plots the postive and negative ranking totals
    
    Parameters:
        labels: list of strings with values 0-1
        values: list that holds the totals for each ranking
    """
    index = np.arange(len(labels))
    plt.bar(index, values)
    plt.xlabel('Postive vs Negative', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks(index, labels, fontsize=12)
    plt.title("Spread of Postive and Negative Reviews")
    plt.show()
    print("Negative Total: " + str(values[0]) + "\n")
    print("Positive Total: " + str(values[1]))
    
def main():
    graph_labels = ['One','Two','Three','Four','Five']
    graph_values = calculate_totals()
    plot_bar(graph_labels,graph_values)
    graph_labels = ['Negative','Positive']
    graph_values = calculate_pos_neg()
    plot_pos_neg(graph_labels,graph_values)

main()
