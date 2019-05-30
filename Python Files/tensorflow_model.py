#TensorFlow: Natural Language Processing
#@author Chance Simmons
#@version December 2018
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
from tensorflow.contrib import rnn

TRAIN_DATA = "Data/train_token.csv"
TEST_DATA = "Data/test_token.csv"
TRAIN_LABEL = "Data/train_label.csv"
TEST_LABEL = "Data/test_label.csv"
VAL_DATA = "Data/val_token.csv"
VAL_LABEL = "Data/val_label.csv"
EPOCHS = 20
BATCH_SIZE = 1000
HIDDEN_SIZE = 32
VOCAB_SIZE = 10000
LEARNING_RATE = .01

tf.reset_default_graph()

def plot(train_acc,train_loss,val_acc,val_loss):
    """
    Plots the training and validation metrics
    
    Parameters:
        train_acc: accuracies from training epochs
        train_loss: loss values for each training epoch
        val_acc: accuracies from validation epochs
        val_loss: loss values for each validation epoch
    """
    numbered_epochs = range(1,len(train_acc)+1)
    
    #Set Up Plot
    figure = plt.figure()
    plt.title("Accuracy Values")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(range(0,len(train_acc) + 1,5))
    
    #Plot Accuracy
    plt.plot(numbered_epochs,train_acc,label = "Training Accuracy")
    plt.plot(numbered_epochs,val_acc,label = "Validation Accuracy")
    plt.legend()
    plt.show()
    
    figure.savefig("Graphs/tensorflow_accuracy.pdf", bbox_inches='tight')
    
    plt.clf()
    
    #Set up second plot
    figure = plt.figure()
    plt.title("Loss Values")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(range(0,len(train_acc) + 1,5))
    
    #Plot loss values
    plt.plot(numbered_epochs,train_loss,label = "Training Loss")
    plt.plot(numbered_epochs,val_loss,label = "Validation Loss")
    plt.legend()
    plt.show()
    
    figure.savefig("Graphs/tensorflow_loss.pdf", bbox_inches='tight')
    
def get_data(data_file,label_file):
    """
    Gets data based off of the specified data and label file
    
    Parameters:
        data_file: CSV file that has data lines
        label_file: CSV file that has label lines
        
    Return:
        data: list of all the data lines
        label: list of all the label lines
    """
    with open(data_file,"r") as data, open(label_file,"r") as label:
        data_reader = csv.reader(data)
        label_reader = csv.reader(label)
        data = []
        label = []
        for data_line in data_reader:
            line_int = [int(i) for i in data_line]
            data.append(line_int)
                
        for label_line in label_reader:
            line_int = [int(i) for i in label_line]
            label.append(line_int)
    
    data = np.array(data)
    label = np.array(label)
    return data,label

def shuffle(data,label):
    """
    Shuffles the data and label in the same pattern
    
    Parameters:
        data: list of data lines
        label: list of label lines
    
    Return:
        data: shuffled data
        label: shuffled labels
    """
    permutation = np.arange(len(data))
    np.random.shuffle(permutation)
    data = data[permutation]
    label = label[permutation]
    return data,label

def validate(val_data,val_label,data_input,label_output,drop,accuracy,loss,sess):
    """
    Validates on the network during training
    
    Parameters:
        val_data: the data lines for validating
        val_label: the labels for validating
        data_input: placeholder for data to be input for the model
        label_output: placeholder for the labels to be placed for the model
        drop: placeholder for dropout for the model
        accuracy: caluclates the accuracy of the model
        loss: calculates the accuracy of the model
        sess: these tensorflow session that is being run
    """
    val_loss = []
    val_acc = []
    start = 0
    end = BATCH_SIZE
    for j in range(len(val_data) // BATCH_SIZE):            
        data = val_data[start:end]
        label = val_label[start:end]
            
        feed_dict = {data_input: data,
                     label_output: label,
                     drop: 1}
            
        loss_value, accuracy_value = sess.run([loss,accuracy], feed_dict = feed_dict)
            
        val_loss.append(loss_value)
        val_acc.append(accuracy_value * 100)
        
        start += BATCH_SIZE
        end += BATCH_SIZE
            
    print("Validation Loss: {:.4f} Accuracy: {:.2f}%".format(np.mean(val_loss), 
                                                                 np.mean(val_acc)))
    return val_acc,val_loss
        
def test(test_data,test_label,data_input,label_output,drop,accuracy,sess):
    """
    Performs test of the the trained model
    
    Parameters:
        test_data: the data lines for testing
        test_label: the labels for testing
        data_input: placeholder for data to be input for the model
        label_output: placeholder for the labels to be placed for the model
        drop: placeholder for dropout for the model
        accuracy: caluclates the accuracy of the model
        sess: these tensorflow session that is being run
    """
    start_time = time.time()
    test_acc = []
    start = 0
    end = BATCH_SIZE
    for i in range(len(test_data) // BATCH_SIZE):      
        data = test_data[start:end]
        label = test_label[start:end]
            
        feed_dict = {data_input: data,
                     label_output:label,
                     drop: 1}
          
        accuracy_value = sess.run([accuracy], feed_dict=feed_dict)
        test_acc.append(accuracy_value * 100)
        
        start += BATCH_SIZE
        end += BATCH_SIZE
        
    print("\nTest:\nTime Spent: {:.3f}s Accuracy: {:.2f}%".format(
              time.time() - start_time,float(np.mean(test_acc) * 100)))

def train(train_data,train_label,data_input,label_output,drop,accuracy,loss,optimizer,sess):
    """
    Performs training if the model for a single epoch
    
    Parameters:
        train_data: the data lines for testing
        train_label: the labels for testing
        data_input: placeholder for data to be input for the model
        label_output: placeholder for the labels to be placed for the model
        drop: placeholder for dropout for the model
        accuracy: caluclates the accuracy of the model
        loss: calculates loss for the training step
        optimizer: used to perform optimization of the model
        sess: these tensorflow session that is being run
    """
    train_loss = []
    train_acc = []
    start = 0
    end = BATCH_SIZE
    for i in range(len(train_data) // BATCH_SIZE):            
            data = train_data[start:end]
            label = train_label[start:end]
            
            feed_dict = {data_input: data,
                         label_output:label,
                         drop: .5}
          
            optim, loss_value,accuracy_value = sess.run([optimizer, loss, accuracy],
                                                        feed_dict=feed_dict)
            
            train_loss.append(loss_value)
            train_acc.append(accuracy_value * 100)
            
            start += BATCH_SIZE
            end += BATCH_SIZE
    return train_acc,train_loss
def model():
    """
    Creates the RNN and returns the different parts of the model
    
    Return:
        data_input: used as a data placeholder
        label_output: used as a label placeholder
        drop: used as a dropout placeholder
        optimizer: used to perform optimization
        loss: used to calculate loss
        accuracy: used to calculate accuracy
    """
    #Create Placeholders
    data_input = tf.placeholder(tf.int32, [None,None])
    label_output = tf.placeholder(tf.float32,[None,1])
    drop = tf.placeholder(tf.float32)
    
    #Embedding
    embedding = tf.Variable(tf.random_uniform([VOCAB_SIZE,HIDDEN_SIZE],-1,1))
    embed = tf.nn.embedding_lookup(embedding,data_input)
    
    #LSTM
    lstm = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
    init_state = lstm.zero_state(BATCH_SIZE,dtype=tf.float32)
    
    lstm_drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob = drop)
    lstm_out,state = tf.nn.dynamic_rnn(lstm,embed,initial_state = init_state,dtype=tf.float32)
    
    #Output
    weight = tf.Variable(tf.random_normal([HIDDEN_SIZE,1]))
    bias = tf.Variable(tf.random_normal([1]))
    
    lstm_out = tf.transpose(lstm_out,[1,0,2])
    prediction = tf.matmul(tf.cast(lstm_out[-1],tf.float32),weight) + bias
    
    #Accuracy
    correct = tf.equal(tf.round(tf.nn.sigmoid(prediction)), label_output)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
    #Loss and Optimizer
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction,labels 
                                                                  = label_output))
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)
    
    return data_input,label_output,drop,optimizer,loss,accuracy

def plot_time(time_list):
    numbered_epochs = range(1,len(time_list)+1)
    
    #Set Up Plot
    figure = plt.figure()
    plt.title("TensorFlow Training Time")
    plt.xlabel("Epochs")
    plt.ylabel("Time")
    plt.xticks(range(0,len(time_list)+1,5))
    
    #Plot Accuracy
    plt.plot(numbered_epochs,time_list)
    plt.show()
    
    figure.savefig("Graphs/tensorflow_time.pdf", bbox_inches='tight')
    
def main():
    train_data,train_label = get_data(TRAIN_DATA,TRAIN_LABEL)
    val_data,val_label = get_data(VAL_DATA,VAL_LABEL)
    test_data,test_label = get_data(TEST_DATA,TEST_LABEL)
    
    data_input,label_output,drop,optimizer, loss, accuracy = model()
    
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    time_list = []
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(1,EPOCHS + 1):
        start_time = time.time()
        
        train_data,train_label = shuffle(train_data,train_label)
        print("Epoch Count: " + str(epoch) + "/" + str(EPOCHS))
        
        #Training
        train_acc,train_loss = train(train_data,train_label,data_input,label_output,
                                     drop,accuracy,loss,optimizer,sess)
        
        train_acc_list.append(np.mean(train_acc))
        train_loss_list.append(np.mean(train_loss))
        
        total_time = time.time() - start_time
        print("Time Spent: {:.3f}s Loss: {:.4f} Accuracy: {:.2f}%".format(
              total_time,np.mean(train_loss),np.mean(train_acc)))
        
        time_list.append(total_time)
        
        #Validate
        val_acc,val_loss = validate(val_data,val_label,data_input,label_output,drop,accuracy,
                                    loss,sess)
        val_acc_list.append(np.mean(val_acc))
        val_loss_list.append(np.mean(val_loss))
            
    plot(train_acc_list,train_loss_list,val_acc_list,val_loss_list)
    plot_time(time_list)
    
    time_sum = 0
    for i in range(len(time_list)):
        time_sum += time_list[i]
        
    print("Average Time: " + str(time_sum / len(time_list)))
    
    #Test  
    test(test_data,test_label,data_input,label_output,drop,accuracy,sess)
    sess.close()
    
main()