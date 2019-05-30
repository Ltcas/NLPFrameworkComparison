#PyTorch: Natural Language Processing
#@author Chance Simmons
#@version December 2018
import numpy as np
import torch
import csv
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
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


class NLPNet(torch.nn.Module):
    """
    Produces a RNN neural network and can be used for forward passes through the network
    """
    def __init__(self,vocab_size,batch_size,hidden_size,output_size):
        """
        Initialized the neural netwrk model

        Parameters:
            vocab_size: the size of the vocabulary used for embedding
            batch_size: the number of elements that will be passed through at once
            hidden_size: the number of hidden layer nodes to create
            output_size: the size of the output
        """
        super(NLPNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        #Embedding
        self.embed = torch.nn.Embedding(vocab_size,hidden_size)

        #LSTM
        self.lstm = torch.nn.LSTM(hidden_size,hidden_size)

        #Fully connected output layer
        self.fc = torch.nn.Linear(hidden_size,output_size)

    def init_hidden(self):
        """
        Intializes the hidden layers for the LSTM

        Return:
            hidden layers that will be used in the LSTM
        """
        hidden_one = Variable(torch.zeros(1,self.batch_size,self.hidden_size))
        hidden_two = Variable(torch.zeros(1,self.batch_size,self.hidden_size))
        if torch.cuda.is_available():
            hidden_one = hidden_one.cuda()
            hidden_two = hidden_two.cuda()

        return hidden_one, hidden_two

    def forward(self, input_data):
        """
        Performs a forward pass through the model

        Parameters:
            input_data: batched list of input lines

        Return:
            the prediction the model made
        """
        #Transpose to correct shape
        input_data = input_data.t()

        #Embedding
        input_embed  = self.embed(input_data)

        #LSTM
        out,self.hidden = self.lstm(input_embed,self.hidden)

        #Output of Last LSTM Cell
        out = self.fc(out[-1])

        return torch.sigmoid(out)    


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
    data_list = []
    label_list = []
    with open(data_file,"r") as data, open(label_file,"r") as label:
        data_reader = csv.reader(data)
        label_reader = csv.reader(label)
        for data_line in data_reader:
            line_int = [int(i) for i in data_line]
            data_list.append(line_int)

        for label_line in label_reader:
            line_int = [int(i) for i in label_line]
            label_list.append(line_int)

    return data_list,label_list

def test(model,test_data,test_label):
    """
    Test the model given a test dataset

    Parameters:
        model: the model that has been trained
        test_data: the data that is used for testing
        test_label: labels to calulate accuracy with
    """
    start_time = time.time()
    model = model.eval()
    test_acc = []
    start = 0
    end = BATCH_SIZE
    with torch.no_grad():
        for i in range(len(test_data) // BATCH_SIZE):
            model.hidden = model.init_hidden()

            data = test_data[start:end]
            label = test_label[start:end]

            data = torch.from_numpy(data)
            label = torch.from_numpy(label)

            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            #Pass through model
            test_out = model(data)

            #Get accuracy
            test_out = (test_out > 0.5).float()
            test_correct = (test_out == label.float()).float().sum()
            test_acc.append(float(100 * test_correct/test_out.shape[0]))

            start += BATCH_SIZE
            end += BATCH_SIZE

    print("\nTest:\nTime Spent: {:.3f}s Accuracy: {:.2f}%".format(
                time.time() - start_time,np.mean(test_acc)))

def validate(model,val_data,val_label,loss_function):
    """
    Performs validation of the model given a validation dataset

    Parameters:
        model: the model that is being trained
        val_data: data used for validation
        val_label: labels used for getting accuracy
        loss_function: used to calculate loss for the validation

    Return:
        val_acc: list of validation accuracies for each batch
        val_loss: list of validation loss for each batch
    """
    model = model.eval()
    with torch.no_grad():
        loss_values = []
        val_acc = []
        val_data,val_label = shuffle(val_data,val_label)
        start = 0
        end = BATCH_SIZE
        for i in range(len(val_data) // BATCH_SIZE):
            model.hidden = model.init_hidden() 

            data = val_data[start:end]
            label = val_label[start:end]

            data = torch.from_numpy(data)
            label = torch.from_numpy(label)

            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            #Pass through model
            val_out = model(data)

            #Get loss value
            loss = loss_function(val_out,label.float())
            loss_values.append(float(loss.data.mean()))

            #Get accuracy
            val_out = (val_out > 0.5).float()
            val_correct = (val_out == label.float()).float().sum()
            val_acc.append(float(100 * val_correct/val_out.shape[0]))

            start += BATCH_SIZE
            end += BATCH_SIZE
    return val_acc,loss_values

def train(model,loss_function,optimizer,train_data,train_label):
    """
    Performs the training for one epoch

    Parameters:
        model: the model that is being trained
        loss_function: function for calculating loss
        optimizer: used to perform optimization on the model
        train_data: the data that is being used to train
        train_label: labels for correct output

    Return:
        train_acc: list of train accuracies for each batch
        train_loss: list of train loss for each batch
    """
    train_acc = []
    train_loss = []
    train_data,train_label = shuffle(train_data,train_label)
    start = 0
    end = BATCH_SIZE
    for i in range(len(train_data) // BATCH_SIZE):
        model.hidden = model.init_hidden()

        data = train_data[start:end]
        label = train_label[start:end]

        data = Variable(torch.from_numpy(data))
        label = Variable(torch.from_numpy(label))
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        #Pass through Model
        train_out = model.forward(data)

        #Get loss value
        loss = loss_function(train_out,label.float())
        train_loss.append(float(loss.data.mean()))

        #Get accuracy
        train_out = (train_out > 0.5).float()
        correct_train = (train_out == label.float()).float().sum()
        train_acc.append(float(100 * correct_train/train_out.shape[0]))

        #Backpropagation    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        start += BATCH_SIZE
        end += BATCH_SIZE
    return train_acc,train_loss

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
    
    figure.savefig("Graphs/pytorch_accuracy.pdf", bbox_inches='tight')

    #Clear figure
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
    
    figure.savefig("Graphs/pytorch_loss.pdf", bbox_inches='tight')

def plot_time(time_list):
    numbered_epochs = range(1,len(time_list)+1)

    #Set Up Plot
    figure = plt.figure()
    plt.title("PyTorch Training Time")
    plt.xlabel("Epochs")
    plt.ylabel("Time")
    plt.xticks(range(0,len(time_list)+1,5))

    #Plot Accuracy
    plt.plot(numbered_epochs,time_list)
    plt.show()

    figure.savefig("Graphs/pytorch_time.pdf", bbox_inches='tight')
    
def main():
    train_data,train_label = get_data(TRAIN_DATA,TRAIN_LABEL)
    test_data,test_label = get_data(TEST_DATA,TEST_LABEL)
    val_data,val_label = get_data(VAL_DATA,VAL_LABEL)

    train_data,train_label = shuffle(np.array(train_data),np.array(train_label))
    test_data,test_label = shuffle(np.array(test_data),np.array(test_label))
    val_data,val_label = shuffle(np.array(val_data),np.array(val_label))

    model = NLPNet(vocab_size = VOCAB_SIZE,batch_size = BATCH_SIZE,hidden_size = HIDDEN_SIZE,
                    output_size = 1)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    time_list = []

    for epoch_count in range(1,EPOCHS + 1):
        start_time = time.time()
        model = model.train()
        print("Epoch Count: " + str(epoch_count) + "/" + str(EPOCHS))

        #Train
        train_acc,train_loss= train(model,loss_function,optimizer,train_data,train_label)

        train_acc_list.append(np.mean(train_acc))
        train_loss_list.append(np.mean(train_loss))
        
        total_time = time.time() - start_time
        
        print("Time Spent: {:.3f}s Loss: {:.4f} Accuracy: {:.2f}%".format(
                time.time() - start_time,np.mean(train_loss),np.mean(train_acc)))

        time_list.append(total_time)
        
        #Validate
        val_acc,val_loss = validate(model,val_data,val_label,loss_function)   

        val_acc_list.append(np.mean(val_acc))
        val_loss_list.append(np.mean(val_loss))

        print("Validation Loss: {:.4f}% Accuracy: {:.2f}".format(float(np.mean(val_loss)),
                                                                    float(np.mean(val_acc))))

    plot(train_acc_list, train_loss_list, val_acc_list,val_loss_list)
    plot_time(time_list)
    
    time_sum = 0
    for i in range(len(time_list)):
        time_sum += time_list[i]
    
    print("Average Time: " + str(time_sum / len(time_list)))
    
    #Test
    test(model,test_data,test_label)

main()