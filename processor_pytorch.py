#################################################################################################
# The program is the main 'processing unit' of the project in a way                             #
# that it calls all the other programs and is the place where global variables                  #
# are defined. The program can be devided into two major parts: 1. the file parser section      #
# and 2. the neural network sectionself.                                                        #
# In the second section of the file, i.e. in the neural network section, an FFNN (Feed Forward  #
# Neural Network) class is instanciated, which is present in the 'network_pytorch.py' file      #
#                                                                                               #
# The author of this program is:                                                                #
# Swapnil Wagle                                                                                 #
# Max Planck Institute of Colloids and Interfaces, Potsdam, Germany                             #
# E-mail id: swapnil.wagle@mpikg.mpg.de                                                         #
#################################################################################################

#! /usr/env/python

import os
from os import path
from file_parser import File_Parser
import network_pytorch
from network_pytorch import Network_pytorch
import numpy
import torch
from torch.autograd import Variable
import torch.optim as optim

# The path for the directory, where all the data is located
path = '/Users/swapnil/Documents/FF_for_swapnil'    

# Initialization of the variables
files =[]
atomtypes = []
optypes = []
global i_vectors
global o_vectors

# Reading the atomtypes.txt files, which creates the the index list for the output vectors
f = open('./atomtypes.txt', "r")
i=0
for x in f.readlines():
    data = x.split()
    atomtypes.append(data[0])
    i = i+1
f.close()

# Reading the optypes.txt files, which creates the the index list for the input vectors
i=0
f = open('./optypes.txt', "r")
for x in f.readlines():
    data = x.split()
    optypes.append(data[0])
    i = i+1
f.close()

# The input and output arrays are redeclared as 2-dimensional numpy arrays, 
# where the second dimension (the coloumn index) is the length of the input/output index, 
# i.e. the optypes and atomtypes


i_vectors = numpy.empty([0, len(optypes)], dtype = numpy.double)
o_vectors = numpy.empty([0, len(atomtypes)*2], dtype = numpy.double)


# This is the first part of the processing unit, i.e. the files parser, 
# it is an abstract part of the pasring process, in which the files are listed. The path is then 
# sent to another program in the Class 'File_Parser', where it is transformed into numpy arrays 
# based on the indexing of the optyeps and atomtypes lists. The numpy arrays (i_vectors and o_vectors) 
# are utilized further by the neural network, which is introduced in the second part of this program.


i=0
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith(".txt"):
            files.append(os.path.join(r, file))
    for fff in sorted(files):
        if ((os.path.exists(fff)) and (os.path.getsize(fff) == 0)):
#            print ("Warning type 1: File exists but is empty " , fff)
            continue
        elif (not (os.path.exists(fff))):
#            print ("Warning type 2: txt file does not exists" , fff)
            continue
        else:
            txt_filepath = fff
            
        itp_filename = "lipid_" + os.path.splitext(fff)[0].split('_')[-2] + "_" + os.path.splitext(fff)[0].split('_')[-1] + ".itp"
        itp_filepath = os.path.join(os.path.dirname(fff), itp_filename)
        if (os.path.exists(itp_filepath) and (os.path.getsize(itp_filepath)) == 0):
#            print ("Warning type 1: File exists but is empty " , itp_filepath)
            continue
        elif (not (os.path.exists(itp_filepath))):
#            print ("Warning type 2: itp file does not exists" , itp_filepath)
            continue
        else:
            i_vector = numpy.array([len(optypes)], dtype=numpy.double)                        
            o_vector = numpy.array([len(atomtypes) * 2], dtype=numpy.double)            
            instance = File_Parser(txt_filepath, itp_filepath, atomtypes, optypes)
            (i_vector, o_vector) = zip(instance.file_parser(txt_filepath, itp_filepath, atomtypes, optypes))
            i_vectors = numpy.append(i_vectors, i_vector, axis = 0)
            o_vectors = numpy.append(o_vectors, o_vector, axis = 0)
            i= i+1
nof = i             # Total number of files, i.e. number of training-data files
i_vectors = torch.from_numpy(i_vectors)
o_vectors = torch.from_numpy(o_vectors)

# This is the second part of the processing unit, i.e. the neural network section.
# The i_vectors and o_vectors obtained from the file_parser section are combined together
# to generate the training data set for the neural network. The i_vectors and o_vectors are two-
# dimensional torch tensor, which contain the 'stacks' of the input and output tensor for training 
# the neural network.

# This section deals with instancing the neural network class (named Network_pytorch) and calling its method 'forward'
# for training the network. 
# eta is the learning rate, layers_sizes is a list containg the number of neurons in each of the layers with 
# first and last layer being the input and output vectors, respectively.
ffnn = network_pytorch.Network_pytorch(len(optypes), 200, len(atomtypes) * 2)    # Instanciating the neural network class
                                                                                 # The arguments are the number of neurons for each layer,
                                                                                 # i.e. input, hidden layers and output layer
loss = torch.nn.MSELoss()                                                        # MSE (Mean Squared Error) Loss function
optimizer = optim.SGD(ffnn.parameters(), lr=0.01)                                # Geadient descent algorithm for parameter optimization
                                                                                 # lr (Learning Rate) is 0.01
iterations = 0
running_loss = 0                                                                 # Parameter for running error/loss in the neural network
                                                                                 # prediction
for i_vector, o_vector in zip (i_vectors, o_vectors):
    x = i_vector.reshape(-1, len(optypes)).float()
    inputs = Variable(x)                                                         # Defining the input variable for the neural network class
    y = o_vector.reshape(-1, len(atomtypes) * 2).float()
    outputs = Variable(y)                                                        # Defining the output variable for the neural network class
    optimizer.zero_grad()                                                        # Making the parameter gradients Zero
    predictions = ffnn(inputs)  
    error = loss(predictions, outputs)
    print ('[%5d] loss: %.3f' % (iterations + 1, error))
    error.backward()                                                             # Backward propagation step
    optimizer.step()
    running_loss += error.data
    iterations += 1
#   if iterations % 20 == 19:    # print every 2000 mini-batches                #Uncomment the following section and remove 
#        print('[%5d] loss: %.3f' % (iterations + 1, running_loss / 20))        #this gap of lines if you want the code to output the cummulative 
#        running_loss = 0.0                                                     # error (or loss) in the training process every 20 iterations of the training. 


