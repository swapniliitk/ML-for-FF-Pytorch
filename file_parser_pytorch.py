#################################################################################################
# The class File_Parser contains the methods used for parsing text from the txt and itp         #
# files and generate  input and output vectors (numpy arrays) from the read text.               #
#                                                                                               #
# The author of this code is:                                                                   #
# Swapnil Wagle                                                                                 #
# Max Planck Institute of Colloids and Interfaces, Potsdam, Germany                             #
# E-mail id: swapnil.wagle@mpikg.mpg.de                                                         #
#################################################################################################

import numpy
import torch
from __future__ import print_function
numpy.set_printoptions(suppress=True)

# Initialization of the class
class File_Parser(object):    

    def __init__(self, i_path, o_path, atomtypes, optypes):
        self.i_path = i_path
        self.o_path = o_path
        self.atomtypes = atomtypes
        self.optypes = optypes
    
    
    def file_parser(self, i_path, o_path, atomtypes, optypes):
        i_list = (self.parse_txt(i_path))
        o_list = (self.parse_itp(o_path))
        return(self.create_vectors(i_list, o_list, atomtypes, optypes))

# Parser program for the itp files. It takes the itp file's path as input and extract the useful data out.
# A numpy array (o_vector) is created by calling the method "create_vectors", which is sent back to the processor
# program and is subsequently used as the output vector for training the neural network.
    def parse_itp(self, path):
        f = open (path, "r")
        flag = 0
        content = []
        for x in f.readlines():
            data=x.split()
            if (len(data) < 3):
                continue
            elif ((data[0]=="[") and (data[1]=="atomtypes") and (data[2]=="]")):
                flag = 1
                sign = "atomtype"
                continue
            elif ((data[0]=="[") and (data[1]=="atoms") and (data[2]=="]")):
                flag = 1
                sign = "atom"
                continue
            elif ((data[0]=="[") and (data[2]=="]")):
                flag = 0
                
            if ((flag==1) and data[0] != ";" and len(data)):
                x = x + sign
                content.append(x)
        f.close()
        return(content)
        
# Parser program for the txt files. It takes the txt file's path as input and extract the useful data out.
# A numpy array (i_vector) is created by calling the method "create_vectors", which is sent back to the processor 
# program and is subsequently used as the input vector for training the neural network.
    def parse_txt(self, path):
        f = open (path, "r")
        i=0
        content=[]
        for x in f.readlines():
            data=x.split()
            if ((data[0]=="fitness") or (data[0]=="final")):
                continue
            else:
                content.append(x)
        f.close()        
        return(content)

# This method of the class deals with transforming the parsed data into input and output numpy arrays for
# training the neural network. It takes the extracted data from the itp and txt file parser methods, and the 
# optypes and atomtypes indices for arranging these data into two one-dimensional numpy arrays (i_vector and o_vector).
    def create_vectors(self, i_list, o_list, atomtypes, optypes):
        i_vector = torch(len(self.optypes))
        o_vector = torch((len(self.atomtypes) * 2))
        for line in i_list:
            y = line.split()
            if y[0] in self.optypes:
                i_vector[self.optypes.index(y[0])] = round(float(y[2]), 10)
            elif y[0] not in self.optypes:
                print (y[0] + " is not present in the optype list")
            else: 
                i_vector[self.optypes.index(y[0])] = numpy.NaN
                print(y[0] + "does not exist in the list of optypes")

        for line in o_list:
            y = line.split()
            if (y[-1] == "atomtype"):
                if y[0] in self.atomtypes:
                    o_vector[self.atomtypes.index(y[0])] = round(float(y[5]),10)
                elif y[0] not in self.atomtypes:
                    print (y[0] + " is not present in the atomtype list")
                else:
                    print(y[0] + "does not exist in the list of atomtypes: ATOMTYPE secion")
            elif (y[-1] == "atom"):
                if y[1] in self.atomtypes:
                    o_vector[self.atomtypes.index(y[1]) + len(self.atomtypes)] = round(float(y[6]),5)
                else:
                    print(y[1] + "does not exist in the list of atomtypes: ATOM secion")
            else:
#                print("Check the entry " + line)
                o_vector[self.atomtypes.index(y[0])] = numpy.NaN
                o_vector[self.atomtypes.index(y[0]) + len(self.atomtypes)] = numpy.NaN
        return (i_vector, o_vector)
        
    
