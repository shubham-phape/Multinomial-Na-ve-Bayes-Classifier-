#Shubham Sunil Phape
#UTA ID: 1001773736
# In[17]:


#importing the libraries
import numpy as np



import pandas as pd
import statistics
from functools import reduce
import os, math , random, re
from collections import Counter


# In[18]:


#saving the main directory path and files
#source: https://www.tutorialspoint.com/python/os_getcwd.htm
main_directory = os.listdir(os.getcwd()+'/20_newsgroups')


# In[19]:


#contents of the folder list
print(main_directory)


# In[20]:


#Function to read the stopwords from file
#source: https://www.w3resource.com/python-exercises/file/python-io-exercise-7.php
def file_read(fname):
        content_array = []
        with open(fname) as f:
                #Content_list is the list that contains the read lines.     
                for line in f:
                        content_array.append(line.rstrip('\n'))
                return content_array


# In[21]:


#saving the stop words to string array the list os stop words is from
#source: https://gist.github.com/larsyencken/1440509
stop_words = file_read('stopwords.txt')


# In[22]:


list_ofall_docs ={}

folder_path = os.getcwd() + '/20_newsgroups/' 

#variables to save trainning and testing data
test_data = {}
train_data = {}


#iterating through all the folder in list
for folders in main_directory:
    #retrieving the list of files in each folder
    file_path = os.listdir(folder_path + folders)
    list_ofall_docs[folders] = file_path
    
    #shuffling the files before splitting 
    shuffled_list = list(range(0, len(file_path)))
    random.shuffle(shuffled_list)
    
    #splitting dataset into training and testing as half
    a=shuffled_list[int((len(file_path) / 2)):]
    c=shuffled_list[:int((len(file_path) / 2))]
    
    #mapping the splitted train and test data for each file in all folders with their respective paths
    #source: https://www.python-course.eu/python3_lambda.php
    train_data[folders] = list(map(lambda data1: file_path[data1], a))
    test_data[folders] = list(map(lambda data1: folder_path + folders + '/' + file_path[data1],c ))


# In[23]:


print("Training data size: "+str(len(train_data))+"\n")
print("Testing data size: "+str(len(test_data)))


# In[24]:


#function to prepocess the data in files line by line
def preprocessing(dataStream):
    with open(dataStream, 'rb') as data:
        
        linebyline = re.findall(rb"[\w']+", data.read().lower())
        stop_words_removal = np.array([word for word in linebyline if not word in stop_words])      
        #removing the numerical values
        non_numerical = np.array([word for word in stop_words_removal if not word.isdigit()])        
        #removing one words
        no_oneword = np.array([word for word in non_numerical if not len(word) == 1])       
        #removing dual words
        no_twoword = np.array([word for word in no_oneword if len(word) > 2])      
        #removing non string 
        non_strings = np.array([str for str in no_twoword if str])      
        #removing special characters
        fin_list = np.array([word for word in non_strings if word.isalnum()])
        
        return fin_list


# In[25]:


#making an dictionary key words as keys and values as the count of occurence of it
#source: https://github.com/gokriznastic/20-newsgroups_text-classification/blob/master/Multinomial%20Naive%20Bayes-%20BOW%20with%20TF.ipynb
#source 2: https://www.python-course.eu/python3_lambda.php

list_of_path=list(map(lambda x: len(list_ofall_docs[x]), main_directory))

#making the data into dictionary
word_count = dict(zip(main_directory,list_of_path ))
counting_train_words = {}
#using this as dict for the key=> value pair as "word"=> count
count = Counter()
sum_counting_train_words = {}

#iterating through all documents and counting the words and adding them to dictionary
for doc in main_directory:
    cnt = Counter()
    for fi in train_data[doc]:
        cnt = cnt + Counter(preprocessing(os.getcwd() + '/20_newsgroups/' + doc + '/' + str(fi)))
    count = count + cnt
    counting_train_words[doc] = dict(cnt)
    sum_counting_train_words[doc] = sum(counting_train_words[doc].values())
count = len(dict(count).keys())


# In[26]:


#displying the count for each document
print(sum_counting_train_words)
#function to calac prob by laplace smoothing
def p_laplace(hash,x, doclen_val):
    if x in hash:
        return math.log(hash[x] + 1.0) / (doclen_val+1)
    else:
        return math.log(1.0 / doclen_val)


# In[27]:


#function to calculate the total probability
def calculate_probability(document, hash, doclen_val):
    li = list(map(lambda x: p_laplace(hash, x, doclen_val), preprocessing(document)))
    return reduce(lambda x, y: x + y, li)

    


# In[28]:


#function to train the model and predit the probability of given test doc
def predict_class(document, hashing, traningSum, d_counter):
    #getting the class as key from the data
    keys = traningSum.keys()
    #getting the probability value
    probability = list(map(lambda x: calculate_probability(document, hashing[x], sum_counting_train_words[x] + d_counter), keys))

    probability = list(map(lambda x: x - max(probability), probability))
    denom= sum(list(map(lambda x: math.exp(x), probability)))
    probability = list(map(lambda x: math.exp(x) / denom, probability))
    
    #getting the index of the maximum found probability
    maximumIndex = [index for index in range(len(probability)) if probability[index] == max(probability)]
    #returning the folder name which is predicted
    return list(keys)[maximumIndex[0]]


# In[33]:


#classifying the test data 
num=0

#for every document in test data we classify
for i in range(0, len(test_data.keys())):
    file_size = len(test_data[list(test_data.keys())[i]])
    for j in range(0, file_size):
        
        output_list=predict_class(test_data[list(test_data.keys())[i]][j], counting_train_words, sum_counting_train_words, count)
        print("Predicted class - ",output_list)


# In[32]:


count1=0
length = len(test_data.keys())
for key in main_directory:
    if output_list==key:
        count1+=1
print("Error",count1/length)


# In[ ]:


#References: https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
#            https://github.com/gokriznastic/20-newsgroups_text-classification/blob/master/Multinomial%20Naive%20Bayes-%20BOW%20with%20TF.ipynb
#            https://gist.github.com/larsyencken/1440509

