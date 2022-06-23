# sourcery skip: do-not-use-bare-except
import nltk
from nltk.stem.lancaster import LancasterStemmer #is used to stem our words
stemmer=LancasterStemmer()

import numpy
import tflearn #TFlearn is a high-level deep learning library built on the top of TensorFlow. It is modular and transparent. It offers excellent facilities and speeds up experimentation. It is easy to use and understand.
import tensorflow #to create machine learning models for desktop, mobile, web, and cloud.
import random
import json
import pickle #Try to load some data (words , lables , traing , output data  )
#we need data for our model
with open("intents.json") as file:
    data=json.load(file)
try:
    with open("data.pickle","rb") as f :
        words,lables , training , output = pickle.load(f)   #we gonna save all of these four variables into our pickle file  
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(pattern)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w !="?"]
    words = sorted((list(set(words))))    
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [0 for _ in range (len(lables))]
            
    for x , doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0) # means this word is nt here so 0
            #To generate the output:
        output_row=out_empty[:]
        output_row[labels.index(docs_y[x])]=1

        training.append(bag)
        output.append(output_row)
        #chagne in our model:
    training = numpy.array(training) 
    output = numpy.array(output)
    with open("data.pickle","wb") as f :
        pickle.dump((words,lables , training , output), f) #write all of these virable into a pickle 
    

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])]) #defie the input shapre were expecting for model
net = tflearn.fully_connected(net, 8) # we have 2 hidden layers with 8 neurons fully connected we add fully connected layer to our neural network which starts at this input data 
# were gonna have 8 neurons for that hidden layer
net = tflearn.fully_connected(net, 8) #another hidden layer

#an output layer that has neurons representing each of our classes
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #we need 2 more layer so 
#activation="softmax": allows us to ge probabiliteas each output 
net = tflearn.regression(net)


#To train our model :
model= tflearn.DNN(net) #DNN type of neuron network(takes the net & use that)

try:
    model.load("model.tflearn")
except:
#number of epoch is the amount of times that its gonna see the same data 
    model.fit(training, output , n_epoch=1000, batch_size=8, show_metric=True) #we startpassing it all of our trainig data
    model.save("model.tflearn") #TO make some predictions 


#To turn a sentence input from the user into a bag of words so :

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
     
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
     
    for se in s_words:
        for i , w in enumerate(words):
            if w == se:
                 bag[i].append(1)
    return numpy.array(bag)

def chat():
     print("start talking with th bog !")
     while True:
        inp = input("You:")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results) #give us the index of greatest value in our list
        tag =lables[results_index] #labels stores all different lables so it will give us the label that it thinks our message is 
        # print(tag)
        
        
# For the  respose:we will find JSON file that specific tag and then pick a random response
        for tg in data["intets"]:
            if tg['tg']==tag:
                responses = tg['responses']
                
        print(random.choice(responses))




chat()
  















