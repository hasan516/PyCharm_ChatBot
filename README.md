# PyCharm_ChatBot
A simple chatbot created using a PyTorch neural network.

The intents.json file is the training data that is used to train the model. The different tags 
correspond to the different kinds of classifications for user input to the chatbot. For each tag,
there are a series of sample inputs which will be used to train the neural network, and then some
responses which will be chosen at random once a tag is decided. 

The nltk_utils.py file creates the basic functions used to turn the input data into vectorised 
numerical data that can be used to train the model. We use the natural language toolkit for this.

The model.py file contains the network architecture. Here we use a simple 2 layer model. Since the
input data is quite small in size, there is no point in using a more complicated model. Indeed, as
discussed in the chat.py file, we find that even this model is prone to overfitting. 

train1.py and train2.py are where we train the model. The only difference between the two is the 
choices of hyperparameters. Also, in train2, we have added an L2 regularisation. 

chat.py is where we execute the chatbot, once we have run one of the training files.
