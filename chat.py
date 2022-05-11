"""Here we use our trained models to actually create the chatbot. Testing out different values for the hyperparameters,
we found that, for epochs much larger than around 1000 there was significant overfitting. The effect of this was that
most input sentences would be interpreted as a greeting, unless they were very close to one of the other tags. When we
added the L2 regularisation, this had a noticeable effect during training, since the training error was higher. This was
as expected, since there should be less overfitting, and we would expect a lot of overfitting since the training dataset
is so small. The regularised model seemed to perform smoother, though there were many sentences which it didn't
understand. This is probably more of an issue with the size of the training dataset, and with a larger dataset this
model would probably perform quite well.
"""


import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"  # File saved from training
data = torch.load(FILE)

# Get model hyperparameters that were found from training
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # Load trained model
model.eval()

# Now run the model
bot_name = "Sam"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
