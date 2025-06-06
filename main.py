import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import requests
import re
from datetime import datetime

# nltk.download('punkt_tab')
# nltk.download('wordnet')




class ChatBot(nn.Module):  # Define a neural network model that inherits from PyTorch's nn.Module
    def __init__(self, input_size, output_size):
       
        super(ChatBot, self).__init__()  # Initialize the base class (nn.Module)
        # First fully connected layer: Maps input features to 128 neurons
        self.input_layer = nn.Linear(input_size, 128)  
        # Second fully connected layer: Maps 128 neurons to 64 neurons
        self.hidden_layer = nn.Linear(128, 64)  
        # Third fully connected layer: Maps 64 neurons to the output layer (number of possible intents)
        self.output_layer = nn.Linear(64, output_size)  
        # Activation function (ReLU): Helps introduce non-linearity and prevents vanishing gradient issues
        self.relu = nn.ReLU()  
        # Dropout layer (regularization): Helps prevent overfitting by randomly disabling neurons during training
        self.dropout = nn.Dropout(0.5)  # 50% dropout probability

    def forward(self, x):
        """
        Forward pass method for the model. Defines how input data flows through the layers.      
        Parameters:
        - x (tensor): Input tensor containing processed feature data.
        Returns:
        - Output tensor representing the network's predictions for each intent category.
        """
        # Apply first fully connected layer, followed by ReLU activation
        x = self.relu(self.input_layer(x))  
        # Apply dropout to introduce randomness and reduce overfitting
        x = self.dropout(x)  
        # Apply second fully connected layer, followed by ReLU activation
        x = self.relu(self.hidden_layer(x))  
        # Apply another dropout for regularization
        x = self.dropout(x)  
        # Apply final fully connected layer to produce raw output (logits) for classification
        x = self.output_layer(x)  
        return x  #(these will be used for classification with softmax or another function)
    




class ChatBro:

    def __init__(self, intents_path, function_mappings = None):
        
        self.model = None  # Placeholder for the trained neural network model
        self.intents_path = intents_path  # Path to the JSON file containing intent definitions

        self.docs = []  # (tokenized_question, intent) pairs
        self.wordbook = []  # unique words from all training questions)
        self.intents = []  # List of intent labels/categories
        self.intents_answers = {}  # Dictionary mapping intents to possible answers

        self.function_mappings = function_mappings  # Optional dictionary linking intents to Python functions

        self.X = None  # Feature matrix (bag-of-words representation of questions)
        self.y = None  # Labels (as indices representing the corresponding intent)


    @staticmethod
    def text_preprocessing(text):
        # Create a lemmatizer instance from NLTK
        Lem = nltk.WordNetLemmatizer()
        # Tokenize the input text into words
        words = nltk.word_tokenize(text)

        # Convert words to lowercase and lemmatize them to reduce to their base form
        words = [Lem.lemmatize(word.lower()) for word in words]

      
        return words


    def word_pocket(self, words):
        # Generate a bag-of-words vector for the input words
        # For each word in the overall vocabulary (wordbook), set 1 if it's in the input, else 0
        return [1 if word in words else 0 for word in self.wordbook]

        
    def intent_analysis(self):
        
        if os.path.exists(self.intents_path):
            
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            # print(f"Loaded intents data with {len(intents_data['intents'])} intents.")

            # Loop through each intent in the loaded data
            for intent in intents_data['intents']:
                # If this intent category is not already in the list, add it
                if intent['category'] not in self.intents:
                    self.intents.append(intent['category'])  # Add intent category
                    self.intents_answers[intent['category']] = intent['answers']  # Map answers to this intent
                #print(f"Intent {i}: category='{intent['category']}' with {len(intent['questions'])} questions and {len(intent['answers'])} answers.")

                # Process each question associated with the intent
                for question in intent['questions']:
                    question_words = self.text_preprocessing(question)  # Tokenize and lemmatize the question
                    self.wordbook.extend(question_words)  # Add words to the overall vocabulary list
                    self.docs.append((question_words, intent['category']))  # Store the processed question with its intent
                    # if j < 2 and i < 2:  # limit output to first 2 questions of first 2 intents
                    # print(f"  Question {j} words: {question_words}")

            # Remove duplicates and sort the vocabulary
            self.wordbook = sorted(set(self.wordbook))
        #     unique_words_before = len(self.wordbook)
        #     self.wordbook = sorted(set(self.wordbook))
        #     unique_words_after = len(self.wordbook)

        #     print(f"Vocabulary size before deduplication: {unique_words_before}")
        #     print(f"Vocabulary size after deduplication and sorting: {unique_words_after}")
        # else:
        #     print(f"Intents file not found at path: {self.intents_path}")


                
    def prepare_data(self):
        pockets = []     
        indexes = []    

        # Loop through each document (question words, intent category)
        for document in self.docs:
            words = document[0]  # Extract the list of words from the document
            pocket = self.word_pocket(words)  # Convert words to bag-of-words vector

            # Second element of the tuple is the intent category,
            # so we get its index from the list of intents
            intent_index = self.intents.index(document[1])

            # Append the input vector and its label
            pockets.append(pocket)
            indexes.append(intent_index)
            # if i < 3:
            # print(f"Doc {i}: words={words}")
            # print(f"Bag of words: {pocket}")
            # print(f"Intent index: {intent_index}")

        # Convert the lists to NumPy arrays for training
        self.X = np.array(pockets)
        self.y = np.array(indexes)
        #print(f"Prepared data shapes -> X: {self.X.shape}, y: {self.y.shape}")
        
        

    def training_model(self, batch, _size, learningRate, epochs):
        # Pytorch tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        # Create a dataset and data loader for batching and shuffling the data
        dataSet = TensorDataset(X_tensor, y_tensor)
        Loader = DataLoader(dataSet, batch_size=batch, shuffle=True)
        self.model = ChatBot(self.X.shape[1], len(self.intents))
        # Define the loss function (CrossEntropy for classification)
        criteria = nn.CrossEntropyLoss()
        optimizor = optim.Adam(self.model.parameters(), lr=learningRate)
        # print(f"Starting training with model input size: {self.X.shape[1]}, output size: {len(self.intents)}")
        
        # device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        # self.model.to(device)
        
        # Loop over the number of training epochs
        for epoch in range(epochs):
            running_loss = 0.0  # Track the loss for each epoch
            # Iterate over batches from the data loader
            for batch_X, batch_y in Loader:
                # Set gradients to zero before the backward pass
                optimizor.zero_grad()
                # Forward pass: compute the output of the model
                output = self.model(batch_X)
                # Compute the loss between prediction and actual label
                loss = criteria(output, batch_y)
                # Backward pass: compute gradients
                loss.backward()
                # Optimizer step: update model weights
                optimizor.step()
                # Accumulate the batch loss
                running_loss += loss.item()
            # Print average loss for the epoch
            print(f"epoch {epoch+1}/{epochs}, loss: {running_loss/len(Loader):.3f}")
            
            
    # def training_model(self, batch, learningRate, epochs):
    #     X_tensor = torch.tensor(self.X, dtype=torch.float32)
    #     y_tensor = torch.tensor(self.y, dtype=torch.long)
    #     dataSet = TensorDataset(X_tensor, y_tensor)
    #     Loader = DataLoader(dataSet, batch_size=batch, shuffle=True)
    #     self.model = ChatBot(self.X.shape[1], len(self.intents))
    #     criteria = nn.CrossEntropyLoss()
    #     optimizor = optim.Adam(self.model.parameters(), lr=learningRate)

    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #         correct = 0
    #         total = 0
    #         self.model.train()

    #         for batch_X, batch_y in Loader:
    #             optimizor.zero_grad()
    #             output = self.model(batch_X)
    #             loss = criteria(output, batch_y)
    #             loss.backward()
    #             optimizor.step()

    #             running_loss += loss.item()
    #             # Get predictions
    #             _, predicted = torch.max(output.data, 1)
    #             total += batch_y.size(0)
    #             correct += (predicted == batch_y).sum().item()

    #         epoch_loss = running_loss / len(Loader)
    #         accuracy = correct / total * 100
    #         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            
    # def evaluate_accuracy(self):
    #     self.model.eval()
    #     X_tensor = torch.tensor(self.X, dtype=torch.float32)
    #     y_tensor = torch.tensor(self.y, dtype=torch.long)

    #     with torch.no_grad():
    #         outputs = self.model(X_tensor)
    #         _, predicted = torch.max(outputs, 1)
    #         correct = (predicted == y_tensor).sum().item()
    #         total = y_tensor.size(0)
    #         accuracy = correct / total * 100
    #     print(f"Accuracy on dataset: {accuracy:.2f}%")
    #     return accuracy



            
    def saving_model(self, model_path, dimensions_path):
        # Save the model's learned parameters (weights) to the specified file path
        torch.save(self.model.state_dict(), model_path)

        # Open the dimensions file in write mode to save model input/output sizes
        with open(dimensions_path, 'w') as f:
            # Save input and output sizes to JSON so they can be reused when loading the model
            json.dump({'input__size': self.X.shape[1], 'output_size': len(self.intents)}, f)

                    
                    
            
    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        
        #import and export the model, the input_size is self.X.shape[1] and the output_size is the length of the intents    
        self.model = ChatBot(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        

 
    def message_processing(self, message):
        # Preprocess the incoming message: tokenize and lemmatize
        words = self.text_preprocessing(message)
        # Convert the processed words into a bag-of-words vector
        pocket = self.word_pocket(words)
        # Convert the vector into a tensor to be used with the model
        pocket_tensor = torch.tensor([pocket], dtype=torch.float32)

        # Set the model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        with torch.no_grad():
            # Get the prediction from the model
            prediction = self.model(pocket_tensor)

        # Find the index of the highest scoring intent
        predicted_index = torch.argmax(prediction, dim=1).item()
        # Map the index to the actual intent name
        predicted_intent = self.intents[predicted_index]
        # print(f"[Predicted Intent Index]: {predicted_index}")
        # print(f"[Predicted Intent]: '{predicted_intent}'")

        # If there are function mappings defined and the intent is linked to a function
        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                # Call the corresponding function with the input message
                return self.function_mappings[predicted_intent](message)

       
        if self.intents_answers[predicted_intent]:
            return random.choice(self.intents_answers[predicted_intent])
            #print(f"[Random Response]: '{response}'")
        else:
            return None


        
        
        
# ## Test Getting functions  
# def my_subject(message=None):
#     subject = ['Modern DB', 'GOOGL', 'DSA' , 'ML', 'AI', 'NLP', 'DL', 'CV', 'RL', 'NLP', 'ML', 'AI']
#     print(random.sample(subject,10))
# def get_assignment_topics(message=None):
#     topics = [
#         "Build a sentiment analysis tool",
#         "Implement a decision tree from scratch",
#         "Create a chatbot using intents",
#         "Analyze customer churn dataset",
#         "Build a recommendation engine"
#     ]
#     print(random.choice(topics))
# def get_exam_dates(message=None):
#     exams = {
#         "AI": "June 12, 10:00 AM",
#         "ML": "June 15, 2:00 PM",
#         "DSA": "June 18, 11:00 AM",
#         "NLP": "June 20, 9:00 AM"
#     }
#     for subject, date in exams.items():
#         print(f"{subject} exam is on {date}")





def get_weather(message=None):
    api_key = "7f532017c006af554cc67fd1815a42c5"
    default_city = "London"
    
    city = default_city  # Default fallback

    if message:
        match = re.search(r'weather\s+(?:in|for)\s+([a-zA-Z\s]+)', message, re.IGNORECASE)
        if match:
            city = match.group(1).strip()

    print(f"Final City Used in API Call: {city}")  # Debug print

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        print(f"Extracted city: {city}")  # Add this before calling API
        print(f"Final Response: Weather in {city.title()}: {desc}, {temp}°C")

        return f"Weather in {city.title()}: {desc}, {temp}°C"
    else:
        return f"Couldn't fetch weather for '{city}'. API response: {response.status_code}"





    
def get_crypto(message=None):
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana,cardano,dogecoin,ripple,polkadot&vs_currencies=usd"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            prices = {
                "Bitcoin": data["bitcoin"]["usd"],
                "Ethereum": data["ethereum"]["usd"],
                "Solana": data["solana"]["usd"],
                "Cardano": data["cardano"]["usd"],
                "Dogecoin": data["dogecoin"]["usd"],
                "Ripple": data["ripple"]["usd"],
                "Polkadot": data["polkadot"]["usd"]
            }

            result = "Current Crypto Prices:\n"
            for name, price in prices.items():
                result += f"- {name}: ${price}\n"
            return result.strip()
        else:
            return "Failed to fetch cryptocurrency prices."
    except Exception as e:
        return f"Error: {e}"



def get_news(message=None):
    api_key = "471bfd28b8cc466994be4c70e5883cf9"  
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])[:50]  # Get top 50 articles
            if not articles:
                return "No news articles found."

            headlines = [f"- {article['title']}" for article in articles]
            return "Top News Headlines:\n" + "\n".join(headlines)
        else:
            return "Failed to fetch news."
    except Exception as e:
        return f"Error: {e}"
    
    
def get_time_date(message=None):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")      # e.g. "2025-05-16"
    time_str = now.strftime("%H:%M:%S")      # e.g. "14:35:20"
    return f"Current date is {date_str} and time is {time_str}."




if __name__ == "__main__":
        # Add new functions to function_mappings
    function_mappings = {
        # 'subjects': my_subject,
        'news': get_news,
        'weather': get_weather,
        'crypto': get_crypto,
        'time_date': get_time_date
    }
    chatBro = ChatBro('intents.json', function_mappings=function_mappings)
    chatBro.intent_analysis()
    chatBro.prepare_data()
    chatBro.training_model(batch=8, _size=8, learningRate=0.001, epochs=100)

    chatBro.saving_model('chatbot_model.pth', 'dimensions.json')
    
    # Interact with the model:
    while True:
        message = input("Please enter your message: ") 
        if message  == '/quit':
            break
        print(chatBro.message_processing(message))   