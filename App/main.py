import pandas as pd
import math
import torch
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from string import punctuation
from tqdm import tqdm
import enchant
from enchant.tokenize import get_tokenizer
from enchant.tokenize import basic_tokenize
import pickle
from flask import Flask, url_for,redirect, render_template, Response,request
from flask import Flask

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                    dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        
        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)          
        output = self.dropout(output) 
        prediction = self.fc(output)
        return prediction, hidden
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                    self.hidden_dim).uniform_(-init_range_other, init_range_other) 
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, 
                    self.hidden_dim).uniform_(-init_range_other, init_range_other) 

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell
    

app = Flask('__name__')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load("entire_model.pth")




with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

@app.route('/')
def home():
    prompt = request.args.get('prompt', '') 
    append_word = request.args.get('append', '')  
    

    if append_word:
        prompt = f"{prompt} {append_word}".strip()
        return redirect(url_for('home', prompt=prompt))

    token_indices = []
    if prompt != '':
        tokens = word_tokenize(prompt)
    else:
        return render_template('index.html', prompt=prompt)
    
    for token in tokens:
        try:
            token = token.lower()
            print(token)
            indice = label_encoder.transform([token])[0]
            token_indices.append(indice)
        except:
            print("hehe")
            
    if len(token_indices) <=0:
        return render_template('index.html', prompt=prompt)

    # token_indices = label_encoder.transform(tokens)
    
    input_tensor = torch.LongTensor(token_indices).unsqueeze(0).to(device)
    
    hidden = model.init_hidden(1,device)
    
    generated_text = tokens.copy()
    with torch.no_grad():
        predictions, hidden = model(input_tensor, hidden)
    
    # predicted_idx = predictions[0, -1].argmax().item()
    top_values, top_indices = torch.topk(predictions[0, -1], 3)
    top_predictions = top_indices.tolist()
    top_words = [label_encoder.inverse_transform([idx])[0] for idx in top_predictions]
    print(top_words)
    # print(predictions.sort())
    # if predicted_idx != 0: 
        
        
    #     try:
    #         predicted_word = label_encoder.inverse_transform([predicted_idx])[0]
    #     except (ValueError, IndexError):
    #         predicted_word = ''

    #     generated_text.append(predicted_word)
    
    

    return render_template('index.html', prompt=prompt, one=top_words[0], two=top_words[1], three=top_words[2])

@app.route('/recommend')
def recommend():
    prompt = request.args.get('prompt', '') 


if __name__ == "__main__":
    app.run(port=5050,debug=True)
