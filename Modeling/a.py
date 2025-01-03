import pandas as pd
import math
import torch
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv('./dataset/ielts-writing-essays.csv')

# Preprocess dataset
label_encoder = LabelEncoder()
essays = df['Essay'].tolist()

essays_tokenized = []
all_tokens = []

for essay in essays:
    tokens = word_tokenize(essay)
    all_tokens.extend(tokens)
    essays_tokenized.append(tokens)

# Create vocabulary
vocab = list(set(all_tokens))
label_encoder.fit(vocab)

print(f"Number of tokenized essays: {len(essays_tokenized)}")
print(f"Total tokens: {len(all_tokens)}")
print(f"Vocabulary size: {len(vocab)}")

def get_data(dataset, batch_size):
    """Convert tokenized dataset to tensor batches."""
    data = [torch.LongTensor(label_encoder.transform(tokens)) for tokens in dataset]
    data = pad_sequence(data, batch_first=True, padding_value=0)
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, -1)  # Shape: (batch_size, total_seq_len)
    return data

# Split dataset
train_len = round(len(essays_tokenized) * 0.7)
test_val_len = round(len(essays_tokenized) * 0.15)

train_data = essays_tokenized[:train_len]
test_data = essays_tokenized[train_len:train_len + test_val_len]
val_data = essays_tokenized[train_len + test_val_len:]

batch_size = 16
train_data = get_data(train_data, batch_size)
test_data = get_data(test_data, batch_size)
val_data = get_data(val_data, batch_size)

print(f"Train data shape: {train_data.shape}")

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, 
            dropout=dropout_rate, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def forward(self, src, hidden):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        predictions = self.fc(output)
        return predictions, hidden

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-1.0 / math.sqrt(self.fc.in_features), 1.0 / math.sqrt(self.fc.in_features))
        self.fc.bias.data.zero_()

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        cell = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        return tuple(h.detach() for h in hidden)

# Hyperparameters
VOCAB_SIZE = len(vocab)
EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT_RATE = 0.1
LR = 1e-3
EPOCHS = 50
SEQ_LEN = 50
CLIP = 0.25

model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

def get_batch(data, seq_len, idx):
    """Extract source and target sequences."""
    src = data[:, idx:idx + seq_len]
    tgt = data[:, idx + 1:idx + seq_len + 1]
    return src, tgt

def train_epoch(model, data, optimizer, criterion, batch_size, seq_len, clip):
    model.train()
    epoch_loss = 0
    num_batches = data.shape[1] // seq_len
    hidden = model.init_hidden(batch_size)

    for idx in tqdm(range(0, num_batches * seq_len, seq_len), desc='Training', leave=False):
        src, tgt = get_batch(data, seq_len, idx)
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)
        predictions, hidden = model(src, hidden)

        predictions = predictions.reshape(-1, VOCAB_SIZE)
        tgt = tgt.reshape(-1)
        loss = criterion(predictions, tgt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / num_batches

def evaluate_epoch(model, data, criterion, batch_size, seq_len):
    model.eval()
    epoch_loss = 0
    num_batches = data.shape[1] // seq_len
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for idx in range(0, num_batches * seq_len, seq_len):
            src, tgt = get_batch(data, seq_len, idx)
            src, tgt = src.to(device), tgt.to(device)

            hidden = model.detach_hidden(hidden)
            predictions, hidden = model(src, hidden)

            predictions = predictions.reshape(-1, VOCAB_SIZE)
            tgt = tgt.reshape(-1)
            loss = criterion(predictions, tgt)

            epoch_loss += loss.item()

    return epoch_loss / num_batches

# Learning rate scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_data, optimizer, criterion, batch_size, SEQ_LEN, CLIP)
    valid_loss = evaluate_epoch(model, val_data, criterion, batch_size, SEQ_LEN)

    lr_scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-val-lstm_lm.pt')

    print(f"Epoch {epoch+1}:")
    print(f"\tTrain Loss: {train_loss:.4f} | Train Perplexity: {math.exp(train_loss):.3f}")
    print(f"\tValid Loss: {valid_loss:.4f} | Valid Perplexity: {math.exp(valid_loss):.3f}")

# Evaluate on test set
model.load_state_dict(torch.load('best-val-lstm_lm.pt'))
test_loss = evaluate_epoch(model, test_data, criterion, batch_size, SEQ_LEN)
print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {math.exp(test_loss):.3f}")
