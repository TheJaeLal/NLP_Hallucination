import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'

import torch
import spacy

import torchtext.legacy as tt
nlp = spacy.load('en_core_web_sm')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("We are working with", device)

KNOWLEDGE = tt.data.Field(tokenize='spacy', tokenizer_language="en_core_web_sm", include_lengths = True)
RESPONSE = tt.data.Field(tokenize='spacy', tokenizer_language="en_core_web_sm", include_lengths = True)
HISTORY = tt.data.Field(tokenize='spacy', tokenizer_language="en_core_web_sm", include_lengths = True)
LABEL = tt.data.LabelField(dtype=torch.float)

path = "task1_bilstm_model.pt"

fields = {"knowledge": ("k", KNOWLEDGE), "response": ("r", RESPONSE), "hallucination": ("l", LABEL), "history": ("h", HISTORY)}
train_data, valid_data, test_data = tt.data.TabularDataset.splits(path="./",
                                     train="faithdial_dataset_train.json",
                                     validation="faithdial_dataset_validation.json",
                                     test="faithdial_dataset_test.json",
                                     format="json",
                                     fields=fields)

MAX_VOCAB_SIZE = 25_000
KNOWLEDGE.build_vocab(train_data,
                      max_size=MAX_VOCAB_SIZE,
                      vectors = "fasttext.simple.300d",
                      unk_init = torch.Tensor.normal_)
RESPONSE.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors = "fasttext.simple.300d",
                     unk_init = torch.Tensor.normal_)
HISTORY.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors = "fasttext.simple.300d",
                     unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(f"Unique tokens in KNOWLEDGE vocabulary: {len(KNOWLEDGE.vocab)}")
print(f"Unique tokens in RESPONSE vocabulary: {len(RESPONSE.vocab)}")
print(f"Unique tokens in HISTORY vocabulary: {len(HISTORY.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

from torch import nn

class LSTM(nn.Module):
    def __init__(self, response_vocab_size, knowledge_vocab_size, history_vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, response_pad_idx, knowledge_pad_idx, history_pad_idx):

        super().__init__()

        # Initialize Embedding Layer
        self.response_embedding = nn.Embedding(num_embeddings=response_vocab_size,
                                               embedding_dim=embedding_dim,
                                               padding_idx=response_pad_idx)

        self.knowledge_embedding = nn.Embedding(num_embeddings=knowledge_vocab_size,
                                                embedding_dim=embedding_dim,
                                                padding_idx=knowledge_pad_idx)
        
        self.history_embedding = nn.Embedding(num_embeddings=history_vocab_size,
                                              embedding_dim=embedding_dim,
                                              padding_idx=history_pad_idx)

        # Initialize LSTM layer
        self.response_lstm = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_dim,
                                     num_layers=n_layers,
                                     bidirectional=bidirectional)

        self.knowledge_lstm = nn.LSTM(input_size=embedding_dim,
                                      hidden_size=hidden_dim,
                                      num_layers=n_layers,
                                      bidirectional=bidirectional)
        
        self.history_lstm = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=n_layers,
                                    bidirectional=bidirectional)

        # Initialize a fully connected layer with Linear transformation
        self.fc = nn.Linear(in_features=3*2*hidden_dim,
                            out_features=output_dim)

        # Initialize Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, response, response_lengths, knowledge, knowledge_lengths, history, history_lengths):
        # Apply embedding layer that matches each word to its vector and apply dropout. Dim [sent_len, batch_size, emb_dim]
        x_r = self.response_embedding(response)
        x_r = self.dropout(x_r)

        x_k = self.knowledge_embedding(knowledge)
        x_k = self.dropout(x_k)

        x_h = self.history_embedding(history)
        x_h = self.dropout(x_h)

        # Run the LSTM along the sentences of length sent_len.
        output_r, (hidden_r, cell_r) = self.response_lstm(x_r)
        output_k, (hidden_k, cell_k) = self.knowledge_lstm(x_k)
        output_h, (hidden_h, cell_h) = self.history_lstm(x_h)

        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden_r = torch.cat((hidden_r[-2,:,:], hidden_r[-1,:,:]), -1)
        hidden_k = torch.cat((hidden_k[-2,:,:], hidden_k[-1,:,:]), -1)
        hidden_h = torch.cat((hidden_h[-2,:,:], hidden_h[-1,:,:]), -1)
        hidden = torch.cat((hidden_r, hidden_k, hidden_h), -1)
        hidden = self.dropout(hidden)

        return self.fc(hidden)

RESPONSE_INPUT_DIM = len(RESPONSE.vocab)
KNOWLEDGE_INPUT_DIM = len(KNOWLEDGE.vocab)
HISTORY_INPUT_DIM = len(HISTORY.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
RESPONSE_PAD_IDX = RESPONSE.vocab.stoi[RESPONSE.pad_token]
KNOWLEDGE_PAD_IDX = KNOWLEDGE.vocab.stoi[KNOWLEDGE.pad_token]
HISTORY_PAD_IDX = HISTORY.vocab.stoi[HISTORY.pad_token]

model = LSTM(RESPONSE_INPUT_DIM,
             KNOWLEDGE_INPUT_DIM,
             HISTORY_INPUT_DIM,
             EMBEDDING_DIM,
             HIDDEN_DIM,
             OUTPUT_DIM,
             N_LAYERS,
             BIDIRECTIONAL,
             DROPOUT,
             RESPONSE_PAD_IDX,
             KNOWLEDGE_PAD_IDX,
             HISTORY_PAD_IDX)

model.load_state_dict(torch.load(path, map_location=device))


def predict_hallucination(model, knowledge, history, response):
    model.eval()
    
    model.to(device)

    tokenized_r = [tok.text for tok in nlp.tokenizer(response)]
    indexed_r = [RESPONSE.vocab.stoi[t] for t in tokenized_r]
    length_r = [len(indexed_r)]
    tensor_r = torch.LongTensor(indexed_r).to(device)
    tensor_r = tensor_r.unsqueeze(1)
    length_tensor_r = torch.LongTensor(length_r).to(device)

    tokenized_k = [tok.text for tok in nlp.tokenizer(knowledge)]
    indexed_k = [KNOWLEDGE.vocab.stoi[t] for t in tokenized_k]
    length_k = [len(indexed_k)]
    tensor_k = torch.LongTensor(indexed_k).to(device)
    tensor_k = tensor_k.unsqueeze(1)
    length_tensor_k = torch.LongTensor(length_k).to(device)

    tokenized_h = [tok.text for tok in nlp.tokenizer(history)]
    indexed_h = [HISTORY.vocab.stoi[t] for t in tokenized_h]
    length_h = [len(indexed_h)]
    tensor_h = torch.LongTensor(indexed_h).to(device)
    tensor_h = tensor_h.unsqueeze(1)
    length_tensor_h = torch.LongTensor(length_h).to(device)
    
    prediction = torch.sigmoid(model(tensor_r, length_tensor_r, tensor_k, length_tensor_k, tensor_h, length_tensor_h))

    return prediction.item()
