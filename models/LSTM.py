import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMSarcasm(nn.Module):
	def __init__(self, 
				output_size, 
				hidden_size, 
				vocab_size, 
				embedding_length,
				num_layers, 
				device):

		super(LSTMSarcasm, self).__init__()
		
		"""
		output_size : 1 = (0 for non sarcastic, 1 for sarcastic)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which are used to create the word_embedding look-up table 
		
		"""
		
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.num_layers = num_layers
		self.device = device

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers, batch_first=True, bidirectional=True)
		self.label = nn.Linear(2 * hidden_size, output_size)
		
	def forward(self, tweet, tweet_batch_len):
		""" 
		tweet: input tweet of shape = (batch_size, num_sequences)
		tweet_batch_len : lengths of tweets in a batch
		
		Returns
		-------
		Output of the linear layer containing the label, (this layer receives input as the final hidden state of the BiLSTM)
		final_output.shape = (batch_size, output_size)
		"""
		
		#print("Tweets: ")
		#print(tweet)

		embeddings = self.word_embeddings(tweet) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		#print("Embedding shape: " + str(embeddings.shape))

		packed_emb_output = pack_padded_sequence(embeddings, tweet_batch_len, batch_first=True, enforce_sorted=True)
		_, (hidden, _) = self.lstm(packed_emb_output)#, (hidden0, cell0)) # shape [batch_size, seq_length, hidden_size]
		
		hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim=1)
		#hidden = torch.cat((hidden[-1], hidden[-2]), dim=-1)
		
		final_output = self.label(hidden)
		final_output = torch.squeeze(final_output, 1)
		return final_output
