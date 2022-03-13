import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

class LSTMSarcasm(nn.Module):
	def __init__(self, 
				output_size, 
				hidden_size, 
				vocab_size, 
				embedding_length,
				num_layers, 
				device,
				bidirectional=False):

		super(LSTMSarcasm, self).__init__()
		
		"""
		output_size : 1 = (0 for non sarcastic, 1 for sarcastic)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		num_layers : Number of layers the network should have
		device: The device on which the model will be sent to
		bidirectional: boolean variable to choose a simple lstm or a bidirectional lstm, default is False
		
		"""
		
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.num_layers = num_layers
		self.device = device
		self.bidirectional = bidirectional

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
		self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
		self.label = nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size)
		
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

		if self.lstm.bidirectional:		
			hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]),dim=1)
		else:
			hidden = hidden[-1, :, :]

		final_output = self.label(hidden)
		final_output = torch.squeeze(final_output, 1)
		return final_output


# LSTM with Attention option
class LSTMSarcasmAttn(nn.Module):
	def __init__(self, 
				output_size, 
				hidden_size, 
				vocab_size, 
				embedding_length,
				num_layers, 
				device
				):

		super(LSTMSarcasmAttn, self).__init__()
		
		"""
		LSTM with Attention
		output_size : 1 = (0 for non sarcastic, 1 for sarcastic)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		num_layers : Number of layers the network should have
		device: The device on which the model will be sent to
		bidirectional: boolean variable to choose a simple lstm or a bidirectional lstm, default is False
		
		"""
		
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.num_layers = num_layers
		self.device = device

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
		self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers, batch_first=True)
		self.label = nn.Linear(hidden_size, output_size)
		
	def attention_mec(self, lstm_output, final_state):

		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state
	

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

		output, (hidden, _) = self.lstm(embeddings) #, (hidden0, cell0)) # shape [batch_size, seq_length, hidden_size]
	
		hidden = hidden[-1, :, :]
		attention_output = self.attention_mec(output, hidden)

		final_output = self.label(attention_output)
		final_output = torch.squeeze(final_output, 1)
		return final_output

