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
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
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
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		"""
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		
		#with torch.no_grad():
		#print("Tweets: ")
		#print(tweet)

		embeddings = self.word_embeddings(tweet) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		#print("Embedding shape: " + str(embeddings.shape))

		#hidden0 = torch.zeros(self.num_layers, tweet.shape[0], self.hidden_size) #.to(device)
		#cell0 = torch.zeros(self.num_layers, tweet.shape[0], self.hidden_size) #.to(device)


		packed_emb_output = pack_padded_sequence(embeddings, tweet_batch_len, batch_first=True, enforce_sorted=True)#.to(self.device)
		_, (hidden, _) = self.lstm(packed_emb_output)#, (hidden0, cell0)) # shape [batch_size, seq_length, hidden_size]
		#packed_output, _ = pad_packed_sequence(output, batch_first=True)
		
		#print("Output shape: " + str(output.shape))

		#out_forward = output[range(len(output)), tweet_batch_len - 1, :self.hidden_size]
		#out_reversed = output[:, 0, self.hidden_size:]
		#out_combined = torch.cat((out_forward, out_reversed), 1)

		#output = output.reshape(output.shape[0], -1)
		hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim=1)
		#hidden = torch.cat((hidden[-1], hidden[-2]), dim=-1)
		
		final_output = self.label(hidden)
		final_output = torch.squeeze(final_output, 1)
		#print("Final output shape: " + str(final_output.shape))
		return final_output
