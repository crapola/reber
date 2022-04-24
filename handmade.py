# Learn Reber grammar using a handmade RNN.

import matplotlib.pyplot as plt
import torch

import reber
from evaluation import generate_word_with_model


class HandmadeRNNCell:
	def __init__(self):
		n_inp=7
		n_out=n_inp
		n_hid=20
		self.W_ih=torch.zeros(n_inp,n_hid).uniform_(-1,1) # Input->Hidden.
		self.W_hh=torch.zeros(n_hid,n_hid).uniform_(-1,1) # Hidden->Hidden_t+1.
		self.W_ho=torch.zeros(n_hid,n_out).uniform_(-1,1) # Hidden->Output.
		self.b_h=torch.zeros(n_hid).uniform_(-1,1) # Bias hidden.
		self.b_o=torch.zeros(n_out).uniform_(-1,1) # Bias output.
		self.W_ih/=torch.linalg.svd(self.W_ih)[1][0]
		self.W_hh/=torch.linalg.svd(self.W_hh)[1][0]
		self.W_ho/=torch.linalg.svd(self.W_ho)[1][0]
		self.W_ih.requires_grad_()
		self.W_hh.requires_grad_()
		self.W_ho.requires_grad_()
		self.b_h.requires_grad_()
		self.b_o.requires_grad_()
		self.hidden_state_clear()
	def forward(self,x_t):
		h_t=torch.tanh(torch.matmul(x_t,self.W_ih) + torch.matmul(self.h_tm1,self.W_hh) + self.b_h)
		y_t=torch.matmul(h_t,self.W_ho) + self.b_o
		y_t=torch.softmax(y_t,-1)
		self.h_tm1=h_t
		return y_t
	def hidden_state_clear(self):
		self.h_tm1=torch.zeros(self.b_h.size())
	def step(self,lr):
		self.W_ih -= self.W_ih.grad * lr
		self.W_hh -= self.W_hh.grad * lr
		self.b_h  -= self.b_h.grad  * lr
		self.W_ho -= self.W_ho.grad * lr
		self.b_o  -= self.b_o.grad  * lr
		self.W_ih.grad.zero_()
		self.W_hh.grad.zero_()
		self.b_h.grad.zero_()
		self.W_ho.grad.zero_()
		self.b_o.grad.zero_()

def bce(y_pred,y):
	return -torch.mean(y*torch.log(y_pred)+(1.-y)*torch.log(1.-y_pred))

def sample():
	"""
	Generate one (input,target) sample.
	"""
	word=reber.vectors_from_string(reber.generate_string())
	word=torch.Tensor(word).float()
	return word[:-1],word[1:]

def main():
	rnn=HandmadeRNNCell()
	lr=0.2
	costs=[]
	for epoch in range(1000):
		# Generate new sequence each epoch.
		x,y=sample()
		# Reset initial hidden state of time step t-1.
		rnn.hidden_state_clear()
		cost=0
		for c in range(len(x)):
			x_t=x[c]
			y_t_true=y[c]
			y_t_pred=rnn.forward(x_t)
			# Accumulate costs over the whole sequence.
			cost+=bce(y_t_pred,y_t_true)
		# Backward when sequence completed.
		cost.backward()
		costs.append(cost.detach().item())
		with torch.no_grad():
			rnn.step(lr)
		if not epoch%100: print("Epoch:",epoch,"Cost:",costs[-1])
	# Use trained model to create new words.
	for _ in range(10):
		word=generate_word_with_model(rnn)
		print("Word:",word,"in grammar:",reber.in_grammar(word))
	# Plot losses.
	plt.plot(range(0,len(costs)), costs)
	plt.show()

if __name__=="__main__":
	main()
