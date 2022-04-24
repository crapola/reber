# Learn Reber grammar using torch.nn.RNNCell.

import copy

import matplotlib.pyplot as plt
import torch

import reber
from evaluation import generate_word_with_model


class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.rnn=torch.nn.RNNCell(7,20)
		self.linear=torch.nn.Linear(20,7)
		self.hidden_state_clear()
	def __str__(self)->str:
		d={k:self.rnn.state_dict()[k].size() for k in self.rnn.state_dict().keys()}
		s=f"{super().__str__()}\nRNN structure:{d}"
		return s
	def clone(self):
		m=Model()
		m.rnn=copy.deepcopy(self.rnn)
		m.linear=copy.deepcopy(self.linear)
		return m
	def forward(self,x):
		self.h=self.rnn(x,self.h)
		out=self.linear(self.h)
		out=torch.sigmoid(out)
		return out
	def hidden_state_clear(self):
		self.h=None

def try_model(model,cnt):
	acc=0
	for _ in range(cnt):
		word=generate_word_with_model(model)
		print("Word:",word,"in grammar:",reber.in_grammar(word))
		acc+=100/cnt if reber.in_grammar(word) else 0
	print(f"{acc:.1f}%")

def main():
	rnn=Model()
	print(rnn)
	optimizer=torch.optim.SGD(rnn.parameters(),lr=0.2)
	best=None
	costs=[]
	for epoch in range(1000):
		# Generate new sequence each epoch.
		x,y=torch.Tensor(reber.sample()).float()
		rnn.hidden_state_clear()
		cost=0
		for c in range(len(x)):
			x_t=x[c].unsqueeze(0)
			y_t_true=y[c].unsqueeze(0)
			y_t_pred=rnn.forward(x_t)
			# Accumulate costs over the whole sequence.
			cost+=torch.nn.BCELoss()(y_t_pred,y_t_true)
		cost.backward()
		optimizer.step()
		optimizer.zero_grad()
		costs.append(cost.detach().item())
		if not epoch%100: print("Epoch:",epoch,"Cost:",costs[-1])
		if costs[-1]<=min(costs):
			#best=copy.deepcopy(rnn)
			best=rnn.clone()
	# Use trained model to create new words.
	print("Latest:")
	try_model(rnn,10)
	print("Best:")
	try_model(best,10)
	# Plot losses.
	plt.plot(range(0,len(costs)), costs)
	plt.show()

if __name__=="__main__":
	main()
