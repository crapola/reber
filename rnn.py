# Learn Reber grammar using torch.nn.RNN.

import copy

import matplotlib.pyplot as plt
import torch

import reber
from evaluation import generate_word_with_model


class Model(torch.nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.rnn=torch.nn.RNN(7,20,batch_first=True,nonlinearity='relu')
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
		# Unsqueeze to add batch dimension as RNN requires shape [B,S,C].
		x=x.unsqueeze(0)
		out,self.h=self.rnn(x,self.h)
		out=self.linear(out)
		out=torch.sigmoid(out).squeeze(0)
		return out
	def hidden_state_clear(self):
		self.h=None

def main():
	rnn=Model()
	print(rnn)
	optimizer=torch.optim.Adam(rnn.parameters(),lr=0.01)
	best=None
	costs=[]
	for epoch in range(2500):
		# Generate new sequence each epoch.
		x,y=torch.Tensor(reber.sample()).float()
		rnn.hidden_state_clear()
		# Pass full sequence at once.
		y_t_pred=rnn.forward(x)
		cost=torch.nn.BCELoss()(y_t_pred,y)
		# Backward.
		cost.backward()
		optimizer.step()
		optimizer.zero_grad()
		# History, status and checkpoint.
		costs.append(cost.detach().item())
		if not epoch%100: print("Epoch:",epoch,"Cost:",costs[-1])
		if costs[-1]<=min(costs):
			#best=copy.deepcopy(rnn)
			best=rnn.clone()
	# Use trained model to create new words.
	def evaluate(model:Model,word_count:int):
		model.eval()
		acc=0
		for _ in range(word_count):
			word=generate_word_with_model(rnn)
			print("Word:",word,"in grammar:",reber.in_grammar(word))
			acc+=100/word_count*int(reber.in_grammar(word))
		print(f"{acc:.1f}%")
	print("Latest:")
	evaluate(rnn,10)
	print("Best:")
	evaluate(best,10)
	# Plot losses.
	plt.plot(range(0,len(costs)), costs)
	plt.show()

if __name__=="__main__":
	main()
