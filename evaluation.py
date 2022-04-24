import torch

import reber


def topk_choice(distribution,k=2):
	"""
	Select top k values of distribution and then pick a random one of those.
	"""
	best=distribution.topk(k)
	scaled=best.values/best.values.sum(dim=-1).unsqueeze(-1)
	idx=scaled.multinomial(num_samples=1,replacement=True)
	return best.indices[idx]

def generate_word_with_model(model)->str:
	"""
	Generate a Reber word using a trained model.
	"""
	model.hidden_state_clear()
	word=["B"]
	for _ in range(100):
		with torch.no_grad():
			ct=model.forward(torch.Tensor(reber.vectors_from_string(word[-1])))[0]
			# The model returns a probability density.
			# Pick one of the two best at random.
			i=topk_choice(ct)
			oh=torch.nn.functional.one_hot(i,7)
			c=reber.string_from_vectors(oh)
			word.append(c)
			if c=="E":
				break
	return "".join(word)
