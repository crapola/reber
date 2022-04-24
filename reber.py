import random

import numpy as np
import numpy.typing as npt

CHARS='BTSXPVE'

GRAPH=[[(1,5),('T','P')] , [(1,2),('S','X')],
	[(3,5),('S','X')], [(6,),('E',)],
	[(3,2),('V','P')], [(4,5),('V','T')] ]

ReberVectors=npt.ArrayLike # shape=(n,7) dtype=float

def generate_string()->str:
	"""
	Create a Reber word.
	"""
	idx=0
	out='B'
	while idx!=6:
		idx,symbol=random.choice(list(zip(*GRAPH[idx])))
		out+=symbol
	return out

def in_grammar(word:str)->bool:
	"""
	Check if word is a valid Reber word.
	"""
	if word[0]!='B':
		return False
	node=0
	for c in word[1:]:
		transitions=GRAPH[node]
		try:
			node=transitions[0][transitions[1].index(c)]
		except ValueError:
			return False
	return True

def sample()->tuple:
	"""
	Generate one (input,target) encoded sample where input and target
	are offset by one letter.
	Example: for 'BTXSE', X='BTXS' Y='TXSE'.
	"""
	word=vectors_from_string(generate_string())
	return np.array( [word[:-1],word[1:]] )

def string_from_vectors(sequence:ReberVectors)->str:
	"""
	Converts a sequence of one-hot unit vectors to a Reber string.
	"""
	return ''.join( np.array(list(CHARS))[np.argmax(sequence,1)] )

def vectors_from_string(s:str)->ReberVectors:
	"""
	Convert a Reber string to a sequence of unit vectors.
	"""
	return np.eye(len(CHARS))[np.char.index(CHARS,np.array(list(s)))]

def main():
	assert in_grammar("BTXSE")
	assert in_grammar("BTSXXTVVE")
	assert not in_grammar("BTSXXTBVE")
	vec=vectors_from_string("BXXX")
	word=string_from_vectors(vec)
	assert word=="BXXX"
	vec=vectors_from_string("BTSXPVE")
	assert np.all(vec==np.eye(7))
	word=string_from_vectors(vec)
	assert word=="BTSXPVE"
	for i in range(0,10):
		word=generate_string()
		print("generate_string:",word)
		assert in_grammar(word)

if __name__=="__main__":
	main()
