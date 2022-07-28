import os
import re
import numpy as np


filename = 'test_results.txt'
dataset = "camrest"
file_loc = os.path.join(os.getcwd(), "runs","gpt2",dataset,filename)


def flatten(t):
    result = [item for sublist in t for item in sublist]
    return list(map(float,result))

bleu = []
f1 = []
k = []

for line in open(file_loc):
    	if line.startswith('Entity-F1'):
        	bleu.append(re.findall("\d+\.\d+",line))

    	if line.startswith('Entity-F1'):
        	f1.append(re.findall("\d+\.\d+",line))

    	if line.startswith('*'):
        	k.append(line)
    
print("Highest BLEU score is %s for %s" %( bleu[np.argmax(flatten(bleu))],k[np.argmax(flatten(bleu))] ) )
print("Highest F1 score is %s for %s" %( f1[np.argmax(flatten(bleu))],k[np.argmax(flatten(bleu))] ) ) 
