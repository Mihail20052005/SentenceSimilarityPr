from scipy.spatial import distance
from torch import load
from src.model import InferSent
from random import randint
import numpy as np
import torch
import time
import torch.nn as nn


params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0}
sentences = ["I want to go to champ"]
test = "I love to go to champ"
print('Test Sentence:', test)
model2 = torch.load('model.pkl')

test_vec = model2.encode([test])[0]


for sent in sentences:
    similarity_score = 1-distance.cosine(test_vec, model2.encode([sent])[0])
    print(f'\nFor {sent}\nSimilarity Score = {similarity_score} ')