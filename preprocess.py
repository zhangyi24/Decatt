# -*- coding: utf-8 -*-

import pickle

import numpy as np

from utils import load_config

#TODO: use absolute path
def build_vocab(word_emb_raw_path, word_emb_path, vocab_path):
	with open(word_emb_raw_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		vocab_size, embed_size = map(int, lines[0].split())
		vocab_size += 1
		vocab = {'UNK': 0}
		word_emb = np.zeros([vocab_size, embed_size])
		word_id = 1
		for line in lines[1:]:
			items = line.split()
			word = ''.join(items[:-embed_size])
			emb = list(map(float, items[-embed_size:]))
			word_emb[word_id, :] = emb
			vocab[word] = word_id
			word_id += 1
	# normalize word_emb
	# word_emb = (word_emb - np.mean(word_emb, axis=0)) / np.std(word_emb, axis=0)
	with open(word_emb_path, 'wb') as f:
		pickle.dump(word_emb, f)
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)
		
		
def l2_normalize(x, axis, epsilon=1e-12):
	square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
	norm = np.sqrt(np.maximum(square_sum, epsilon))
	return np.divide(x, norm)
	
	
if __name__ == '__main__':
	cfg_path = 'config/config.yaml'
	cfg = load_config(cfg_path, 'default')
	
	word_emb_raw_path = cfg['word_emb_raw_path']
	word_emb_path = cfg['word_emb_path']
	vocab_path = cfg['vocab_path']
	build_vocab(word_emb_raw_path, word_emb_path, vocab_path)
	
	
