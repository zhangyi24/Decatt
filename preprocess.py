# -*- coding: utf-8 -*-

import pickle
import datetime

import numpy as np

from utils import load_config, load_vocab, load_embed


#TODO: use absolute path
def build_vocab(word_emb_raw_path, word_emb_path, vocab_path, dataset_name):
	print(datetime.datetime.now(), 'start')
	if dataset_name == 'atec':
		return build_vocab_zh(word_emb_raw_path, word_emb_path, vocab_path)
	elif dataset_name == 'quora':
		return build_vocab_en(word_emb_raw_path, word_emb_path, vocab_path)
	print(datetime.datetime.now(), 'end')
	

def build_vocab_zh(word_emb_raw_path, word_emb_path, vocab_path):
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
	print('vocab_size:', len(vocab), 'embed_size:', word_emb.shape)


def build_vocab_en(word_emb_raw_path, word_emb_path, vocab_path):
	embed_size = 300
	word_emb = [np.zeros([1, embed_size])]
	vocab = {'UNK': 0}
	with open(word_emb_raw_path, 'r', encoding='utf-8') as f:
		word_id = 1
		for line in f:
			line = line.split()
			word = ''.join(line[:-embed_size])
			emb = np.expand_dims(list(map(float, line[-embed_size:])), axis=0)
			word_emb.append(emb)
			vocab[word] = word_id
			word_id += 1
			if word_id % 50000 == 0:
				print(datetime.datetime.now(), word_id)
	word_emb = np.concatenate(word_emb, axis=0)
	with open(word_emb_path, 'wb') as f:
		pickle.dump(word_emb, f)
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)
	print('vocab_size:', len(vocab), 'embed_size:', word_emb.shape)


def l2_normalize(x, axis, epsilon=1e-12):
	square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
	norm = np.sqrt(np.maximum(square_sum, epsilon))
	return np.divide(x, norm)
	
	
if __name__ == '__main__':
	cfg_path = 'config/config.yaml'
	dataset_name = load_config(cfg_path, section='default')['dataset']
	cfg = load_config(cfg_path, section=dataset_name)

	
	word_emb_raw_path = cfg['word_emb_raw_path']
	word_emb_path = cfg['word_emb_path']
	vocab_path = cfg['vocab_path']
	# build_vocab(word_emb_raw_path, word_emb_path, vocab_path, dataset_name)
	
	print(load_vocab(vocab_path))
	print(load_embed(word_emb_path))
	
