import csv
import yaml
import random
import pickle
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
import jieba


def load_config(yaml_file, section):
	with open(yaml_file, 'r', encoding='utf-8') as f:
		cfg = yaml.load(f)
	return cfg[section]


def load_hparams(yaml_file, section):
	hparams = load_config(yaml_file, section)
	hparams = HParams(**hparams)
	return hparams

	
def feed_data(model, embed_sen0, embed_sen1, mask_sen0, mask_sen1, onehot_labels, training):
	feed_dict = {
		model.embed_sen0: embed_sen0,
		model.embed_sen1: embed_sen1,
		model.mask_sen0: mask_sen0,
		model.mask_sen1: mask_sen1,
		model.onehot_labels: onehot_labels,
		model.training: training
	}
	return feed_dict
	

def load_vocab(vocab_path):
	with open(vocab_path, 'rb') as f:
		return pickle.load(f)


def load_embed(embed_path):
	with open(embed_path, 'rb') as f:
		return pickle.load(f)


# TODO: lazy
class Dataset(object):
	def __init__(self, data_path, dataset_name):
		self._data_path = data_path
		self._dataset_name = dataset_name
		self._data, self._len = self.build_dataset()
		self._idx = 0
		self.init()
	
	def build_dataset(self):
		data = []
		data_len = 0
		if self._dataset_name == 'atec':
			with open(self._data_path, 'r', encoding='utf-8-sig') as f:
				csvreader = csv.reader(f, delimiter='\t')  # [id_pair, sentence_0, sentence_1, label]
				for line in csvreader:
					sen0, sen1, label = line[1], line[2], int(line[3])
					data.append({'sen0': sen0, 'sen1': sen1, 'label': label})
					data_len += 1
		elif self._dataset_name == 'quora':
			with open(self._data_path, 'r', encoding='utf-8-sig') as f:
				csvreader = csv.reader(f)
				for line in csvreader:
					if line[0].isdigit():
						sen0, sen1, label = line[3].rstrip('?'), line[4].rstrip('?'), int(line[5])
						data.append({'sen0': sen0, 'sen1': sen1, 'label': label})
						data_len += 1
		return data, data_len
	
	def label_ratio(self):
		cnt_labels = {}
		for datum in self._data:
			label = datum['label']
			if label in cnt_labels:
				cnt_labels[label] += 1
			else:
				cnt_labels[label] = 1
		return cnt_labels
		
	def init(self):
		random.shuffle(self._data)
		self._idx = 0
		
	def next_batch(self, batch_size=-1):
		if batch_size >= 0:
			id_end = min(self._idx + batch_size, self._len)
			batch_data = self._data[self._idx: id_end]
			self._idx = id_end
			if self._idx == self._len:
				self.init()
		else:
			batch_data = self._data
		return batch_data
	
	def __len__(self):
		return self._len

	def sen2num(self, sen, max_len, vocab, embed):
		if self._dataset_name == 'atec':
			sen = jieba.lcut(sen)
		if self._dataset_name == 'quora':
			sen = sen.lower().split()
		sen_len = min(len(sen), max_len)
		sen = sen[0: sen_len]
		words_id = [vocab[word] if word in vocab else vocab['UNK'] for word in sen]
		words_id.extend([0] * (max_len - sen_len))
		embed_sen = embed[words_id, :]
		mask = np.pad(np.ones([sen_len], dtype=np.float64), [0, max_len-sen_len], 'constant')
		return embed_sen, mask

	# TODO: save data as numbers to accelerate the training
	def data2num(self, batch_data, max_len, num_classes, vocab, embed):
		batch_size = len(batch_data)
		embed_size = embed.shape[1]
		embed_sen0 = np.zeros(shape=[batch_size, max_len, embed_size], dtype=np.float64)
		embed_sen1 = np.zeros(shape=[batch_size, max_len, embed_size], dtype=np.float64)
		mask_sen0 = np.zeros(shape=[batch_size, max_len], dtype=np.float64)
		mask_sen1 = np.zeros(shape=[batch_size, max_len], dtype=np.float64)
		onehot_labels = np.zeros(shape=[batch_size, num_classes], dtype=np.int32)
		
		for id_data in range(batch_size):
			data = batch_data[id_data]
			# sen0
			sen0 = data['sen0']
			embed_sen, mask = self.sen2num(sen0, max_len, vocab, embed)
			embed_sen0[id_data, :, :] = embed_sen
			mask_sen0[id_data, :] = mask
			# sen1
			sen1 = data['sen1']
			embed_sen, mask = self.sen2num(sen1, max_len, vocab, embed)
			embed_sen1[id_data, :, :] = embed_sen
			mask_sen1[id_data, :] = mask
			# label
			onehot_labels[id_data, data['label']] = 1
		return embed_sen0, embed_sen1, mask_sen0, mask_sen1, onehot_labels


if __name__ == '__main__':
	cfg_path = 'config/config.yaml'
	dataset_name = load_config(cfg_path, section='default')['dataset']
	cfg = load_config(cfg_path, section=dataset_name)
	hparams_path = 'config/hparams.yaml'
	hp = load_hparams(hparams_path, section='default')
	
	vocab = load_vocab(cfg['vocab_path'])
	word_emb = load_embed(cfg['word_emb_path'])
	trainset = Dataset(cfg['train_data_path'], dataset_name)
	validset = Dataset(cfg['valid_data_path'], dataset_name)
	
	
	batch_size = 1024
	batch_num = (len(trainset)-1) // batch_size + 1
	for _ in range(batch_num):
		batch_data = trainset.next_batch(batch_num)
		trainset.data2num(batch_data, hp.max_len, hp.num_classes, vocab, word_emb)




