import argparse
import datetime
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from utils import Dataset, feed_data
from utils import load_config, load_hparams, load_vocab, load_embed
from model import Decatt
from logger import create_logger


def evaluate(dataset, model, sess, vocab, word_emb):
	batch_size = 1024
	num_batch = (len(dataset) - 1) // batch_size + 1
	num_sample = 0
	loss = 0
	labels = np.array([])
	pred = np.array([])
	for id_batch in range(num_batch):
		batch_data = dataset.next_batch(batch_size)
		len_batch_data = len(batch_data)
		batch_data = dataset.data2num(batch_data, max_len=model.hp.max_len, num_classes=model.hp.num_classes, vocab=vocab, embed=word_emb)
		feed_dict = feed_data(model, *batch_data, training=False)
		loss_batch, labels_batch, pred_batch = sess.run(fetches=(model.loss, model.labels, model.pred), feed_dict=feed_dict)
		num_sample += len_batch_data
		loss += loss_batch * len_batch_data
		labels = np.concatenate([labels, labels_batch])
		pred = np.concatenate([pred, pred_batch])
	loss = loss / num_sample
	TP = np.count_nonzero(pred * labels)
	TN = np.count_nonzero((1 - pred) * (1 - labels))
	FP = np.count_nonzero(pred * (1 - labels))
	FN = np.count_nonzero((1 - pred) * labels)
	acc = (TP + TN) / (TP + TN + FP + FN)
	if TP:
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		f1 = 2 * precision * recall / (precision + recall)
	else:
		precision, recall, f1 = 0, 0, 0
	
	return loss, acc, precision, recall, f1


def train_epoch(trainset, model, sess, vocab, word_emb, id_epoch, FLAGS):
	num_batch = (len(trainset) - 1) // model.hp.batch_size + 1
	num_sample, loss_total, correct_total = 0, 0, 0
	for id_batch in range(num_batch):
		batch_data = trainset.next_batch(model.hp.batch_size)
		len_batch_data = len(batch_data)
		batch_data = trainset.data2num(batch_data, max_len=model.hp.max_len, num_classes=model.hp.num_classes, vocab=vocab, embed=word_emb)
		feed_dict = feed_data(model, *batch_data, training=True)
		if FLAGS.print:
			loss_batch, correct_batch, learning_rate, global_step, _, _ = sess.run(
				fetches=(model.loss, model.correct, model.learning_rate, model.global_step, model.train, model.print), feed_dict=feed_dict)
		else:
			loss_batch, correct_batch, learning_rate, global_step, _ = sess.run(
				fetches=(model.loss, model.correct, model.learning_rate, model.global_step, model.train), feed_dict=feed_dict)
		num_sample += len_batch_data
		loss_total += loss_batch * len_batch_data
		correct_total += np.sum(correct_batch)
		if (id_batch + 1) % 100 == 0:
			loss_train = loss_total / num_sample
			acc_train = correct_total / num_sample
			logger.info('Epoch: %d\tbatch: %d\tloss_train: %.4f\tacc_train: %.4f\tlr: %.4e\tstep: %d',
				id_epoch + 1, id_batch+1, loss_train, acc_train, learning_rate, global_step)
			num_sample, loss_total, correct_total = 0, 0, 0


def train(dataset_name, logger, cfg, hp, FLAGS):
	vocab = load_vocab(cfg['vocab_path'])
	word_emb = load_embed(cfg['word_emb_path'])
	
	trainset = Dataset(cfg['train_data_path'], dataset_name=dataset_name)
	logger.info('trainset label_ratio: {0}'.format(trainset.label_ratio()))
	validset = Dataset(cfg['valid_data_path'], dataset_name=dataset_name)
	logger.info('validset label_ratio: {0}'.format(validset.label_ratio()))
	
	logger.info('loading model...')
	graph = tf.Graph()
	with graph.as_default():
		model = Decatt(hparams=hp, logger=logger)
	
	with tf.Session(graph=graph) as sess:
		# debug
		if FLAGS.debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root='tfdbg')
		# init
		sess.run(tf.global_variables_initializer())
		logger.info('params initialized')
		# create a saver
		saver = tf.train.Saver()
		save_path = cfg['save_model_path']
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		# performance of the model before training
		loss_valid, acc, precision, recall, f1 = evaluate(validset, model, sess, vocab, word_emb)
		logger.info('loss_valid: %.4f\tacc: %.4f\tprecision: %.4f\trecall: %.4f\tf1: %.4f', loss_valid, acc, precision, recall, f1)
		best_result = {'loss_valid': loss_valid, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
		patience = 0
		# train
		for id_epoch in range(hp.num_epoch):
			train_epoch(trainset, model, sess, vocab, word_emb, id_epoch, FLAGS)
			loss_valid, acc, precision, recall, f1 = evaluate(validset, model, sess, vocab, word_emb)
			logger.info('Epoch: %d\tloss_valid: %.4f\tacc: %.4f\tprecision: %.4f\trecall: %.4f\tf1: %.4f', id_epoch + 1, loss_valid, acc, precision, recall, f1)
			if loss_valid < best_result['loss_valid']:
				saver.save(sess=sess, save_path=save_path)
				logger.info('model saved in %s', save_path)
				best_result = {'loss_valid': loss_valid, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
				patience = 0
			else:
				patience += 1
				if patience >= hp.earlystop_patience:
					logger.info('earlystop.')
					logger.info('Best result: loss_valid: %.4f\tacc: %.4f\tprecision: %.4f\trecall: %.4f\tf1: %.4f',
					            best_result['loss_valid'], best_result['acc'], best_result['precision'],
					            best_result['recall'], best_result['f1'])
					break
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', nargs='?', const=True, default=False)
	parser.add_argument('--print', nargs='?', const=True, default=False)
	FLAGS = parser.parse_args()
	
	cfg_path = 'config/config.yaml'
	dataset_name = load_config(cfg_path, section='default')['dataset']
	logger = create_logger(dataset_name)
	logger.info('dataset:%s', dataset_name)
	cfg = load_config(cfg_path, section=dataset_name)
	hparams_path = 'config/hparams.yaml'
	hp = load_hparams(hparams_path, section=dataset_name)
	
	train(dataset_name, logger, cfg, hp, FLAGS)
