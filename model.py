#!/usr/bin/python3
'''
    Author: Yi Zhang
    Date created: 5/14/2019
'''
import sys

import tensorflow as tf
from utils import load_config, load_hparams


class Decatt(object):
	def __init__(self, hparams, logger):
		# hparams
		self._hp = hparams
		self._logger = logger
		# placeholder
		self.embed_sen0 = tf.placeholder(dtype=tf.float64, shape=[None, self._hp.max_len, self._hp.embed_size],
		                                 name='embed_sen0')
		self.embed_sen1 = tf.placeholder(dtype=tf.float64, shape=[None, self._hp.max_len, self._hp.embed_size],
		                                 name='embed_sen1')
		self.mask_sen0 = tf.placeholder(dtype=tf.float64, shape=[None, self._hp.max_len], name='mask_sen0')
		self.mask_sen1 = tf.placeholder(dtype=tf.float64, shape=[None, self._hp.max_len], name='mask_sen1')
		self.onehot_labels = tf.placeholder(dtype=tf.int32, shape=[None, self._hp.num_classes], name='Onehot_labels')
		self.training = tf.placeholder(dtype=tf.bool, shape=None, name='training')
		
		# Decatt model
		self._initializer = tf.initializers.glorot_normal()
		self._project()
		self._attend()
		self._compare()
		self._aggregate()
		# calculate loss, pred, prob, correct, acc based on logits
		self._calc_loss()
		self._prob = tf.nn.softmax(self._logits, axis=1, name='Prob')
		self._pred = tf.argmax(self._logits, axis=1, output_type=tf.int32, name='Pred')
		self.labels = tf.argmax(self.onehot_labels, axis=1, output_type=tf.int32, name='Label')
		self._correct = tf.equal(self._pred, self.labels, name='Correct')
		
		self.train = self._train_op()
		self.print = self._print_op()
		self._logger.info(self._tvars)
		self._logger.info(self.hp)
	
	def _fnn_block(self, inputs, num_layers, units):
		outputs = inputs
		for id_layer in range(num_layers):
			with tf.name_scope(name='FNN_%d' % id_layer):
				outputs = tf.keras.layers.Dropout(rate=1 - self._hp.keep_prop).apply(outputs, training=self.training)
				outputs = tf.layers.dense(outputs, units=units, activation=tf.nn.relu,
				                          kernel_initializer=self._initializer)
		return outputs
	
	def _project(self):
		with tf.variable_scope('Project'):
			self._sen0 = tf.layers.dense(self.embed_sen0, units=self._hp.hidden_size, activation=tf.nn.relu,
			                             kernel_initializer=self._initializer)
		with tf.variable_scope('Project', reuse=True):
			self._sen1 = tf.layers.dense(self.embed_sen1, units=self._hp.hidden_size, activation=tf.nn.relu,
			                             kernel_initializer=self._initializer)
	
	def _attend(self):
		with tf.variable_scope('Attend'):
			f_sen0 = self._fnn_block(self._sen0, num_layers=self._hp.num_layers, units=self._hp.hidden_size)
		with tf.variable_scope('Attend', reuse=True):
			f_sen1 = self._fnn_block(self._sen1, num_layers=self._hp.num_layers, units=self._hp.hidden_size)
		score = tf.matmul(f_sen0, f_sen1, transpose_b=True, name='Att_score')
		mask = tf.matmul(tf.expand_dims(self.mask_sen0, axis=2), tf.expand_dims(self.mask_sen1, axis=1))
		score = score - (1 - mask) * 1.0e8
		self.att_on_sen0 = tf.nn.softmax(score, axis=1, name='Att_1on0')
		self.att_on_sen1 = tf.nn.softmax(score, axis=2, name='Att_0on1')
		self._compare_sen0 = tf.matmul(self.att_on_sen0, self._sen0, transpose_a=True, name='Comp_sen0')
		self._compare_sen1 = tf.matmul(self.att_on_sen1, self._sen1, name='Comp_sen1')
	
	def _compare(self):
		with tf.variable_scope('Compare'):
			self._cmp_results_sen0_words = self._fnn_block(tf.concat([self._sen0, self._compare_sen1], axis=2),
			                                               num_layers=self._hp.num_layers, units=self._hp.hidden_size)
		with tf.variable_scope('Compare', reuse=True):
			self._cmp_results_sen1_words = self._fnn_block(tf.concat([self._sen1, self._compare_sen0], axis=2),
			                                               num_layers=self._hp.num_layers, units=self._hp.hidden_size)
	
	def _aggregate(self):
		# self._cmp_results_sen0 = tf.reduce_sum(self._cmp_results_sen0_words, axis=1)
		# self._cmp_results_sen1 = tf.reduce_sum(self._cmp_results_sen1_words, axis=1)
		self._cmp_results_sen0 = tf.squeeze(
			tf.matmul(self._cmp_results_sen0_words, tf.expand_dims(self.mask_sen0, axis=2), transpose_a=True), axis=2)
		self._cmp_results_sen1 = tf.squeeze(
			tf.matmul(self._cmp_results_sen1_words, tf.expand_dims(self.mask_sen1, axis=2), transpose_a=True), axis=2)
		self._cmp_results = tf.concat([self._cmp_results_sen0, self._cmp_results_sen1], axis=1)
		with tf.variable_scope('Aggregate'):
			hidden = self._fnn_block(self._cmp_results, num_layers=self._hp.num_layers, units=self._hp.hidden_size)
			hidden = tf.keras.layers.Dropout(rate=1 - self._hp.keep_prop).apply(hidden, training=self.training)
			self._logits = tf.keras.layers.Dense(units=self._hp.num_classes,
			                                     kernel_initializer=self._initializer).apply(hidden)
	
	def _calc_loss(self):
		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels, logits=self._logits,
		                                             label_smoothing=self._hp.label_smoothing)
		self._tvars = tf.trainable_variables()
		if self._hp.scale_l2:
			loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self._tvars if 'bias' not in v.name])
			self._loss = self._loss + self._hp.scale_l2 * loss_l2
	
	def _train_op(self):
		self._global_step = tf.train.create_global_step()
		self._learning_rate = self._hp.learning_rate
		self._learning_rate = tf.train.exponential_decay(learning_rate=self._learning_rate,
		                                                 global_step=self._global_step,
		                                                 decay_steps=self._hp.decay_steps,
		                                                 decay_rate=self._hp.decay_rate,
		                                                 staircase=False)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
		gvs = self.optimizer.compute_gradients(self._loss)
		g, v = zip(*gvs)
		if self._hp.grad_max:
			g, _ = tf.clip_by_global_norm(g, self._hp.grad_max)
		clipped_gvs = zip(g, v)
		return self.optimizer.apply_gradients(clipped_gvs, global_step=self._global_step)
	
	# for debug
	def _print_op(self):
		# for debug
		tensors_to_print = [self._tvars]
		print_op = tf.print(*tensors_to_print, output_stream=sys.stdout)
		return print_op
	
	@property
	def hp(self):
		return self._hp
	
	@property
	def loss(self):
		return self._loss
	
	@property
	def correct(self):
		return self._correct
	
	@property
	def pred(self):
		return self._pred
	
	@property
	def learning_rate(self):
		return self._learning_rate
	
	@property
	def global_step(self):
		return self._global_step


if __name__ == '__main__':
	cfg_path = 'config/config.yaml'
	cfg = load_config(cfg_path, section='atec')
	hparams_path = 'config/hparams.yaml'
	hp = load_hparams(hparams_path, section='default')
	model = Decatt(hp)
