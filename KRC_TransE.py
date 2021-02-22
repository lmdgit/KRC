#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class KRC_TransE(Model):
	
	def _calc(self, h, t, r, ws, wo, bs, bo, lambda_s=0.3, lambda_o=0.3):
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)

		h_r = tf.nn.l2_normalize(h * ws,-1)
		t_r = tf.nn.l2_normalize(t * wo,-1)
		rh = tf.nn.l2_normalize(bs * ws,-1)
		rt = tf.nn.l2_normalize(bo * wo,-1)

		return abs(h + r - t) + lambda_s * abs(h_r - rh) + lambda_o * abs(t_r - rt)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.wrs_embeddings = tf.get_variable(name="wrs_embeddings", shape=[config.relTotal, config.hidden_size],
											  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		self.wro_embeddings = tf.get_variable(name="wro_embeddings", shape=[config.relTotal, config.hidden_size],
											  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		self.brs_embeddings = tf.get_variable(name="brs_embeddings", shape=[config.relTotal, config.hidden_size],
											  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		self.bro_embeddings = tf.get_variable(name="bro_embeddings", shape=[config.relTotal, config.hidden_size],
											  initializer=tf.contrib.layers.xavier_initializer(uniform=False))


	def loss_def(self, soft_margin=False):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		
		p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		p_wrs = tf.nn.embedding_lookup(self.wrs_embeddings, pos_r)
		p_wro = tf.nn.embedding_lookup(self.wro_embeddings, pos_r)
		p_brs = tf.nn.embedding_lookup(self.brs_embeddings, pos_r)
		p_bro = tf.nn.embedding_lookup(self.bro_embeddings, pos_r)
        
		n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		n_wrs = tf.nn.embedding_lookup(self.wrs_embeddings, neg_r)
		n_wro = tf.nn.embedding_lookup(self.wro_embeddings, neg_r)
		n_brs = tf.nn.embedding_lookup(self.brs_embeddings, neg_r)
		n_bro = tf.nn.embedding_lookup(self.bro_embeddings, neg_r)

		_p_score = self._calc(p_h, p_t, p_r, p_wrs, p_wro, p_brs, p_bro)
		_n_score = self._calc(n_h, n_t, n_r, n_wrs, n_wro, n_brs, n_bro)

		p_score = tf.reduce_sum(_p_score, axis=-1, keep_dims = True)
		n_score = tf.reduce_sum(_n_score, axis=-1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		if soft_margin:
			wr = self.get_weight(in_batch=True)
			margin = wr*config.margin
		else:
			margin = config.margin

		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + margin, 0))

	def predict_def(self):
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
		predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
		predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
        
		predict_wrs = tf.nn.embedding_lookup(self.wrs_embeddings, predict_r)
		predict_wro = tf.nn.embedding_lookup(self.wro_embeddings, predict_r)
		predict_brs = tf.nn.embedding_lookup(self.brs_embeddings, predict_r)
		predict_bro = tf.nn.embedding_lookup(self.bro_embeddings, predict_r)
		self.predict = tf.reduce_mean(
			self._calc(predict_h_e, predict_t_e, predict_r_e, predict_wrs, predict_wro, predict_brs, predict_bro), -1,
			keep_dims=False)




