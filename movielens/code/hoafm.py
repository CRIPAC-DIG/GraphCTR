'''
Tensorflow implementation of HoAFM described in:
HoAFM: A High-order Attentive Factorization Machine for CTR Prediction
'''

import os
import numpy as np
import tensorflow as tf
from time import time
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

'''
The following two functions are adapted from kyubyong park's implementation of transformer
We slightly modify the code to make it suitable for our work.(add relu, delete key masking and causality mask)
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

class HOAFM():
    def __init__(self, args, feature_size, run_cnt):

        self.feature_size = feature_size  # denote as n, dimension of concatenated features
        self.field_size = args.field_size  # denote as M, number of total feature fields
        self.embedding_size = args.embedding_size  # denote as d, size of the feature embedding
        self.blocks = args.blocks  # number of the blocks
        self.heads = args.heads  # number of the heads
        self.block_shape = args.block_shape
        # self.output_size = args.block_shape[-1]

        self.has_residual = args.has_residual
        self.deep_layers = args.deep_layers  # whether to joint train with deep networks as described in paper

        self.batch_norm = args.batch_norm
        self.batch_norm_decay = args.batch_norm_decay
        self.drop_keep_prob = args.dropout_keep_prob
        self.l2_reg = args.l2_reg
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer_type = args.optimizer_type

        self.save_path = args.save_path + str(run_cnt) + '/'
        self.is_save = args.is_save
        if (args.is_save == True and os.path.exists(self.save_path) == False):
            os.makedirs(self.save_path)

        self.verbose = args.verbose
        self.random_seed = args.random_seed
        self.loss_type = args.loss_type
        self.eval_metric = roc_auc_score
        self.best_loss = 1.0
        self.greater_is_better = args.greater_is_better
        self.train_result, self.valid_result = [], []
        self.train_loss, self.valid_loss = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            # placeholder for single-value field.
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index")  # None * M-1
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name="feat_value")  # None * M-1

            # placeholder for multi-value field. (movielens dataset genre field)
            self.genre_index = tf.placeholder(tf.int32, shape=[None, None],
                                              name="genre_index")  # None * 6
            self.genre_value = tf.placeholder(tf.float32, shape=[None, None],
                                              name="genre_value")  # None * 6

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1

            # In our implementation, the shape of dropout_keep_prob is [3], used in 3 different places.
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_prob")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                     self.feat_index)  # None * M-1 * d
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size - 1, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)  # None * M-1 * d

            # for multi-value field
            self.embeddings_m = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                       self.genre_index)  # None * 6 * d
            genre_value = tf.reshape(self.genre_value, shape=[-1, 6, 1])
            self.embeddings_m = tf.multiply(self.embeddings_m, genre_value)
            self.embeddings_m = tf.reduce_sum(self.embeddings_m, axis=1)  # None * d
            self.embeddings_m = tf.div(self.embeddings_m,
                                       tf.reduce_sum(self.genre_value, axis=1, keep_dims=True))  # None * d

            # concatenate single-value field with multi-value field
            self.embeddings = tf.concat([self.embeddings, tf.expand_dims(self.embeddings_m, 1)], 1)  # None * M * d
            self.embeddings = tf.nn.dropout(self.embeddings, self.dropout_keep_prob[1])  # None * M * d

            # joint training with feedforward nn
            if self.deep_layers != None:
                self.y_dense = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
                for i in range(0, len(self.deep_layers)):
                    self.y_dense = tf.add(tf.matmul(self.y_dense, self.weights["layer_%d" % i]),
                                          self.weights["bias_%d" % i])  # None * layer[i]
                    if self.batch_norm:
                        self.y_dense = self.batch_norm_layer(self.y_dense, train_phase=self.train_phase,
                                                             scope_bn="bn_%d" % i)
                    self.y_dense = tf.nn.relu(self.y_dense)
                    self.y_dense = tf.nn.dropout(self.y_dense, self.dropout_keep_prob[1])
                self.y_dense = tf.add(tf.matmul(self.y_dense, self.weights["prediction_dense"]),
                                      self.weights["prediction_bias_dense"], name='logits_dense')  # None * 1

            # ---------- main part of HOAFM-------------------
            self.y_deep = self.embeddings  # None * M * d

            for i in range(self.blocks):
                A_deep_order = self.embeddings
                slice_list = []
                for j in range(self.field_size):
                    slice_left = self.y_deep[:, 0:j, :]
                    slice_right = self.y_deep[:, (j+1):, :]
                    slice_j = tf.concat((slice_left, slice_right), axis=1)
                    # self.weights['field_weight_layer%d_slice%d' % (i, j)] = tf.contrib.sparsemax.sparsemax(
                    #    self.weights['field_weight_layer%d_slice%d' % (i, j)])
                    self.weights['field_weight_layer%d_slice%d' % (i, j)] = tf.nn.l2_normalize(
                        self.weights['field_weight_layer%d_slice%d' % (i, j)], dim=0)

                    y_slice_j = tf.multiply(slice_j, self.weights['field_weight_layer%d_slice%d' % (i, j)])
                    slice_list.append(tf.reduce_sum(y_slice_j, axis=1))
                B_deep_order = tf.transpose(slice_list, perm=[1, 0, 2]) # [batch_size, field_size, hidden_size]
                # B_deep_order = self.xn_activation(B_deep_order)
                # B_deep_order = tf.nn.dropout(B_deep_order, self.xn_keep[i])
                self.y_deep = tf.multiply(A_deep_order, B_deep_order)
                # self.y_deep = self.activation(self.y_deep)
                # self.y_deep = tf.nn.dropout(self.y_deep, self.xn_keep[i])

                # the i-th order output

                self.y_deep_comp = tf.reduce_sum(self.y_deep, axis=1)
                if self.batch_norm:
                    self.y_deep_comp = self.batch_norm_layer(self.y_deep_comp, train_phase=self.train_phase,
                                                             scope_bn="y_deep_comp%d" % i)
                # self.y_deep_comp = self.activation(self.y_deep_comp)
                self.y_deep_comp = tf.nn.dropout(self.y_deep_comp, self.dropout_keep_prob[1])
                self.y_deep_comp = tf.matmul(self.y_deep_comp, self.weights['comp_weight_%d' % i])  # None * 1
                # self.out = tf.add(self.out, self.y_deep_comp)
                self.out = self.y_deep_comp

                self.y_deep = tf.nn.leaky_relu(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_prob[1])

            if self.deep_layers != None:
                self.out += self.y_dense

            # ---------- Compute the loss ----------
            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out, name='pred')
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # l2 regularization on weights
            if self.l2_reg > 0:
                if self.deep_layers != None:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.var1 = [v for v in tf.trainable_variables() if v.name != 'feature_bias:0']
            self.var2 = [tf.trainable_variables()[1]]  # self.var2 = [feature_bias]

            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        beta1=0.9, beta2=0.999, epsilon=1e-8). \
                    minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8). \
                    minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate). \
                    minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95). \
                    minimize(self.loss, global_step=self.global_step)

            # init
            self.saver = tf.train.Saver(max_to_keep=5)
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            self.count_param()

    def readout(self, state, i):
        with tf.variable_scope("readout"):
            if i > 0: tf.get_variable_scope().reuse_variables()
            g = tf.layers.dense(state, self.output_size, activation=tf.tanh, use_bias=True, name="weights")
            # g = tf.nn.dropout(g, self.dropout_keep_prob[1])
            output = tf.reduce_max(g, axis=1) + tf.reduce_sum(g, axis=1)
            #output = tf.reduce_sum(g, axis=1)

        return output

    def prediction_layer(self, states):
        emb = []
        for i, state in enumerate(states):
            g_emb = self.readout(state, i)
            emb.append(g_emb)
        emb = tf.reshape(tf.concat(emb, 1), [-1, self.output_size * (self.GNN_step + 1)])
        with tf.variable_scope("prediction"):
            score = tf.layers.dense(emb, 1, activation=None, use_bias=True,
                                        name="score")

        return score

    def count_param(self):
        k = (np.sum([np.prod(v.get_shape().as_list())
                     for v in tf.trainable_variables()]))

        print("total parameters :%d" % k)
        print("extra parameters : %d" % (k - self.feature_size * self.embedding_size))

    def _init_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size(n) * d

        # input_size = self.output_size * self.field_size

        # dense layers
        if self.deep_layers != None:
            num_layer = len(self.deep_layers)
            layer0_size = self.field_size * self.embedding_size
            glorot = np.sqrt(2.0 / (layer0_size + self.deep_layers[0]))
            weights["layer_0"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(layer0_size, self.deep_layers[0])), dtype=np.float32)
            weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                            dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            glorot = np.sqrt(2.0 / (self.deep_layers[-1] + 1))
            weights["prediction_dense"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                dtype=np.float32, name="prediction_dense")
            weights["prediction_bias_dense"] = tf.Variable(
                np.random.normal(), dtype=np.float32, name="prediction_bias_dense")

        # # ---------- prediciton weight ------------------#
        # glorot = np.sqrt(2.0 / (input_size + 1))
        # weights["prediction"] = tf.Variable(
        #     np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
        #     dtype=np.float32, name="prediction")
        # weights["prediction_bias"] = tf.Variable(
        #     np.random.normal(), dtype=np.float32, name="prediction_bias")

        # ---------- attention weight ------------------#
        initializer = tf.contrib.layers.xavier_initializer()
        for i in range(self.blocks):
            weights["A_w_%d" % i] = tf.Variable(initializer([self.embedding_size, self.block_shape[i]]),
                                                name="A_w_%d" % i)
            if i == 0:
                pre_emb_size = self.embedding_size
            else:
                pre_emb_size = self.block_shape[i - 1]
            weights["B_w_%d" % i] = tf.Variable(initializer([pre_emb_size, self.block_shape[i]]),
                                                name="B_w_%d" % i)
            weights['comp_weight_%d' % i] = tf.Variable(initializer([self.embedding_size, 1]),
                                                        name='comp_weight_%d' % i)
            weights['comp_bias_%d' % i] = tf.Variable(initializer([1]), name='comp_bias_%d' % i)

            for j in range(self.field_size):
                weights['field_weight_layer%d_slice%d' % (i, j)] = tf.Variable(
                    initializer([self.field_size - 1, self.embedding_size]), name='field_weight_layer%d_slice%d' % (i, j))

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch(self, Xi, Xv, Xi_genre, Xv_genre, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], Xi_genre[start:end], Xv_genre[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c, d, e):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)

    def fit_on_batch(self, Xi, Xv, Xi_genre, Xv_genre, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.genre_index: Xi_genre,
                     self.genre_value: Xv_genre,
                     self.label: y,
                     self.dropout_keep_prob: self.drop_keep_prob,
                     self.train_phase: True}
        step, loss, opt = self.sess.run((self.global_step, self.loss, self.optimizer), feed_dict=feed_dict)
        return step, loss

    # Since the train data is very large, they can not be fit into the memory at the same time.
    # We separate the whole train data into several files and call "fit_once" for each file.
    def fit_once(self, Xi_train, Xv_train, Xi_train_genre, Xv_train_genre, y_train,
                 epoch, Xi_valid=None,
                 Xv_valid=None, Xi_valid_genre=None, Xv_valid_genre=None, y_valid=None,
                 early_stopping=False):

        has_valid = Xv_valid is not None
        last_step = 0
        t1 = time()
        self.shuffle_in_unison_scary(Xi_train, Xv_train, Xi_train_genre, Xv_train_genre, y_train)
        total_batch = int(len(y_train) / self.batch_size)
        for i in range(total_batch):
            Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi_train, Xv_train,
                                                                                         Xi_train_genre, Xv_train_genre,
                                                                                         y_train, self.batch_size, i)
            step, loss = self.fit_on_batch(Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch)
            last_step = step

        # evaluate training and validation datasets
        train_result, train_loss = self.evaluate(Xi_train, Xv_train, Xi_train_genre, Xv_train_genre, y_train)
        self.train_result.append(train_result)
        self.train_loss.append(train_loss)
        if has_valid:
            valid_result, valid_loss = self.evaluate(Xi_valid, Xv_valid, Xi_valid_genre, Xv_valid_genre, y_valid)
            self.valid_result.append(valid_result)
            self.valid_loss.append(valid_loss)
            if valid_loss < self.best_loss and self.is_save == True:
                old_loss = self.best_loss
                self.best_loss = valid_loss
                self.saver.save(self.sess, self.save_path + 'model.ckpt', global_step=last_step)
                print("[%d] model saved!. Valid loss is improved from %.4f to %.4f"
                      % (epoch, old_loss, self.best_loss))

        if self.verbose > 0 and ((epoch - 1) * 9) % self.verbose == 0:
            if has_valid:
                print("[%d] train-result=%.4f, train-logloss=%.4f, valid-result=%.4f, valid-logloss=%.4f [%.1f s]" % (
                epoch, train_result, train_loss, valid_result, valid_loss, time() - t1))
            else:
                print("[%d] train-result=%.4f [%.1f s]" \
                      % (epoch, train_result, time() - t1))
        if has_valid and early_stopping and self.training_termination(self.valid_loss):
            return False
        else:
            return True

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def predict(self, Xi, Xv, Xi_genre, Xv_genre):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi, Xv, Xi_genre, Xv_genre,
                                                                                     dummy_y, self.batch_size,
                                                                                     batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.genre_index: Xi_batch_genre,
                         self.genre_value: Xv_batch_genre,
                         self.label: y_batch,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi, Xv, Xi_genre, Xv_genre,
                                                                                         dummy_y, self.batch_size,
                                                                                         batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, Xi_genre, Xv_genre, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv, Xi_genre, Xv_genre)
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
        return self.eval_metric(y, y_pred), log_loss(y, y_pred)

    def restore(self, save_path=None):
        if (save_path == None):
            save_path = self.save_path
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            if self.verbose > 0:
                print("restored from %s" % (save_path))
