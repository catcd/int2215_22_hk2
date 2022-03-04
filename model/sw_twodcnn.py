import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from math import ceil

from dataset.dataset import pad_sequences
from utils import Timer, Log


seed = 13
np.random.seed(seed)


class SwTwoDCNN:
    def __init__(self, model_name, embeddings, embeddings_wordnet_superset, batch_size, constants, model_variant='NORMAL'):
        self.model_name = model_name
        self.embeddings = embeddings
        self.embeddings_wordnet_superset = embeddings_wordnet_superset
        self.batch_size = batch_size
        self.model_variant = model_variant
        assert model_variant in {'NORMAL', 'DEPENDENCY_UNIT'}, 'model_variant must be \'NORMAL\' or \'DEPENDENCY_UNIT\''

        self.max_sent_length = constants.MAX_SENT_LENGTH

        self.k = constants.K

        self.input_w2v_dim = constants.INPUT_W2V_DIM

        self.cnn_filters = constants.CNN_FILTERS

        # self.use_pos = constant.USE_POS
        # self.npos = constant.NPOS
        # self.input_lstm_pos_dim = constant.INPUT_LSTM_POS_DIM
        # self.output_lstm_pos_dim = constant.OUTPUT_LSTM_POS_DIM

        # self.use_relation = constants.USE_RELATION
        # self.nrelations = constants.NRELATIONS
        # self.relation_embedding_dim = constants.RELATION_EMBEDDING_DIM

        # self.use_direction = constants.USE_DIRECTION
        # self.ndirections = constants.NDIRECTIONS
        # self.direction_embedding_dim = constants.DIRECTION_EMBEDDING_DIM

        # self.use_dependency = self.use_relation or self.use_direction

        # self.use_wordnet_superset = constants.USE_WORDNET_SUPERSET
        # self.input_wordnet_superset_dim = constants.INPUT_WORDNET_SUPERSET_DIM
        # self.output_lstm_wordnet_superset_dim = constants.OUTPUT_LSTM_WORDNET_SUPERSET_DIM

        # self.hidden_layers = constants.HIDDEN_LAYERS

        # self.num_of_class = len(constants.ALL_LABELS)
        # self.all_labels = constants.ALL_LABELS

        # self.use_weighted_loss = constants.USE_WEIGHTED_LOSS
        # self.loss_weight = constants.LOSS_WEIGHT

        # self.trained_models = constants.TRAINED_MODELS

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.max_seq_len = tf.placeholder(dtype=tf.int32, shape=[], name='max_seq_len')
        self.labels = tf.placeholder(name="y_true", shape=[None], dtype='int32')
        self.word_ids = tf.placeholder(name='word_ids', dtype=tf.int32, shape=[None, self.k, None])
        self.relation_ids = tf.placeholder(name='relation_ids', dtype=tf.int32, shape=[None, self.k, None])
        self.direction_ids = tf.placeholder(name='direction_ids', dtype=tf.int32, shape=[None, self.k, None])
        self.dropout_embedding = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_embedding')
        self.dropout_cnn = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_cnn')
        self.dropout_hidden_layer = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_hidden_layer')
        self.is_training = tf.placeholder(tf.bool, name='phase')

        # self.sequence_lens = tf.placeholder(name='sequence_lens', dtype=tf.int32, shape=[None])
        # self.word_pos_ids = tf.placeholder(name='word_pos', shape=[None, None], dtype='int32')
        # self.wordnet_superset_ids = tf.placeholder(name='wordnet_superset_ids', shape=[None, None], dtype='int32')
        # self.char_ids = tf.placeholder(name='char_ids', shape=[None, None, None], dtype='int32')
        # self.word_lengths = tf.placeholder(name="word_lengths", shape=[None, None], dtype='int32')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope("embedding"):
            _embeddings = tf.Variable(self.embeddings, name="lut", dtype=tf.float32, trainable=False)
            self.word_embeddings = tf.nn.embedding_lookup(_embeddings, self.word_ids, name="embeddings")
            self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout_embedding)

        if self.use_dependency:
            if self.use_relation:
                with tf.variable_scope('relation_embedding'):
                    _relation_embeddings = tf.get_variable(
                        name='lut', dtype=tf.float32,
                        shape=[self.nrelations, self.relation_embedding_dim],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                    )
                    relation_embeddings = tf.nn.embedding_lookup(
                        _relation_embeddings, self.relation_ids,
                        name='embeddings'
                    )
            else:
                relation_embeddings = None

            if self.use_direction:
                with tf.variable_scope('direction_embedding'):
                    _direction_embeddings = tf.get_variable(
                        name='lut', dtype=tf.float32,
                        shape=[self.ndirections, self.direction_embedding_dim],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                    )
                    direction_embeddings = tf.nn.embedding_lookup(
                        _direction_embeddings, self.direction_ids,
                        name='embeddings'
                    )
            else:
                direction_embeddings = None

            if self.use_relation and self.use_dependency:
                dependency_embeddings = tf.concat([direction_embeddings, relation_embeddings], axis=-1)
            elif self.use_relation:
                dependency_embeddings = relation_embeddings
            else:
                dependency_embeddings = direction_embeddings

            self.dependency_embeddings = tf.nn.dropout(dependency_embeddings, self.dropout_embedding)
        else:
            self.dependency_embeddings = None

        # if self.use_wordnet_superset:
        #     with tf.variable_scope("wordnet_superset_embedding"):
        #         _wordnet_superset_embeddings = tf.Variable(
        #             self.embeddings_wordnet_superset,
        #             name="lut", dtype=tf.float32, trainable=False
        #         )
        #         self.wordnet_superset_embeddings = tf.nn.embedding_lookup(
        #             _wordnet_superset_embeddings,
        #             self.wordnet_superset_ids,
        #             name="embeddings"
        #         )
        #         self.wordnet_superset_embeddings = tf.nn.dropout(self.wordnet_superset_embeddings, self.dropout_op)

        # if self.use_pos:
        #     with tf.variable_scope('pos_embedding'):
        #         _pos_embeddings = tf.get_variable(
        #             name='lut', dtype=tf.float32,
        #             shape=[self.npos, self.input_lstm_pos_dim],
        #             initializer=tf.contrib.layers.xavier_initializer(),
        #             regularizer=tf.contrib.layers.l2_regularizer(1e-4)
        #         )
        #         pos_embeddings = tf.nn.embedding_lookup(
        #             _pos_embeddings, self.word_pos_ids,
        #             name='embeddings'
        #         )
        #         pos_embeddings = tf.nn.dropout(pos_embeddings, self.dropout_op)

        # if self.use_char:
        #     with tf.variable_scope("chars_embedding"):
        #         _char_embeddings = tf.get_variable(
        #             name="lut", dtype=tf.float32,
        #             shape=[self.nchars, self.input_lstm_char_dim],
        #             initializer=tf.contrib.layers.xavier_initializer()
        #         )
        #         char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="embeddings")
        #         char_embeddings = tf.nn.dropout(char_embeddings, self.dropout_op)
        #
        #         # put the time dimension on axis=1
        #         s = tf.shape(char_embeddings)
        #         char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.input_lstm_char_dim])
        #         word_lengths = tf.reshape(self.word_lengths, shape=[-1])
        #
        #     with tf.variable_scope("bi_lstm_char"):
        #         cell_fw = BasicLSTMCell(self.output_lstm_char_dim)
        #         cell_bw = BasicLSTMCell(self.output_lstm_char_dim)
        #         _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
        #             cell_fw, cell_bw,
        #             char_embeddings,
        #             sequence_length=word_lengths,
        #             dtype=tf.float32
        #         )
        #
        #     with tf.variable_scope("bi_lstm_char_output"):
        #         self.output_char_f = tf.reshape(output_fw, shape=[-1, s[1], self.output_lstm_char_dim])
        #         self.output_char_b = tf.reshape(output_bw, shape=[-1, s[1], self.output_lstm_char_dim])
        #
        #         self.output_char = tf.concat([self.output_char_f, self.output_char_b], axis=-1)

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        final_we = self.word_embeddings
        dim_word_final = (
            self.input_w2v_dim
        )

        final_de = self.dependency_embeddings
        dim_dependency_final = (
            (self.relation_embedding_dim if self.use_relation else 0)
            + (self.direction_embedding_dim if self.use_direction else 0)
        )

        if self.model_variant == 'NORMAL':
            cnn_filter_width = max(dim_word_final, dim_dependency_final)
            cnn_step = 2

            final_we = tf.pad(final_we, [[0, 0], [0, 0], [0, 0], [0, max(dim_dependency_final - dim_word_final, 0)]])
            final_de = tf.pad(final_de, [[0, 0], [0, 0], [0, 1], [0, max(dim_word_final - dim_dependency_final, 0)]])

            stacked_cnn_input = tf.stack([final_we, final_de], axis=-2)
            stacked_cnn_input = tf.reshape(stacked_cnn_input, shape=[-1, self.k, self.max_seq_len*2, max(dim_dependency_final, dim_word_final)])
        else:
            cnn_filter_width = 2*dim_word_final + dim_dependency_final
            cnn_step = 1

            cnn_input_component = []

            we1 = final_we[:, :, :-1, :]
            cnn_input_component.append(we1)

            if self.use_dependency:
                cnn_input_component.append(final_de)

            we2 = final_we[:, :, 1:, :]
            cnn_input_component.append(we2)

            stacked_cnn_input = tf.concat(cnn_input_component, axis=-1)

        with tf.variable_scope("conv"):
            cnn_outputs = []
            stacked_cnn_input = tf.expand_dims(stacked_cnn_input, -1)
            cnn_inputs = tf.unstack(stacked_cnn_input, axis=1)

            for k in self.cnn_filters:
                with tf.variable_scope("cnn-{}".format(k)):
                    filters = self.cnn_filters[k]
                    height = (k * 2 - 1) if self.model_variant == 'NORMAL' else k

                    acnn_ops = []
                    for cnn_input in cnn_inputs:
                        acnn_op = tf.layers.conv2d(
                            cnn_input, filters=filters,
                            kernel_size=(height, cnn_filter_width),
                            strides=(cnn_step, 1),
                            padding="valid", name='cnn-{}'.format(k),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            reuse=tf.AUTO_REUSE,
                            activation=tf.nn.relu,
                        )
                        acnn_ops.append(acnn_op)

                    final_cnn_op = tf.stack(acnn_ops, axis=1)
                    final_cnn_op = tf.reduce_max(final_cnn_op, [1, 2, 3])
                    final_cnn_op = tf.reshape(final_cnn_op, [-1, filters])

                    cnn_outputs.append(final_cnn_op)

            final_cnn_output = tf.concat(cnn_outputs, axis=-1)
            final_cnn_output = tf.nn.dropout(final_cnn_output, self.dropout_cnn)

        with tf.variable_scope("logit"):
            final_features = final_cnn_output
            for i, v in enumerate(self.hidden_layers, start=1):
                final_features = tf.layers.dense(
                    inputs=final_features, units=v, name="hidden_{}".format(i),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                    activation=tf.nn.tanh,
                )
                final_features = tf.nn.dropout(final_features, self.dropout_hidden_layer)

            self.logits = tf.layers.dense(
                inputs=final_features, units=self.num_of_class, name="logit_f",
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )
            self.predict = tf.nn.softmax(self.logits)

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('loss_layers'):
            if self.use_weighted_loss:
                class_weights = tf.constant([self.loss_weight])
                onehot_labels = tf.one_hot(self.labels, self.num_of_class, on_value=1.0, axis=-1)
                weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

                self.loss = 0
                unweighted_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
                weighted_losses = unweighted_losses * weights
                self.loss += tf.reduce_mean(weighted_losses)

            else:
                self.loss = 0
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
                self.loss += tf.reduce_mean(losses)

            regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss += tf.reduce_sum(regularizer)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))
            # self.train_op = optimizer.minimize(self.loss)

    @staticmethod
    def batch_normalization(inputs, training, decay=0.9, epsilon=1e-3):

        scale = tf.get_variable('scale', inputs.get_shape()[-1], initializer=tf.ones_initializer(), dtype=tf.float32)
        beta = tf.get_variable('beta', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        pop_mean = tf.get_variable('pop_mean', inputs.get_shape()[-1], initializer=tf.zeros_initializer(),
                                   dtype=tf.float32, trainable=False)
        pop_var = tf.get_variable('pop_var', inputs.get_shape()[-1], initializer=tf.ones_initializer(),
                                  dtype=tf.float32, trainable=False)

        axis = list(range(len(inputs.get_shape()) - 1))

        def Train():
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            pop_mean_new = pop_mean * decay + batch_mean * (1 - decay)
            pop_var_new = pop_var * decay + batch_var * (1 - decay)
            with tf.control_dependencies([pop_mean.assign(pop_mean_new), pop_var.assign(pop_var_new)]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

        def Eval():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

        return tf.cond(training, Train, Eval)

    def build(self):
        timer = Timer()
        timer.start("Building model...")

        self._add_placeholders()
        self._add_word_embeddings_op()

        self._add_logits_op()
        self._add_loss_op()

        self._add_train_op()
        # f = tf.summary.FileWriter("summary_relation")
        # f.add_graph(tf.get_default_graph())
        # f.close()
        # exit(0)
        timer.stop()

    def load_data(self, train, validation):
        """
        :param dataset.dataset.Dataset train:
        :param dataset.dataset.Dataset validation:
        :return:
        """
        timer = Timer()
        timer.start("Loading data")

        self.dataset_train = train
        self.dataset_validation = validation

        print("Number of training examples:", len(self.dataset_train.labels))
        print("Number of validation examples:", len(self.dataset_validation.labels))
        timer.stop()

    def _accuracy(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.dropout_hidden_layer] = 1.0
        feed_dict[self.is_training] = False

        logits = sess.run(self.logits, feed_dict=feed_dict)
        accuracy = []
        f1 = []
        predict = []
        exclude_label = []
        for logit, label in zip(logits, feed_dict[self.labels]):
            logit = np.argmax(logit)
            exclude_label.append(label)
            predict.append(logit)
            accuracy += [logit == label]

        average = 'macro' if self.num_of_class > 2 else 'binary'
        f1.append(f1_score(predict, exclude_label, average=average))
        return accuracy, np.mean(f1)

    def _loss(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.dropout_hidden_layer] = 1.0
        feed_dict[self.is_training] = False

        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

    def _next_batch(self, data):
        start = 0
        idx = 0
        while start < len(data['words']):
            w_batch = data['words'][start:start + self.batch_size]
            l_batch = data['labels'][start:start + self.batch_size]
            r_batch = data['relations'][start:start + self.batch_size]
            d_batch = data['directions'][start:start + self.batch_size]
            # p_batch = data['poses'][start:start + self.batch_size]
            # wns_batch = data['wnss'][start:start + self.batch_size]

            word_ids, _ = pad_sequences(w_batch, pad_tok=0, sent_length=self.k, nlevels=2)
            relation_ids, _ = pad_sequences(r_batch, pad_tok=0, sent_length=self.k, nlevels=2)
            direction_ids, _ = pad_sequences(d_batch, pad_tok=0, sent_length=self.k, nlevels=2)

            labels = l_batch

            # char_ids, word_lengths = pad_sequences(
            #     char_ids, pad_tok=0, nlevels=2, max_sent_length=self.max_sent_length
            # )
            # pos_ids, _ = pad_sequences(p_batch, max_sent_length=self.max_sent_length, pad_tok=0)
            # wn_superset_ids, _ = pad_sequences(wns_batch, max_sent_length=self.max_sent_length, pad_tok=0)

            start += self.batch_size
            idx += 1
            yield (labels, word_ids, relation_ids, direction_ids)

    def _train(self, epochs, early_stopping=True, patience=10, verbose=True, cont=False):
        Log.verbose = verbose
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

        saver = tf.train.Saver(max_to_keep=2)
        best_f1 = 0.0
        nepoch_noimp = 0
        with tf.Session() as sess:
            if cont:
                saver.restore(sess, self.model_name)
            else:
                sess.run(tf.global_variables_initializer())

            for e in range(epochs):
                labels_shuffled, words_shuffled, relations_shuffled, direction_shuffled = shuffle(
                    self.dataset_train.labels,
                    self.dataset_train.words,
                    self.dataset_train.relations,
                    self.dataset_train.directions,
                    # self.dataset_train.poses,
                    # self.dataset_train.wordnet_supersets,
                )

                data = {
                    'words': words_shuffled,
                    'labels': labels_shuffled,
                    'relations': relations_shuffled,
                    'directions': direction_shuffled,
                    # 'ps': poses_shuffled,
                    # 'wnss': wordnet_supersets_shuffled,
                }

                for idx, batch in enumerate(self._next_batch(data=data)):
                    labels, words, relation, direction = batch
                    feed_dict = {
                        self.max_seq_len: len(words[0][0]),
                        self.word_ids: words,
                        self.labels: labels,
                        self.relation_ids: relation,
                        self.direction_ids: direction,
                        self.dropout_embedding: 0.5,
                        self.dropout_cnn: 0.5,
                        self.dropout_hidden_layer: 0.5,
                        self.is_training: True,
                        # self.word_pos_ids: poses,
                        # self.wordnet_superset_ids: wn_superset,
                        # self.char_ids: chars,
                    }

                    _, loss_train = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if idx % 5 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_train))

                Log.log("End epochs {}".format(e + 1))

                # # stop by loss
                # if early_stopping:
                #     total_loss = []
                #
                # data = {
                #     'words': self.dataset_validation.words,
                #     'labels': self.dataset_validation.labels,
                #     'relations': self.dataset_validation.relations,
                #     'directions': self.dataset_validation.directions,
                #     # 'ps': poses_shuffled,
                #     # 'wnss': wordnet_supersets_shuffled,
                # }
                #
                # for idx, batch in enumerate(self._next_batch(data=data)):
                #     labels, words, relation, direction = batch
                #     feed_dict = {
                #         self.word_ids: words,
                #         self.labels: labels,
                #         self.relation_ids: relation,
                #         self.direction_ids: direction,
                #         # self.word_pos_ids: poses,
                #         # self.wordnet_superset_ids: wn_superset,
                #         # self.char_ids: chars,
                #     }
                #
                #     loss = self._loss(sess, feed_dict=feed_dict)
                #         total_loss.append(loss)
                #
                #     val_loss = np.mean(total_loss)
                #     Log.log('Val loss: {}'.format(val_loss))
                #     if val_loss < best_loss:
                #         saver.save(sess, self.model_name)
                #         Log.log('Save the model at epoch {}'.format(e + 1))
                #         best_loss = val_loss
                #         nepoch_noimp = 0
                #     else:
                #         nepoch_noimp += 1
                #         Log.log("Number of epochs with no improvement: {}".format(nepoch_noimp))
                #         if nepoch_noimp >= patience:
                #             Log.log('Best loss: {}'.format(best_loss))
                #             break

                # stop by F1
                if early_stopping:
                    total_acc = []
                    total_f1 = []

                    data = {
                        'words': self.dataset_validation.words,
                        'labels': self.dataset_validation.labels,
                        'relations': self.dataset_validation.relations,
                        'directions': self.dataset_validation.directions,
                        # 'ps': poses_shuffled,
                        # 'wnss': wordnet_supersets_shuffled,
                    }

                    for idx, batch in enumerate(self._next_batch(data=data)):
                        labels, words, relation, direction = batch
                        feed_dict = {
                            self.max_seq_len: len(words[0][0]),
                            self.word_ids: words,
                            self.labels: labels,
                            self.relation_ids: relation,
                            self.direction_ids: direction,
                            # self.word_pos_ids: poses,
                            # self.wordnet_superset_ids: wn_superset,
                            # self.char_ids: chars,
                        }

                        acc, f1 = self._accuracy(sess, feed_dict=feed_dict)
                        total_acc += acc
                        total_f1.append(f1)

                    f1 = np.mean(total_f1)
                    Log.log("F1 val: {}".format(f1))
                    Log.log('Acc val: {}'.format(np.mean(total_acc)))
                    if f1 > best_f1:
                        saver.save(sess, self.model_name)
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        best_f1 = f1
                        nepoch_noimp = 0
                        # learning_rate/= 1.2
                    else:
                        nepoch_noimp += 1
                        Log.log("Number of epochs with no improvement: {}".format(nepoch_noimp))
                        if nepoch_noimp >= patience:
                            print("Best F1: {}".format(best_f1))
                            break

            if not early_stopping:
                saver.save(sess, self.model_name)

    def run_train(self, epochs, early_stopping=True, patience=10, cont=False):
        timer = Timer()
        timer.start("Training model...")
        self._train(epochs=epochs, early_stopping=early_stopping, patience=patience, cont=cont)
        timer.stop()

    # test
    def predict_on_test(self, test, predict_class=True):
        """

        :param predict_class:
        :param dataset.dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            Log.log("Testing model over test set")
            # a = tf.train.latest_checkpoint(self.model_name)
            saver.restore(sess, self.model_name)

            y_pred = []

            data = {
                'words': test.words,
                'labels': test.labels,
                'relations': test.relations,
                'directions': test.directions,
                # 'ps': poses_shuffled,
                # 'wnss': wordnet_supersets_shuffled,
            }

            for idx, batch in enumerate(self._next_batch(data=data)):
                labels, words, relation, direction = batch
                feed_dict = {
                    self.max_seq_len: len(words[0][0]),
                    self.word_ids: words,
                    self.labels: labels,
                    self.relation_ids: relation,
                    self.direction_ids: direction,
                    self.dropout_embedding: 1.0,
                    self.dropout_cnn: 1.0,
                    self.dropout_hidden_layer: 1.0,
                    self.is_training: False,
                    # self.word_pos_ids: poses,
                    # self.wordnet_superset_ids: wn_superset,
                    # self.char_ids: chars,
                }
                logits = sess.run(self.logits, feed_dict=feed_dict)

                for logit in logits:
                    if predict_class:
                        decode_sequence = np.argmax(logit)
                        y_pred.append(decode_sequence)
                    else:
                        y_pred.append(logit)

        return y_pred
