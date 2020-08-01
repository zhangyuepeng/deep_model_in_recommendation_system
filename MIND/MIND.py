# -*- coding: utf8 -*-
from time import time

from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import Embedding, Dense, Flatten, Concatenate
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import optimizers
from tensorflow.python.keras.regularizers import l2

from Dataset import Dataset
DEBUG = False


def debug_print(msg):
    if DEBUG:
        print(msg)


def squash(inputs):
    vec_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std

    def build(self, input_shape):
        super(CapsuleLayer, self).build(input_shape)
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=RandomNormal(stddev=self.init_std),
                                              trainable=False, name="B")
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(stddev=0.01),
                                                       name="S")

    def call(self, inputs, **kwargs):
        behavior_embeddings, seq_len = inputs
        batch_size = tf.shape(behavior_embeddings)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])
        mask = tf.sequence_mask(seq_len_tile, self.max_len)
        debug_print(mask)
        pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)

        for i in range(self.iteration_times):
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]), pad)
            weight = tf.nn.softmax(routing_logits_with_padding)
            behavior_embdding_mapping = tf.tensordot(behavior_embeddings, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embdding_mapping)
            interest_capsules = squash(Z)
            delta_routing_logits = tf.reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(behavior_embdding_mapping, perm=[0, 2, 1])),
                axis=0, keepdims=True
            )
            self.routing_logits.assign_add(delta_routing_logits)
        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules


class MINDModel(Model):

    def __init__(self, num_items, num_brands=0, num_categories=0, embedding_dim=32, neg_sampled=5,
                 dnn_hidden_units=[64, 32], l2_dnn_reg=[0, 0], max_seq_len=100, k_max=4, use_user_feat=False):
        super(MINDModel, self).__init__()
        self.num_items = num_items
        self.num_brands = num_brands
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.neg_sampled = neg_sampled
        self.k_max = k_max
        self.use_user_feat = use_user_feat
        self.item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding',
                                        embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                        mask_zero=True)
        if num_brands > 0:
            self.brands_embedding = Embedding(input_dim=num_brands, output_dim=embedding_dim, name='brands_embedding',
                                              embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                              mask_zero=True)

        if num_categories > 0:
            self.categories_embedding = Embedding(input_dim=num_categories, output_dim=embedding_dim, name='categories_embedding',
                                                  embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                                  mask_zero=True)
        self.flatten = Flatten()
        self.average = tf.keras.layers.Average()
        self.capsule = CapsuleLayer(embedding_dim, embedding_dim, max_seq_len, k_max)
        self.concatenate = Concatenate(axis=-1)
        self.dense_layers = []
        for idx in range(0, len(dnn_hidden_units)):
            self.dense_layers.append(Dense(dnn_hidden_units[idx], kernel_regularizer=l2(l2_dnn_reg[idx]), activation='relu',
                                           name='user_embedding_%d' % idx))

    def call(self, inputs):
        """
        :param inputs: type：list；structure:[behavior_item_list, behavior_brands_list, behavior_categories_list,
                                             target_item_list, target_brands_list, target_categories_list,
                                             user_profile_feature, seq_len]
                       user_profile_feature: for easy, we directly use the user_profile_feature, omit the embedding layer
                       seq_len: the behavior length
        :return:
        """
        item_embedding = self.item_embedding(inputs[0])
        target_item_embedding = self.item_embedding(inputs[3])
        debug_print(item_embedding.shape)
        debug_print(target_item_embedding.shape)
        if self.num_brands > 0:
            brands_embedding = self.brands_embedding(inputs[1])
            target_brands_embedding = self.brands_embedding(inputs[4])
        if self.num_categories > 0:
            categories_embedding = self.categories_embedding(inputs[2])
            target_categories_embedding = self.categories_embedding(inputs[5])

        if self.num_brands > 0 and self.num_categories > 0:
            avg_embedding = self.average([item_embedding, brands_embedding, categories_embedding])
            target_avg_embedding = self.average([target_item_embedding, target_brands_embedding, target_categories_embedding])
        elif self.num_brands > 0:
            avg_embedding = self.average([item_embedding, brands_embedding])
            target_avg_embedding = self.average([target_item_embedding, target_brands_embedding])
        elif self.num_categories > 0:
            avg_embedding = self.average([item_embedding, categories_embedding])
            target_avg_embedding = self.average([target_item_embedding, target_categories_embedding])
        else:
            avg_embedding = item_embedding
            target_avg_embedding = target_item_embedding
        debug_print(avg_embedding.shape)
        debug_print(target_avg_embedding.shape)
        capsule_embeddings = self.capsule([avg_embedding, inputs[7]])
        debug_print(capsule_embeddings)
        user_profile = inputs[6]
        if self.use_user_feat:
            profile_embeddings = tf.tile(tf.expand_dims(user_profile, axis=1), [1, self.k_max, 1])
            debug_print(profile_embeddings.shape)
            user_embeddings = self.concatenate([capsule_embeddings, profile_embeddings])
        else:
            user_embeddings = capsule_embeddings
        debug_print("----------------")
        debug_print(user_embeddings.shape)
        for dense_layer in self.dense_layers:
            user_embeddings = dense_layer(user_embeddings)
            debug_print(user_embeddings.shape)
        debug_print("-------LabelAwareAttention---------")
        debug_print(target_avg_embedding.shape)
        target_avg_embedding = tf.reshape(target_avg_embedding, [-1, self.embedding_dim, 1])
        debug_print(target_avg_embedding.shape)
        weight = tf.matmul(user_embeddings, target_avg_embedding)
        debug_print(weight)
        index = tf.argmax(weight, 1)
        debug_print(index)
        user_embeddings_select = tf.gather_nd(user_embeddings, index, batch_dims=1)
        debug_print(user_embeddings_select.shape)
        return user_embeddings_select


if __name__ == '__main__':

    embedding_dim = 32
    # Loading data
    t1 = time()
    dataset = Dataset("../data/ml-1m")
    debug_print("Load data done [%.1f s]." % (time() - t1))
    mind_model = MINDModel(num_items=dataset.num_items, max_seq_len=dataset.max_seq_len)

    # loss
    w_init = tf.random_normal_initializer(stddev=0.01, mean=0)
    sample_softmax_W = tf.Variable(w_init(shape=(dataset.num_items, embedding_dim), dtype="float32"), trainable=True, name="W")
    b_init = tf.zeros_initializer()
    sample_softmax_bias = tf.Variable(b_init(shape=(dataset.num_items,), dtype="float32"), trainable=True, name="bias")


    def custom_sample_softmax(y_true, y_pred):
        loss = tf.nn.sampled_softmax_loss(weights=sample_softmax_W,
                                          biases=sample_softmax_bias,
                                          labels=y_true,
                                          inputs=y_pred,
                                          num_sampled=5,
                                          num_classes=dataset.num_items,
                                          num_true=1)
        final_loss = tf.reduce_mean(loss)
        return final_loss
    mind_model.compile(optimizer=optimizers.Adam(), loss=custom_sample_softmax)
    # we use the behavior_item_list to replace the behavior_brands_list and behavior_categories_list,
    # use target_item_list to repalce target_brands_list and target_categories_list,
    # use seq_len repalce user_profile, because these data has no effect in the model, we only feed some simulate
    # data for the model to go on
    mind_model.fit(
        (dataset.train_sample_items, dataset.train_sample_items, dataset.train_sample_items,
         dataset.target_item, dataset.target_item, dataset.target_item, dataset.sample_seq_len, dataset.sample_seq_len),
         dataset.target_item, epochs=5)
