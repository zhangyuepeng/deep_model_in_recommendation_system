# -*- coding: utf8 -*-
import argparse
import numpy as np
from time import time

from tensorflow.keras.optimizers import Adagrad, RMSprop, SGD, Adam

from Dataset import Dataset
from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.regularizers import l2
from evaluate import evaluate_model
import tensorflow as tf


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


class NCFModel(Model):

    def __init__(self, num_users, num_items, layers_unit, reg_layers):
        super(NCFModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers_unit = layers
        self.reg_layers = reg_layers
        self.MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0]//2, name='user_embedding',
                                            embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                            embeddings_regularizer=l2(reg_layers[0]),
                                            input_length=1)
        self.MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0]//2, name='item_embedding',
                                            embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                            embeddings_regularizer=l2(reg_layers[0]),
                                            input_length=1)
        self.flatten = Flatten()
        self.concat = Concatenate(axis=1)
        self.dense_layers = []
        for idx in range(1, len(layers_unit)):
            self.dense_layers.append(Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu',
                                           name='layer%d' % idx))

        self.prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_uniform, name='prediction')

    def call(self, inputs):
        user_embedding = self.MLP_Embedding_User(inputs[0])
        item_embedding = self.MLP_Embedding_Item(inputs[1])
        print(user_embedding.shape)
        print(item_embedding.shape)
        user_embedding = self.flatten(user_embedding)
        item_embedding = self.flatten(item_embedding)
        all_embedding = self.concat([user_embedding, item_embedding])
        print(user_embedding.shape)
        print(item_embedding.shape)
        print(all_embedding.shape)
        for dense_layer in self.dense_layers:
            all_embedding = dense_layer(all_embedding)
            print(all_embedding.shape)
        score = self.prediction(all_embedding)
        print(score.shape)
        return score


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of"
                             " user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # path = args.path
    # dataset = args.dataset
    # layers = eval(args.layers)
    # reg_layers = eval(args.reg_layers)
    # num_negatives = args.num_neg
    # learner = args.learner
    # learning_rate = args.lr
    # batch_size = args.batch_size
    # epochs = args.epochs
    # verbose = args.verbose

    path = "./data/"
    dataset = "ml-1m"
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    num_negatives = 4
    learner = "adam"
    learning_rate = 0.01
    batch_size = 256
    epochs = 20
    verbose = 1
    args.out = 0



    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % args)
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(path + dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    ncf_model = NCFModel(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        ncf_model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    elif learner.lower() == "rmsprop":
        ncf_model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    elif learner.lower() == "adam":
        ncf_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        ncf_model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # ncf_model = NCFModel(1000, 1000, [64, 32, 16, 8], [0, 0, 0, 0])
    # a = [np.array([[1], [2], [3]]), np.array([[4], [5], [6]])]
    # a = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    # print(ncf_model(a))
    # print(ncf_model.summary())
    #
    # ncf_model.compile(optimizer='adam',
    #                   loss='binary_crossentropy',
    #                   metrics=['accuracy'])
    # ncf_model.fit(a, np.array([[0], [1], [0]]))


    # e = Embedding(input_dim=3, output_dim=16, name='user_embedding',
    #               embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
    #               input_length=2)
    # print(e(np.array([[2], [1]])))
    # print(Flatten()(e(np.array([2, 1]))))

        # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(ncf_model, testRatings, testNegatives, topK, evaluation_threads)
    print(ncf_model.summary())
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
        hist = ncf_model.fit([np.array(user_input), np.array(item_input)],  # input
                             np.array(labels),  # labels
                             batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(ncf_model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    ncf_model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))

