# -*- coding: utf8 -*-
import argparse
import numpy as np
from time import time

from tensorflow.keras.optimizers import Adagrad, RMSprop, SGD, Adam

from Dataset import Dataset
from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import Embedding, Dense, Flatten, Multiply
from tensorflow.keras.regularizers import l2
from evaluate import evaluate_model


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

    def __init__(self, num_users, num_items, latent_dim, reg_layers=[0, 0]):
        super(NCFModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.reg_layers = reg_layers
        self.MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                           embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                           embeddings_regularizer=l2(reg_layers[0]),
                                           input_length=1)
        self.MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                          embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                          embeddings_regularizer=l2(reg_layers[1]),
                                          input_length=1)
        self.flatten = Flatten()
        self.multiply = Multiply()
        self.prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_uniform, name='prediction')

    def call(self, inputs):
        user_embedding = self.MF_Embedding_User(inputs[0])
        item_embedding = self.MF_Embedding_Item(inputs[1])
        print(user_embedding.shape)
        print(item_embedding.shape)
        user_embedding = self.flatten(user_embedding)
        item_embedding = self.flatten(item_embedding)
        all_embedding = self.multiply([user_embedding, item_embedding])
        print(user_embedding.shape)
        print(item_embedding.shape)
        print(all_embedding.shape)
        score = self.prediction(all_embedding)
        print(score.shape)
        return score


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
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
    path = args.path
    dataset = args.dataset
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    # path = "./data/"
    # dataset = "ml-1m"
    # layers = [64, 32, 16, 8]
    # reg_layers = [0, 0, 0, 0]
    # num_negatives = 4
    # learner = "adam"
    # learning_rate = 0.01
    # batch_size = 256
    # epochs = 20
    # verbose = 1
    # args.out = 0

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % args)
    model_out_file = 'pretrain/%s_MLP_%s_%d.h5' % (args.dataset, num_factors, time())

    # Loading data
    t1 = time()
    dataset = Dataset(path + dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    ncf_model = NCFModel(num_users, num_items, num_factors, regs)
    if learner.lower() == "adagrad":
        ncf_model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    elif learner.lower() == "rmsprop":
        ncf_model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    elif learner.lower() == "adam":
        ncf_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        ncf_model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

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
        print("The best MLP model is saved to %s" % model_out_file)

