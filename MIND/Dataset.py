'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
from collections import defaultdict


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, train_sample=8):
        '''
        Constructor
        '''
        self.train_sample = train_sample
        self.num_items = 0
        self.num_users = 0
        self.max_seq_len = 0
        self.train_sample_items = []
        self.target_item = []
        self.sample_seq_len = []
        self.load_rating_file_as_matrix(path + ".train.rating")
        # self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        # self.testNegatives = self.load_negative_file(path + ".test.negative")
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items, max_seq_len = 0, 0, 0
        behavior_dict = defaultdict(list)
        temp_behavior_list = []
        temp_user = 0
        with open(filename, "r") as f:
            for line in f:
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                if temp_user != u and temp_user >= 0:
                    temp_behavior_list.reverse()
                    behavior_dict[temp_user] = temp_behavior_list
                    max_seq_len = max(len(temp_behavior_list), max_seq_len)
                    temp_behavior_list = [i]
                    temp_user = u
                else:
                    temp_behavior_list.append(i)
        behavior_dict[temp_user] = temp_behavior_list
        max_seq_len = max(len(temp_behavior_list), max_seq_len)
        self.num_items = num_items
        self.num_users = num_users
        max_seq_len = min(max_seq_len, 100)
        self.max_seq_len = max_seq_len
        # Construct train sample
        for key, value in behavior_dict.items():
            length = len(value)
            for i in range(self.train_sample):
                if length - i - 1 > self.max_seq_len:
                    self.sample_seq_len.append([self.max_seq_len])
                    self.target_item.append([value[length - 1 - i]])
                    self.train_sample_items.append(value[length - 1 - i - self.max_seq_len: length - 1 - i])
                else:
                    self.sample_seq_len.append([length-1-i])
                    self.target_item.append([value[length-1-i]])
                    self.train_sample_items.append(value[:length-1-i] + [0] * (max_seq_len-length+1+i))
        self.sample_seq_len = np.array(self.sample_seq_len)
        self.target_item = np.array(self.target_item)
        self.train_sample_items = np.array(self.train_sample_items)


