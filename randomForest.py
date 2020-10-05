import pandas as pd
import numpy as np
from itertools import repeat
from random import randrange
import math

df = pd.read_csv('interviewee.csv')
banks_training = pd.read_csv('banks.csv')
banks_testing = pd.read_csv('banks-test.csv')

def entropy(arr):
    unique, counts = np.unique(arr, return_counts=True)
    category_probs = np.array(counts/arr.size)
    plogp = category_probs * np.log2(category_probs)
    return np.sum(plogp) * -1

def information_gain(df, attribute):
    labels = np.array(df['label'])
    entropy_total = entropy(labels)
    value, counts = np.unique(np.array(df[attribute]), return_counts=True)
    attr_probs = dict(zip(value, counts/labels.size))
    partition_sum = 0
    for key, value in attr_probs.items():
            new_df = df[(df[attribute] == key)]
            new_label = np.array(new_df['label'])
            partition_sum += value * entropy(new_label)
    return entropy_total - partition_sum

class Tree(object):
    def __init__(self):
        self.left = None
        self.child = []
        self.data = ''
    def createChildren(self,partitions):
        for partition in partitions:
            self.child.append({partition: Tree()})
    def setValue(self,attr):
        self.data = attr

def decision_tree(root, df, attribute_list):
    unique_labels, counts = np.unique(np.array(df['label']), return_counts=True)
    if (unique_labels.size == 1) or (not attribute_list):
        #termination_condtion
        unique_label_index = np.argmax(counts)
        root.setValue({'label': unique_labels[unique_label_index]})
        return
    else:
        ig = list(map(information_gain, repeat(df), attribute_list))
        max_ig_index = ig.index(max(ig))
        split_attr_name = attribute_list[max_ig_index]
        partitions = np.unique(np.array(df[split_attr_name]))
        root.createChildren(partitions)
        root.setValue(split_attr_name)
        for idx,partition in enumerate(partitions):
            partition_df = df[df[split_attr_name] == partition]
            decision_tree(root.child[idx][partition], partition_df, [x for x in attribute_list if x != split_attr_name])
def predict(root, obj):
    while root:
        attr = root.data
        value = obj[attr]
        for child in root.child:
            child_node = child.get(value)
            if child_node:
                if type(child_node.data) is dict and (child_node.data.get('label') or child_node.data.get('label') == False):
                    return child_node.data.get('label')
                else:
                    root = child_node
                    break


def random_sampling(attributes, number):
    rand_attr = []
    temp_length = len(attributes)
    temp_attributes = attributes
    for i in range(0, number):
        rand_index = randrange(temp_length)
        rand_attr.append(temp_attributes[rand_index])
        temp_attributes = [x for x in temp_attributes if x != temp_attributes[rand_index]]
        temp_length -= 1
    return rand_attr


def random_forest(df, numberOfTrees, percentageOfAttributes):
    trees = []
    columns = df.columns.tolist()
    columns.remove('label')
    number_of_attributes = math.floor((percentageOfAttributes/100) * len(columns))
    for i in range(0, numberOfTrees):
        rand_attr_list = random_sampling(columns, number_of_attributes)
        root = Tree()
        decision_tree(root, df, rand_attr_list)
        trees.append(root)
    return trees


def TrainAndTestRandomForest(trainingdata, numberOfTrees, percentageOfAttributes, testdata):
    trees = random_forest(trainingdata, numberOfTrees, percentageOfAttributes)

    total_count = 0
    correct_prediction = 0
    for row in banks_testing.to_dict(orient="records"):
        print('testing')
        votes = np.array([])
        real_value = row['label']
        for tree in trees:
            votes = np.append(votes, predict(tree, row))
        unique_votes, votes_count = np.unique(votes, return_counts=True)
        max_vote_index = np.argmax(votes_count)
        predicted = unique_votes[max_vote_index]
        print(predicted)
        if predicted == real_value:
            correct_prediction += 1
        total_count += 1
    accuracy = correct_prediction/total_count
    print(accuracy)

TrainAndTestRandomForest(banks_training, 7, 70, banks_testing)