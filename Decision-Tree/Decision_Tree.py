#!/usr/bin/env python3
"""
@author: Pulkit Maloo
Date: January 26, 2018

Decision_Tree.py
"""

import pandas as pd
from math import log2
import pprint


class CreateNode(object):
    def __init__(self):
        self.leftChild = None
        self.rightChild = None

    def set_split_info(self, best_split_info):
        self.col = best_split_info["col"]
        self.left_rows = best_split_info["left"]
        self.right_rows = best_split_info["right"]
        self.threshold = best_split_info["threshold"]
        self.info_gain = best_split_info["info_gain"]
        self.impurity_child = best_split_info["impurity_child"]

    def set_label(self, label):
        self.label = label

    def __str__(self):
        res = "Node("
        if hasattr(self, "label"):
            res += "label: " + str(self.label)
        else:
            res += "col: " + str(self.col)
            res += ", threshold: " + str(self.threshold)
            res += ", left: " + str(len(self.left_rows))
            res += ", right: " + str(len(self.right_rows))
            res += ", info_gain: " + str(self.info_gain)
        res += ")"
        return res

    def __repr__(self):
        if hasattr(self, "label"):
            return "label: " + str(self.label)
        else:
            return "Feature " + str(self.col) + " >= " + \
                    str(round(self.threshold, 3))


def load_data(fname="test_data.csv"):
    """
    make sure class variable is the last column
    """
    df = pd.read_csv(fname, header=None)
    df.columns = range(len(df.columns))
    df.iloc[:, -1] = df.iloc[:, -1].astype("category").cat.codes
    return df


def Gini(df, rows):
    """
    col_num: feature, class_col
    """
    class_col = list(df.columns)[-1]
    counts = df.iloc[rows, [class_col]].groupby([class_col]).size()
    return 1 - (sum(counts**2) / (len(rows)**2))


def Entropy(df, rows):
    class_col = list(df.columns)[-1]
    counts = df.iloc[rows, [class_col]].groupby([class_col]).size() / len(rows)
    return - sum(counts * counts.apply(lambda x: log2(x) if x > 0 else 0))


def find_best_split(df, rows, cols, I=Entropy):
    """ I: Impurity measure """
    best_split_info = {"info_gain": 0}

    for col in cols:
        df_sorted = df.iloc[rows, [col]+[-1]].sort_values([col])

        threshold = []

        for i in range(len(df_sorted.index)-1):
            idx = df_sorted.index[i]
            idx_n = df_sorted.index[i+1]
            x = df_sorted.loc[idx].iloc[-1]
            y = df_sorted.loc[idx_n].iloc[-1]
            if x - y != 0:
                thres = (df_sorted.loc[idx, col] + df_sorted.loc[idx_n, col])/2
                threshold += [thres]

        best_gain = best_split_info["info_gain"]

        for t in threshold:
            rows_left = df_sorted[df_sorted.loc[:, col] < t].index.values
            rows_right = df_sorted[df_sorted.loc[:, col] >= t].index.values
            left_I, right_I = I(df, rows_left), I(df, rows_right)
            val = (len(rows_left)*left_I + len(rows_right)*right_I) / len(rows)
            gain = I(df, rows) - val

            if gain > best_gain:
                best_gain = gain
                best_split_info["info_gain"] = gain
                best_split_info["impurity_child"] = val
                best_split_info["left"] = rows_left
                best_split_info["right"] = rows_right
                best_split_info["threshold"] = t
                best_split_info["col"] = col

    return best_split_info


def stopping_cond(df, rows, cols):
    class_col = list(df.columns)[-1]
    # no features remaining
    if len(cols) <= 0:     return True
    # if less than n rows
    elif len(rows) <= 3:   return True
    # all records have same class label
    elif len(set(df.iloc[rows, class_col])) <= 1:
        return True
    elif find_best_split(df, rows, cols)["info_gain"] < 10E-5:
        return True
    # all featueres have same value
    elif sum(df.iloc[rows, list(df.columns[:-1])].duplicated()) >= len(rows):
        return True
    return False


def Classify(df, rows):
    class_col = list(df.columns)[-1]
    counts = df.iloc[rows, [class_col]].groupby([class_col]).size()
    return counts.idxmax()


def TreeGrowth(df, rows, cols, I=Entropy):
    if stopping_cond(df, rows, cols):
        leaf = CreateNode()
        label = Classify(df, rows)
        leaf.set_label(label)
        return leaf
    else:
        root = CreateNode()
        best_split_info = find_best_split(df, rows, cols, I)
        root.set_split_info(best_split_info)

#        cols.remove(best_split_info["col"])    # remove feature
        root.leftChild = TreeGrowth(df, root.left_rows, cols, I)
        root.rightChild = TreeGrowth(df, root.right_rows, cols, I)

    return root


def test_instance(instance, root):
    """instance: list of features"""
    if hasattr(root, "label"):
        return root.label
    elif instance[root.col] >= root.threshold:
        return test_instance(instance, root.rightChild)
    else:
        return test_instance(instance, root.leftChild)


def Test(df, root):
    df['pred'] = df.apply(lambda x: test_instance(list(x)[:-1], root), axis=1)
    return sum(df.iloc[:, -2] == df.iloc[:, -1]) / df.shape[0]


def train_test_split(df, split=0.5):
    df_train = df.sample(frac=split)
    df_test = df.drop(df_train.index)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def build_model(df):
    df_train, df_test = train_test_split(df)
    rows, cols = list(df_train.index), list(df_train.columns)[:-1]
    root = TreeGrowth(df_train, rows, cols)
    acc = Test(df_test, root)
    return (root, acc, df_train, df_test)


def printree(root):
    if hasattr(root, "label"):
        return (root, )
    else:
        return [root, (printree(root.rightChild), "else",
                       printree(root.leftChild))]


def print_tree(root):
    pp = pprint.PrettyPrinter(indent=1)
    print("\n" + ">"*30, "\n    Sample Decision Tree\n")
    pp.pprint(printree(root))
    print(">"*30)


def main(fname="bezdekIris.csv"):
    df = load_data(fname)
    models = []
    acc = 0

    for i in range(5):
        models.append(build_model(df))
        acc += models[i][1] * 100

    print("-"*30, "\n  Average Accuracy =", round(acc/5, 2), "%\n" + "-"*30)
    return models


if __name__ == '__main__':
    models = main()
    root, _, _, _ = max(models, key=lambda x: x[1])
    print_tree(root)
