#! /bin/bash/env python
import pandas as pd


def load_car_data():
    df = pd.read_csv("car.data", header=None)
    return pd.get_dummies(df)


def load_test_data():
    data = pd.DataFrame([['bread', 'milk'],
                         ['bread', 'diaper', 'beer', 'egg'],
                         ['milk', 'diaper', 'beer', 'cola'],
                         ['bread', 'milk', 'diaper', 'beer'],
                         ['bread', 'milk', 'diaper', 'cola']])
    return pd.get_dummies(data)


def load_nursery_data():
    df = pd.read_csv("nursery.data", header=None)
    return pd.get_dummies(df)


def load_chess_data():
    df = pd.read_csv("krkopt.data", header=None)
    df.loc[:, [1,3,5]] = df.loc[:, [1,3,5]].astype(str)
    return pd.get_dummies(df)


def load_nursery_data():
    df = pd.read_csv("cmc.data", header=None)
    return pd.get_dummies(df)


def load_data(data="test"):
    if data == "test":
        return load_test_data()
    elif data == "car":
        return load_car_data()
    elif data == "nursery":
        return load_nursery_data()
    elif data == "chess":
        return load_chess_data()
    else:
        print("Dataset not found")


if __name__ == "__main__":
    for data in ["car", "nursery", "chess"]:
        print(load_data(data).head())

