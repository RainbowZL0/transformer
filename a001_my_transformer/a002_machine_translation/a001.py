import pandas as pd
import json

json_file_path = "./train-v2.0.json"


def try_0():
    data = pd.read_json(json_file_path).to_dict()
    data = pd.json_normalize(data=data)
    data.to_excel("./test.xlsx", index=False)


def try_1():
    pass


if __name__ == '__main__':
    try_0()

