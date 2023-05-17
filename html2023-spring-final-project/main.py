import numpy as np
import csv
from liblinear.liblinearutil import *

with open('train.csv', newline='', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    data_list_t = list(reader)

data_list = np.array(data_list_t).T.tolist()

# print(len(data_list[17]))
# print(data_list[17][len(data_list[17])-1])

prob  = problem(data_list[0], data_list[1])
param = parameter('-s 11 -c 4 -B 1')
m = train(prob, param)
