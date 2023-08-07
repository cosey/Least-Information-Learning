import numpy as np
from numpy.core.numeric import array_equal

def compute_one_others(order,arr):
    res=[]
    for i in order:
        res.append(sum([arr[i][index] for index in order if i != index ]))
    return res

def min_except_arr(arr1,arr2):
    min_index=None
    min_num=1000
    for index,num in enumerate(arr2):
        if num<min_num and index not in arr1:
            min_index=index
            min_num=num
    return min_index

def init_order(arr,l):
    arr_data=[]
    if len(arr)==0:
        return False
    if l<=0 or l>len(arr[0]):
        return False
    order=[0]
    arr_data.append(arr[0][0])
    for i in range(1,l):
        current_num=order[-1]
        mid=min_except_arr(order,arr[current_num])
        order.append(mid)
        arr_data.append(arr[current_num][mid])
    return order

def init_sequence(bb,l):
    order=init_order(bb,l)
    while True:
        # print(order)
        data=compute_one_others(order,bb)
        max_data=max(data)
        max_data_index=data.index(max_data)
        try_min=max_data
        try_min_index=max_data_index
        for other_one in order:
            if other_one not in order:
                try_order=order
                try_order.pop(max_data_index)
                try_order.append(other_one)
                current_d=compute_one_others(try_order,bb)[-1]
                if current_d<try_min:
                    try_min=current_d
                    try_min_index=other_one
        if try_min_index==max_data_index:
            break
        order.pop(max_data_index)
        order.append(try_min_index)
    return order
