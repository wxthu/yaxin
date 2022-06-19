import random
import numpy as np
import pandas as pd

def gen_rand(img_num=200, detector_num=7):

    rates= {}
    batch_pool = [0, 1, 2, 3, 4]
    for i in range(img_num):
        rate = [random.choice(batch_pool) for _ in range(detector_num)]
        rates.update({str(i) : rate})

    df = pd.DataFrame(data=rates)
    df.to_csv('mock_batched_request_rate.csv', index=False)   

if __name__ == '__main__':
    gen_rand()
    df = pd.read_csv('mock_request_rate.csv')
    for col in df:
        print(df[col].values)
        print(sum(df[col].values))
        break
    
