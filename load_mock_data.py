import numpy as np
import pandas as pd
import random

def load_mock_data(filename='mock_batched_request_rate.csv', freq_low=5, freq_high=10):
    random.seed(0)
    def get_random_latency(n):
        latency_list = [1000/i for i in range(freq_low, freq_high+1)]
        return [int(random.choice(latency_list)) for _ in range(n)]
    
    data = pd.read_csv(filename)
    data = data.to_numpy()
    data = data.tolist()
    data_with_timestamp = []
    
    for row in data:
        row_list = []
        for col in row:
            row_list.append([col, get_random_latency(col)])
        data_with_timestamp.append(row_list)
    
    data = np.array(data_with_timestamp)
    return data, data.shape[1]


if __name__ == "__main__":
    data, col = load_mock_data()
    print(f'type : {type(data)}')
    print(f'size : {col}')
    print((data[:,2,:][:, 0]))
