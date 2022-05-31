import torch
import torch.nn as nn
import math 
import csv

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 1)
    
    def forward(self, vector):
        out = nn.functional.tanh(self.fc1(vector))
        out = self.fc2(out)
        return out
    
data = []
with open('test_datasets.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    headline = next(reader)
    for line in reader:
        data.append([[int(line[3]), int(line[6]), float(line[7]), float(line[8])], [float(line[0])]])

for i in range(len(data)):
    data[i][0] = torch.Tensor(data[i][0]).cuda()
    data[i][1] = torch.Tensor(data[i][1]).cuda()

model = torch.load('./mlp_pred.pkl', map_location=torch.device('cuda'))
error_distribution = [0 for _ in range(10)]
for x, y in data:
    y_ = model(x)
    pct = abs(y_ - y) / y * 100
    id = math.floor(pct / 10) 
    if id > 9:
        id = 9
        print(f'large error: {pct}')
    error_distribution[id] += 1

print(error_distribution)

 


