import torch
import torch.nn as nn
import random
import csv
import math

# We train a discrete latency predictor ,so it can be regard as a classifier
# by 10ms interval and assume latency will not exceed 1s
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 100)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, vector):
        out = nn.functional.tanh(self.fc1(vector))
        out = nn.functional.tanh(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        return out


def BatchDataset(data, batchsize=100):

    real_data = []
    group = math.ceil(len(data)/batchsize)
    
    count = 0
    for i in range(group):
        
        tx = data[count][0]
        ty = data[count][1]
        real_bs = len(data) - count if len(data) - count < batchsize else batchsize
        
        for _ in range(1, real_bs):
            tx = torch.vstack((tx, data[count+1][0]))
            ty = torch.hstack((ty, data[count+1][1]))
            count += 1
            
        real_data.append([tx, ty])
        count += 1
    
    print('Batched dataset contructed !!!')
    return real_data

 
train_data = []
with open('train_datasets.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    headline = next(reader)
    for line in reader:
        
        real_val = float(line[0])
        label=int(real_val/10)
        train_data.append([[int(line[3]), int(line[6]), float(line[7]), float(line[8])], [label]])
        
random.shuffle(train_data)

for elem in train_data:
    elem[0] = torch.Tensor(elem[0])
    elem[1] = torch.Tensor(elem[1]).type(torch.long)


batch_train = BatchDataset(train_data) 

model= MLP().cuda()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss().cuda()
epoch = 10000

# training
for round in range(epoch):
    for i, data in enumerate(batch_train):
        
        optimizer.zero_grad()
        inputs, latencies = data
        inputs = inputs.cuda()
        latencies = latencies.cuda()
        
        outputs = model(inputs)
        loss = loss_func(outputs, latencies)
        loss.backward()
        
        optimizer.step()
        if i % 100 == 0:
            print(f'loss : {loss}')
    print(f'=== finished epoch {round} ===')

torch.save(model, 'mlp_pred.pkl')
