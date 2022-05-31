import torch
import torch.nn as nn
import random
import csv
import math

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 1)
    
    def forward(self, vector):
        out = nn.functional.tanh(self.fc1(vector))
        out = self.fc2(out)
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
            ty = torch.vstack((ty, data[count+1][1]))
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
        train_data.append([[int(line[3]), int(line[6]), float(line[7]), float(line[8])], [float(line[0])]])
random.shuffle(train_data)

for elem in train_data:
    elem[0] = torch.Tensor(elem[0])
    elem[1] = torch.Tensor(elem[1])


batch_train = BatchDataset(train_data) 

model= MLP().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.MSELoss().cuda()
epoch = 10000

# training
for round in range(epoch):
    for i, data in enumerate(batch_train):
        
        optimizer.zero_grad()
        inputs, latencies = data
        inputs = torch.autograd.Variable(inputs).cuda()
        latencies = torch.autograd.Variable(latencies).cuda()
        
        outputs = model(inputs)
        loss = loss_func(outputs, latencies)
        loss.backward()
        
        optimizer.step()
        if i % 100 == 0:
            print(f'loss : {loss}')
    print(f'=== finished epoch {round} ===')

torch.save(model, 'mlp_pred.pkl')
