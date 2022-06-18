import torch
import random
import pickle
import csv

from threading import Thread

class Controller:
    def __init__(self, models, c2wQueues, w2cQueues, sample_times) -> None:
        self.models = list(models.values())
        self.c2wQueues = c2wQueues
        self.w2cQueues = w2cQueues
        self.names = list(models.keys())
        self.sample_times = sample_times
        
        self.module_num = list()
        self.module_count = 0
        self.handler = torch.nn.modules.module.register_module_forward_pre_hook(self.forward_pre_hook())
        
        self.parse_topology()
        
    def parse_topology(self):
        fake_img = torch.zeros(1, 3, 224, 224)
        for m in self.models:
            m(fake_img)
            self.module_num.append(self.module_count)
            self.module_count = 0
            
        self.handler.remove()
    
    def forward_pre_hook(self):
        def hook(module, inputs):
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                self.module_count += 1
        return hook
    
    def gen_samples(self):
        
        print('To generate samples !!')
        number = len(self.models)
        for full_model_id in range(number):
            for group_model_id in range(number):
                if group_model_id == full_model_id:
                    continue
            
                layer_ids = set()
                layer_num = self.module_num[group_model_id]
                for i in range(self.sample_times):
                    interval = random.choices(range(1, layer_num), k=2)
                    if interval[0] == interval[1]:
                        continue
                    interval.sort()
                    layer_ids.add(tuple(interval))
                    
                for x in layer_ids:
                    while self.c2wQueues[0].full():
                        pass
                    
                    self.c2wQueues[0].put(pickle.dumps(self.names[full_model_id]))
                    x = list(x)
                    x.append(self.names[group_model_id])
                    self.c2wQueues[1].put(pickle.dumps(x))
                  
        for i in range(2):
            self.c2wQueues[i].put(pickle.dumps('exit'))
    
    def post_process(self):
        csvfile = open('sampled_data.csv', 'a')
        datawriter = csv.writer(csvfile, dialect='excel')
        datawriter.writerow(['inference', 'model_1', 'op_range', 'batchsize', 'model_2', 'op_range', 'batchsize'])
        while True:
            # auto blocking mode
            cmd0 = pickle.loads(self.w2cQueues[0].get())
            cmd1 = pickle.loads(self.w2cQueues[1].get())
            # print(' ** Get profiling results successfully !')
            
            if cmd0 == 'finish' and cmd1 == 'finish':
                break
            
            output = [0.5 * (cmd0[0] + cmd1[0]) * 1000]
            output.extend(cmd0[1:])
            output.extend(cmd1[1:])
            datawriter.writerow(output)
            print(output)
        
    def run(self):
        td1 = Thread(target=self.gen_samples)
        td2 = Thread(target=self.post_process)
        
        print('sub-threads in controller begin ...')
        td1.start()
        td2.start()
        
        td1.join()
        td2.join()
        
        print('controller threads exited ...')
