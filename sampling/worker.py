import torch
import pickle
import time

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
  
class Worker:
    def __init__(self, id, models, c2wQueue, w2cQueue, barrier = None) -> None:
        self.id = id
        self.models = models
        self.c2wQueue= c2wQueue
        self.w2cQueue = w2cQueue
        self.timestamp = []
        self.start_op = 0
        self.stop_op = 0
        self.cursor = 0
        self.count = 0  # just for sleeping to cold down the GPU
        self.cold_down_round = 1000
        self.barrier = barrier
        self.batch_pairs = [[1,1], [1,2], [1,3], [1,4], [2,1], [2,2], [2,3], [2,4],
                            [3,1], [3,2], [3,3], [3,4], [4,1], [4,2], [4,3], [4,4]]
                
        for k, v in self.models.items():
            self.models[k].eval().cuda()
                    
    def warm_up(self):
        img = torch.zeros(1,3,224,224).cuda()
        for _ in range(5):
            for _, model in self.models.items():
                model(img)
        return
    
    def forward_pre_hook(self):
        def group_hook(module, inputs):
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                self.cursor += 1
                if self.cursor == self.start_op or self.cursor == self.stop_op:
                    moment = time_sync()
                    self.barrier.wait()
                    self.timestamp.append(time_sync())
                    
        return group_hook
    
    @torch.no_grad()             
    def run(self):
        self.warm_up()
        while True:
            if not self.c2wQueue.empty():
                
                command = pickle.loads(self.c2wQueue.get())
                if command == 'exit':
                    print(f'worker {self.id} profiling finished !!!')
                    break
                
                for bs in self.batch_pairs:
                    
                    if self.id == 0:
                        img = torch.zeros(bs[0], 3, 384, 640).cuda()
                        self.barrier.wait()
                        self.timestamp.append(time_sync())
                        self.models[command](img)
                        moment = time_sync()
                        self.barrier.wait()
                        self.timestamp.append(time_sync())
                        msg = [self.timestamp[1] - self.timestamp[0], command, 'full', bs[0]]
                    else:
                        img = torch.zeros(bs[1], 3, 384, 640).cuda()
                        self.start_op = command[0]
                        self.stop_op = command[1]
                        handle = torch.nn.modules.module.register_module_forward_pre_hook(self.forward_pre_hook())
                        self.models[command[2]](img)
                        
                        handle.remove()
                        self.cursor = 0
                        msg = [self.timestamp[1] - self.timestamp[0], command[2], [self.start_op, self.stop_op], bs[1]]
                    
                    torch.cuda.empty_cache() 
                    self.w2cQueue.put(pickle.dumps(msg))
                    self.timestamp.clear()
                    
                    self.count += 1
                    if self.count % self.cold_down_round == 0:
                        print('### sleep for some time to cold down ###')
                        time.sleep(100)
        
        self.w2cQueue.put(pickle.dumps('finish'))
                        