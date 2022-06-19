
from decision import Decision
from load_mock_data import load_mock_data
import time 

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, detector_num=3, img_num=800, mem=1024, config=None, strategy='infer_time',
                c2wQueues=None, w2cQueues=None, imgQueues=None):
        """ 
        two-way communiaction to prevent message contention
        """
        self.weights = []
        self.prices = []
        self.batches = []
        self.bs = None
        self.deadlines = []
        self.memory = mem  # memory contraint
        self.detector_num = detector_num
        self.act_ids = [] 
        self.wait_ids = [] 
        self.candidate = []
        self.img_num = img_num
        self.c2wQueues = c2wQueues
        self.w2cQueues = w2cQueues
        self.imgQueues = imgQueues
        self.cmds = []  # each element is bool list which indicate the models to be executed at the moment
        self.cursor = 0 # add cursor for cmds
        self.task_num = 0 # record the number of tasks at the moment
        self.total_requests = 0
        self.config = config  # list, each element is dictionary
        self.strat = strategy
        self.ddls = None # single round deadlines for each worker
        self.qos_violation_count = 0
        self.avg_violation_ratio = []
        self.completion_time = None # recording the completion time for each worker
        
    def initialization(self):
        for q in range(len(self.imgQueues)):
            for i in range(self.img_num):
                # Too large data will block process 
                # q.put(np.zeros(shape=(1920, 1080, 3)))
                self.imgQueues[q].put(i+1)

        data, col = load_mock_data()
        for i in range(col):
            self.cmds.append(list(data[:, i, :][:, 0]))
            self.deadlines.append(list(data[:, i, :][:, 1]))

        for x in self.cmds:
            self.total_requests += sum(x)

    # select worker from waiting list 
    def addWorker(self, memory):
        price = 0
        selected_w_id = -1
        for w in self.wait_ids:
            if self.weights[w] <= memory:
                if self.prices[w] > price:
                    price = self.prices[w]
                    selected_w_id = w
        
        return selected_w_id
    
    def remainingMemory(self):
        total = 0
        for i in self.act_ids:
            total += self.weights[i]
        
        return self.memory - total
    
    def compute_qos_violation(self):
        if self.completion_time is not None:
            for real, expected, bsize in zip(self.completion_time, self.ddls, self.bs):
                if real > expected:
                    self.qos_violation_count += bsize
                    vio_ratio = [(real - expected) / expected * 100 for _ in range(bsize)]
                    self.avg_violation_ratio.extend(vio_ratio)
            
    def update_cmd_queue(self):
        if self.task_num == 0:
            print(f'\033[1;34m [ROUND] : {self.cursor}\033[0m')
            self.compute_qos_violation()
            cmd = self.cmds[self.cursor]
            ddls = self.deadlines[self.cursor]
            self.batches.clear()
            self.ddls = [0.0 for _ in range(self.detector_num)]
            self.candidate.clear()
            self.completion_time = [0.0 for _ in range(self.detector_num)]
            self.bs = [0 for _ in range(self.detector_num)]
            
            # add 0 in the front of the list for dynamic programming
            wts = [0] 
            prc = [0]
            for i, (batch, ddl) in enumerate(zip(cmd, ddls)):
                if batch:
                    self.candidate.append(i)
                    self.batches.append(batch)
                    self.bs.append(batch)
                    self.weights.append(self.config[i][batch][0])
                    self.ddls[i] = sum(ddl)
                    self.bs[i] = batch
                    # srt = (sum(ddl) - self.config[i][batch][1]['infer_time']) / batch
                    # self.prices.append(100/srt)   # price must be positive
                    srt = (sum(ddl) - self.config[i][batch][1]['infer_time']) / batch
                    self.prices.append(self.config[i][batch][1]['infer_time'])
                    self.completion_time[i] = time.time()
                    wts.append(self.config[i][batch][0])
                    # prc.append(100/srt)
                    prc.append(self.config[i][batch][1]['infer_time'])
                else:
                    # when detector i is not selected , we still add one element in the list 
                    # to ensure the consistency of indexing
                    self.weights.append(self.config[i][1][0])
                    self.prices.append(self.config[i][1][1]['infer_time'])
            
            self.task_num = len(self.candidate)
            self.cursor += 1

            det = Decision(weights=wts, prices=prc, number=len(self.candidate), capacity=self.memory)
            select = det.decision()
            self.act_ids = [self.candidate[x - 1] for x in select]

            assert all(x in self.candidate for x in self.act_ids)
            self.wait_ids = [x for x in self.candidate if x not in self.act_ids]

            # To notify worker to load model into GPU memory
            for i in self.act_ids:
                self.c2wQueues[i].put('to_load')
                self.c2wQueues[i].put('to_infer')
                ic = self.candidate.index(i)    
                self.c2wQueues[i].put(self.batches[ic])
        
        popLists = []
        for j in self.act_ids:
            if self.w2cQueues[j].empty() is False:
                cmd = self.w2cQueues[j].get()
                if cmd == 'batch_done':
                    self.task_num -= 1
                    self.completion_time[j] = 1000 * (time.time() - self.completion_time[j])

                    popLists.append(j)

        self.popWorkers(popLists)
        popLists.clear()

        reMem = self.remainingMemory()
        if reMem > 0:
            while len(self.wait_ids) > 0:
                ind = self.addWorker(reMem)
                if ind < 0:
                    break
                else:
                    idx = self.wait_ids.index(ind)
                    ic = self.candidate.index(ind) 
                    self.c2wQueues[ind].put('to_load')
                    self.c2wQueues[ind].put('to_infer')
                    self.c2wQueues[ind].put(self.batches[ic])  
                    self.act_ids.append(ind)
                    print(f'add new worker {ind} success !!!')
                    reMem -= self.weights[ind]

                    # Must pop it in time to avoid dead lock
                    self.wait_ids.pop(idx)
        
        return

    def popWorkers(self, lists):
        for e in lists:
            ind = self.act_ids.index(e)
            self.act_ids.pop(ind)

    def run(self):
        self.initialization()
        while True:
            self.update_cmd_queue()
            if self.cursor >= len(self.cmds) and len(self.act_ids) + len(self.wait_ids) == 0:
                break

        for i in range(self.detector_num):
            self.c2wQueues[i].put('exit')
        
        print(f"All workers has finished jobs, processing {self.total_requests} images in total!")
        print(f'Total number of overtime task is : {self.qos_violation_count}')
        if self.qos_violation_count > 0:
            print(f'Average QoS violation ratio is : {sum(self.avg_violation_ratio)/self.qos_violation_count}')
        return