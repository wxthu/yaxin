
from decision import Decision
from load_mock_data import load_mock_data
import time 
import pickle

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, detector_num=6, img_num=800, mem=1024, config=None,
                c2wQueues=None, w2cQueues=None, imgQueues=None):
        """ 
        two-way communiaction to prevent message contention
        """
        self.lCursor = 0
        self.tCursor = 0 # These two flags indicate whether current tasks finished loading or inference
        self.load_state = [False for _ in range(detector_num)]
        self.infer_state = [False for _ in range(detector_num)]
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
   
    
    def compute_qos_violation(self):
        if self.completion_time is not None:
            for real, expected, bsize in zip(self.completion_time, self.ddls, self.bs):
                if real > expected:
                    self.qos_violation_count += bsize
                    vio_ratio = [(real - expected) / expected * 100 for _ in range(bsize)]
                    self.avg_violation_ratio.extend(vio_ratio)
    
    def compute_candidates(self, wts, prc, number, cap):
        det = Decision(wts, prc, number, cap)
        select = det.decision()
        return [self.candidate[x-1] for x in select]
    
    def notify_to_load(self, workers):
        for w in workers:
            if not self.load_state[w]:
                self.c2wQueues[w].put(pickle.dumps(['to_load', self.bs[w]]))
                self.lCursor += 1
    
    def notify_to_infer(self, workers):
        for w in workers:
            if not self.infer_state[w] and self.load_state[w]:
                self.c2wQueues[w].put(pickle.dumps(['to_infer']))
                self.infer_state[w] = True
                self.tCursor += 1
        
            
    def update_cmd_queue(self):
        if self.task_num == 0:
            print(f'\033[1;34m [ROUND] : {self.cursor}\033[0m')
            self.compute_qos_violation()
            cmd = self.cmds[self.cursor]
            ddls = self.deadlines[self.cursor]
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
                    self.ddls[i] = sum(ddl)
                    self.bs[i] = batch
                    self.completion_time[i] = time.time()
                    wts.append(self.config[i][batch][0])
                    prc.append(self.config[i][batch][1]['infer_time'])
            
            self.task_num = len(self.candidate)
            self.cursor += 1

            self.act_ids = self.compute_candidates(wts, prc, len(self.candidate), self.memory)
            
            assert all(x in self.candidate for x in self.act_ids)
            
            self.wait_ids = [x for x in self.candidate if x not in self.act_ids]
            self.notify_to_load(self.act_ids)
        
        # blocking mode 
        for ind in self.act_ids:
            signal = pickle.loads(self.w2cQueues[ind].get())
            if signal[0] == 'load_over':
                self.lCursor -= 1
                self.load_state[ind] = True
            elif signal[0] == 'task_over':
                self.tCursor -= 1
                self.task_num -= 1
                self.completion_time[ind] = 1000 * (signal[1] - self.completion_time[ind])
            else:
                assert False, '===== unknown command ====='
                    
        if self.lCursor <= 0:
            self.notify_to_infer(self.act_ids)
            
        # TODO complete logic of updating wait_ids iteratively, here just for simplicity
        if self.tCursor <= 0:
            for ind in self.act_ids:
                self.infer_state[ind] = False
                self.load_state[ind] = False
                
            self.act_ids = []
            if len(self.wait_ids) > 0:
                self.notify_to_load(self.wait_ids)
                self.act_ids = self.wait_ids
                self.wait_ids = []
            
        return

    def run(self):
        self.initialization()
        while True:
            self.update_cmd_queue()
            if self.cursor >= len(self.cmds) and len(self.act_ids) + len(self.wait_ids) == 0:
                break

        for i in range(self.detector_num):
            self.c2wQueues[i].put(pickle.dumps(['exit']))
        
        print(f"All workers has finished jobs, processing {self.total_requests} images in total!")
        print(f'Total number of overtime task is : {self.qos_violation_count}')
        if self.qos_violation_count > 0:
            print(f'Average QoS violation ratio is : {sum(self.avg_violation_ratio)/self.qos_violation_count}')
        return