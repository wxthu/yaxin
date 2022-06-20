
from load_mock_data import load_mock_data
import pickle
import time

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, detector_num=6, img_num=800, config=None,
                c2wQueues=None, w2cQueues=None, imgQueues=None):
        """ 
        two-way communiaction to prevent message contention
        """
        self.detector_num = detector_num
        self.load_state = [False for _ in range(detector_num)]
        self.infer_state = [False for _ in range(detector_num)]
        self.lCursor = 0
        self.tCursor = 0  # These two flags record current inference state and next loading state
        self.act_id = -1
        self.next_id = -1
        self.candidate = []
        self.img_num = img_num
        self.c2wQueues = c2wQueues
        self.w2cQueues = w2cQueues
        self.imgQueues = imgQueues
        self.cmds = []  # batch number 
        self.deadlines = []
        self.round = 0 
        self.cursor = 0 
        self.task_num = 0 # record the number of tasks at the moment
        self.total_requests = 0
        self.config = config  # list, each element is dictionary
        self.qos_violation_count = 0
        self.avg_violation_ratio = []
                
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
            
        self.compute_candidates()
    
    def notify_to_load(self, w_id):
        if w_id >= 0 and not self.load_state[w_id]:
            command = ['loading', self.candidate[self.cursor][1]]
            self.c2wQueues[self.next_id].put(pickle.dumps(command))
            self.lCursor += 1
            
    def notify_to_infer(self, w_id):     
        if w_id >= 0 and self.load_state[w_id] and not self.infer_state[w_id]:
            self.c2wQueues[w_id].put(pickle.dumps(['inference']))
            self.infer_state[w_id] = True
            self.tCursor += 1
             
    def compute_candidates(self):
        '''
        This function is mainly responsible for computing candidates
        at the start if each round and preloading the first model to 
        infer
        '''
        if self.round >= len(self.cmds):
            return 
        
        cmd = self.cmds[self.round]
        ddls = self.deadlines[self.round]
        self.candidate.clear()
        
        for i, (batch, ddl) in enumerate(zip(cmd, ddls)):
            if batch:
                srt = sum(ddl) - self.config[i][batch][1]['infer_time']
                # The fourth element denote Job Completion Time for statistics purpose
                self.candidate.append([i, batch, srt, time.time(), sum(ddl)])
        
        self.candidate.sort(key=lambda obj: obj[2])
        if len(self.candidate) == 0:
            return
        
        self.act_id = -1
        self.next_id = self.candidate[0][0]
        self.task_num = len(self.candidate)
        
        self.notify_to_load(self.next_id)
        self.round += 1
        print(f'\033[1;34m [ROUND] : {self.round}\033[0m')
    
    def compute_qos_violation(self):
        for c in self.candidate:
            if c[3] > c[4]:
                batch = c[1]    
                self.qos_violation_count += batch
                vio_ratio = [(c[3] - c[4])/c[4] * 100 for _ in range(batch)]
                self.avg_violation_ratio.extend(vio_ratio)
                
    # Shortest Remaining Time scheduling strategy
    def update_cmd_queue(self):
        
        for ind in [self.act_id, self.next_id]:
            if ind >= 0 and not self.w2cQueues[ind].empty():    
                command = self.w2cQueues[ind].get()
                if command == 'task_over':
                    self.load_state[ind] = False
                    self.infer_state[ind] = False
                    self.task_num -= 1
                    self.tCursor -= 1
                    self.candidate[self.cursor-1][3] = 1000 * (time.time() - self.candidate[self.cursor-1][3])
                elif command == 'load_over':
                    self.load_state[ind] = True
                    self.lCursor -= 1
                else:
                    print(f'=== Unknown command: {command} ===')
                    assert False
        
        if self.task_num == 0:
            self.cursor = 0
            self.compute_qos_violation()
            self.compute_candidates()
            
        if self.lCursor <= 0 and self.tCursor <= 0:
            self.notify_to_infer(self.next_id)
            self.act_id = self.next_id
            self.cursor += 1
            if self.cursor < len(self.candidate):
                self.next_id = self.candidate[self.cursor][0]
                self.notify_to_load(self.next_id)
            
        return

    def run(self):
        self.initialization()
        while True:
            self.update_cmd_queue()
            if self.round >= len(self.cmds) and self.cursor >= len(self.candidate):
                break

        for i in range(self.detector_num):
            self.c2wQueues[i].put(pickle.dumps(['exit']))
        
        print(f"All workers has finished jobs, processing {self.total_requests} images in total!")
        print(f'Total number of overtime task is : {self.qos_violation_count}')
        print(f'Average QoS violation ratio is : {sum(self.avg_violation_ratio)/self.qos_violation_count}')
        return