
from load_mock_data import load_mock_data
import time

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, detector_num=3, img_num=800, config=None,
                c2wQueues=None, w2cQueues=None, imgQueues=None):
        """ 
        two-way communiaction to prevent message contention
        """
        self.detector_num = detector_num
        self.load_state = [False for _ in range(detector_num)]
        self.act_id = None
        self.next_id = None
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
        self.completion_time = [] # recording the completion time for each worker
        
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
        # assert len(self.candidate) > 0, 'no image input, please re-generate mocked data'
        
        self.act_id = self.candidate[0][0]
        self.task_num = len(self.candidate)
        
        self.c2wQueues[self.act_id].put('loading')
        self.c2wQueues[self.act_id].put(self.candidate[0][1])
        self.c2wQueues[self.act_id].put('inference')
        self.load_state[self.act_id] = True
        self.cursor += 1
        self.next_id = self.candidate[self.cursor][0] if self.cursor < len(self.candidate) else None
        self.load_state[self.act_id] = True
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
        
        if self.task_num == 0:
            self.cursor = 0
            self.compute_qos_violation()
            self.compute_candidates()
            
        command = self.w2cQueues[self.act_id].get()
        if command == 'task_over':
            self.load_state[self.act_id] = False
            self.task_num -= 1
            self.candidate[self.cursor-1][3] = 1000 * (time.time() - self.candidate[self.cursor-1][3])
            if self.next_id is not None:
                self.c2wQueues[self.next_id].put('inference')
                self.act_id = self.next_id
                self.cursor += 1
                self.next_id = self.candidate[self.cursor][0] if self.cursor < len(self.candidate) else None
        
        elif command == 'load_over':
            # Pipeline: loading next model while current inference
            if self.next_id is not None:
                if self.load_state[self.next_id] is False:
                    self.c2wQueues[self.next_id].put('loading')
                    self.c2wQueues[self.next_id].put(self.candidate[self.cursor][1])
                    self.load_state[self.next_id] = True    
        else:
            print(f'=== Unknown command: {command} ===')
            assert False
            
        return

    def run(self):
        self.initialization()
        while True:
            self.update_cmd_queue()
            if self.round >= len(self.cmds) and self.cursor >= len(self.candidate):
                break

        for i in range(self.detector_num):
            self.c2wQueues[i].put('exit')
        
        print(f"All workers has finished jobs, processing {self.total_requests} images in total!")
        print(f'Total number of overtime task is : {self.qos_violation_count}')
        print(f'Average QoS violation ratio is : {sum(self.avg_violation_ratio)/self.qos_violation_count}')
        return