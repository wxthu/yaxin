
import pickle
from load_mock_data import load_mock_data
from analysis import Analyser
import torch
import joblib
import numpy as np

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, img_num=800, configs=None, models=None, c2wQueues=None, w2cQueues=None, imgQueues=None):
        """ 
        two-way communiaction to prevent message contention
        """
        self.detector_num = len(configs)
        self.load_state = [False for _ in range(self.detector_num)]
        self.candidate = []
        self.act_id = None
        self.aux_id = None
        self.next_act_id = None
        self.next_aux_id = None
        self.actives = [] # set of act_id and aux_id
        self.aar = []     # active and ready workers
        self.t_count = 0  # indicate whether parallel two tasks have completed, maximal value 2
        self.l_count = 0  # indicate whether parallel two models have loaded into gpu , maximal value 2
        self.aux_new_cur_mod = 0
        self.img_num = img_num
        self.c2wQueues = c2wQueues
        self.w2cQueues = w2cQueues
        self.imgQueues = imgQueues
        self.cmds = []  # batch number 
        self.deadlines = []
        self.round = 0 
        self.task_num = 0 # record the number of tasks at the moment
        self.total_requests = 0
        self.config = configs  # list, each element is dictionary
        self.models = models
        self.load_time_limit = 10000000
        self.to_compute = False
        self.both_full = False # denote whether parallerl op groups have reached the end of models
        self.regressor = joblib.load('./train/lat_pred.pkl')
        
        # for i, m in enumerate(model_configs):
        #     self.config.append(m.pop())
        #     self.models.append(Analyser(m.pop(), i))
            
        # print('*** Controller needs to collect models properties and wait for a while .. ***')
        # for m in self.models:
        #     fake_img = torch.randn(1, 3, 384, 640).cuda()
        #     m(fake_img)
            
        # torch.cuda.empty_cache()
        print('##### Controller init success #####')
        
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
        
        cmd = self.cmds[self.round]
        ddls = self.deadlines[self.round]
        self.candidate.clear()
        
        for i, (batch, ddl) in enumerate(zip(cmd, ddls)):
            if batch:
                self.models[i].bs = batch
                self.models[i].ddl = sum(ddl)
                self.models[i].lat = self.config[i][batch][1]['infer_time']
                self.models[i].srt = sum(ddl) - self.models[i].lat
                self.candidate.append(self.models[i])
        
        self.candidate.sort(key=lambda obj: obj.srt)
        assert len(self.candidate) > 0, 'no image input, please re-generate mocked data'
        self.task_num = len(self.candidate)

        self.act_id = self.candidate[0].id
        self.aux_id, stop_module_index = self.determine_parallel_ops()
        self.aux_new_cur_mod = stop_module_index   # possible to get -1
        self.c2wQueues[self.act_id].put(pickle.dumps(['loading', self.candidate[0].module_num, self.candidate[0].bs]))
        self.c2wQueues[self.act_id].put(pickle.dumps(['inference']))
        self.load_state[self.act_id] = True
        
        if self.aux_id >= 0: 
            self.c2wQueues[self.aux_id].put(pickle.dumps(['loading', stop_module_index, self.candidate[1].bs]))
            self.c2wQueues[self.aux_id].put(pickle.dumps(['inference']))
            
        self.load_time_limit = self.get_predicted_latency(self.candidate[0].module_num, stop_module_index)
        self.actives = [self.act_id] if self.aux_id < 0 else [self.act_id, self.aux_id]
        self.compute_next_workers()
        self.aar = [self.act_id, self.aux_id, self.next_act_id, self.next_aux_id]
        self.round += 1
       
    # To check prediction model , TODO in future
    def get_predicted_latency(self, stop1, stop2):
        flops1 = self.get_module_flops(self.candidate[0].id, self.candidate[0].cur_mod, stop1)
        if len(self.candidate) > 1:
            flops2 = self.get_module_flops(self.candidate[1].id, self.candidate[1].cur_mod, stop2)
        else:
            return self.regressor.predict(
                np.array([self.candidate[0].bs, 0, flops1, 0]).reshape(1,4))
            
        return self.regressor.predict(
            np.array([self.candidate[0].bs, self.candidate[1].bs, flops1, flops2]).reshape(1,4))
        
    
    # Shortest Remaining Time scheduling strategy
    def update_cmd_queue(self):
        
        if self.task_num == 0:
            self.load_time_limit = 10000000
            self.compute_candidates()
        
        for id in self.aar:  
            if id < 0:
                continue 
            command = None if self.w2cQueues[id].empty() else self.w2cQueues[id].get()
            if command is not None:
                if command == 'task_over':
                    self.load_state[id] = False 
                    self.task_num -= 1
                    increment = 2 if len(self.actives) == 1 else 1
                    self.t_count += increment
                    print(f'+!!!!inf over  ,worker {id} and count: {self.t_count}')
                elif command == 'partial_task_over':
                    self.load_state[id] = False
                    self.t_count += 1
                    print(f'+!!!!PPinf over  ,worker {id} and count: {self.t_count}')
                elif command == 'load_over':
                    self.load_state[id] = True
                    increment = 2 if len(self.actives) == 1 else 1
                    self.l_count += increment
                    print(f'++++++++++++++ loadddd ,worker {id} and count: {self.l_count}')
                elif command == 'partial_load_over':
                    self.load_state[id] = True
                    self.l_count += 1
                    print(f'++PP++++++++++++ loadddd ,worker {id} and count: {self.l_count}')
                else:
                    print(f'=== Unknown command: {command} ===')
                    assert False
                    
        if self.l_count >= 2:
            # Pipeline: loading next model while current inference
            print(f'GGGGGGGGGGGGGGGGGGGGGGGG {self.aar}')
            print(f'load next and aux : {self.next_act_id}/{self.next_aux_id}')
            if self.next_act_id >= 0:
                if not self.load_state[self.next_act_id]:
                    self.c2wQueues[self.next_act_id].put(pickle.dumps(
                        ['loading', self.candidate[0].module_num, self.candidate[0].bs]))
                    print(f'ctrl send 1 lllll **********worker {self.next_act_id}*********')
                if self.next_aux_id >= 0 and not self.load_state[self.next_aux_id]:
                    self.c2wQueues[self.next_aux_id].put(pickle.dumps(
                        ['loading', self.aux_new_cur_mod, self.candidate[1].bs]))
                    print(f'ctrl send aux lllll **********worker {self.next_aux_id}*********')
                    self.l_count = 0
                else:
                    self.l_count = 1
            print(f'ctrl reset lcount : {self.next_act_id}&&{self.next_aux_id}')                            
            
        if self.t_count >= 2:
            print(f'next act and aux : {self.next_act_id}/{self.next_aux_id}')
            print(self.load_state[self.next_act_id])
            if self.next_act_id >= 0:
                if self.next_aux_id >= 0:
                    # make sure model parameters have loaded successfully
                    if self.load_state[self.next_act_id] and self.load_state[self.next_aux_id]:
                        print(f'ctrl send 2 inf **********{self.next_act_id}${self.next_aux_id}***************')
                        self.c2wQueues[self.next_aux_id].put(pickle.dumps(['inference']))
                        self.aux_id = self.next_aux_id
                        # self.t_count = 0
                        self.to_compute = True
                else:
                    if self.load_state[self.next_act_id]:
                        print(f'ctrl send 1 inf *****************worker {self.next_act_id}*********')
                        self.aux_id = -1
                        # self.t_count = 1
                        self.to_compute = True
                        
                if self.to_compute:
                    self.c2wQueues[self.next_act_id].put(pickle.dumps(['inference']))
                    self.act_id = self.next_act_id
                    self.compute_next_workers()
                    if len(self.candidate) > 0:
                        self.load_time_limit = self.get_predicted_latency(
                            self.candidate[0].module_num, self.aux_new_cur_mod)
                        
                    self.actives = [self.act_id] if self.aux_id < 0 else [self.act_id, self.aux_id]
                    self.aar = [self.act_id, self.aux_id, self.next_act_id, self.next_aux_id]
                    self.to_compute = False
                    print(f'7777777 new aar : {self.aar} *******load count : {self.l_count}')
                    if self.next_act_id >= 0:
                        self.c2wQueues[self.next_act_id].put(pickle.dumps(['loading', self.candidate[0].module_num, self.candidate[0].bs]))
                    if self.next_aux_id >= 0:
                        self.c2wQueues[self.next_aux_id].put(pickle.dumps(['loading', self.candidate[1].module_num, self.candidate[1].bs]))
        
                    self.t_count = 0
                    
    
    def get_module_flops(self, model_id, start, stop):
        if stop <= start:
            return 0
        return sum(self.models[model_id].flops[start:stop])
    
    def get_module_loading_time(self, model_id, start, stop):
        if stop <= start:
            return 0
        return sum(self.models[model_id].mlt[start:stop])
        
    def determine_parallel_ops(self):
        if len(self.candidate) <= 1:
            return -1, -1
        
        start1 = self.candidate[0].cur_mod
        stop1 = self.candidate[0].module_num 
        
        if self.get_module_loading_time(self.candidate[0].id, start1, stop1) >= self.load_time_limit:
            print('*** current modules loading too long and no parallel modules ***')
            return -1, -1
        
        for i in range(1, len(self.candidate)):
            if self.candidate[i].ddl <= self.candidate[0].lat:
                print('### urgent task need to be executed immediately ###')
                self.both_full = True
                return i, self.candidate[i].module_num
        
        start2 = self.candidate[1].cur_mod
        stop2 = self.candidate[1].module_num
        flops1 = self.get_module_flops(self.candidate[0].id, start1, stop1)
        
        # binary search to find maximal op range
        rs = start2
        rt = stop2
        index = stop2
        selected = index
        
        while rt > rs:
            flops2 = self.get_module_flops(self.candidate[1].id, start2, index)
            pred_lat = self.regressor.predict(
                np.array([self.candidate[0].bs, self.candidate[1].bs, flops1, flops2]).reshape(1,4))
            if pred_lat <= self.candidate[0].ddl:
                rs = index + 1
            else:
                rt = index - 1
        
            selected = index
            index = int((rs + rt) / 2)
        
        ltime1 = self.get_module_loading_time(self.candidate[0].id, start1, stop1)            
        while start2 < selected:
            ltime2 = self.get_module_loading_time(self.candidate[1].id, start2, selected)
            if ltime1 + ltime2 <= self.load_time_limit:
                break
            selected -= 1
        
        # set flag in order to update states later
        if selected == stop2:  
            self.both_full = True
            
        if selected == start2:
            return -1, -1
        
        return self.candidate[1].id, selected 
    
    def update_candidates_state(self):
        self.candidate.pop(0)
        if self.both_full:
            self.candidate.pop(0)
            self.both_full = False
            
        w_num = len(self.candidate)
        if w_num > 0:
            for i in range(0, w_num):            
                if i == 0:
                    self.candidate[i].cur_mod = self.aux_new_cur_mod
                else:
                    self.candidate[i].srt -= self.load_time_limit
        
            self.candidate.sort(key=lambda obj: obj.srt)
            return True
        
        return False
        
    def compute_next_workers(self):
        notOver = self.update_candidates_state()
        self.next_act_id = self.candidate[0].id if notOver else -1
        if self.next_act_id > 0:
            self.next_aux_id, self.aux_new_cur_mod = self.determine_parallel_ops()
        else:
            self.next_aux_id = -1
            
            
    def run(self):
        self.initialization()
        while True:
            self.update_cmd_queue()
            if self.round >= len(self.cmds) and len(self.candidate) == 0:
                break

        for i in range(self.detector_num):
            self.c2wQueues[i].put(pickle.dumps('exit'))
        
        print(f"all workers has finished jobs, processing {self.total_requests} images in total!")
        return