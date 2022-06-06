
import torch
import torch.nn as nn
from nnutils.cnn_complexity_analyzer.profile import Profiler
from utils.torch_utils import time_sync

class ForwardModule(nn.Module):
    def __init__(self,module):
        super(ForwardModule,self).__init__()
        self.module=module

    def forward(self,x):
        return self.module(x)

class Analyser:
    def __init__(self, model, id) -> None:
        self.model = model.eval()
        self.id = id
        self.modules = list()
        self.mShape = list()
        self.flops = []
        self.mlt = [] # module loading time
        self.module_num = 0
        self.bs = 0
        self.ddl = 0
        self.lat = 0
        self.srt = 0
        self.cur_mod = 0 # denotes the current model module to execute
        
        self.model._apply(self.pinned_memory)
    
    def __call__(self, img):
        self.run(img)
    
    def pinned_memory(self, tensor):
        return tensor.pin_memory()
        
    def run(self, img):
        handler = torch.nn.modules.module.register_module_forward_pre_hook(self.flops_hook())
        self.model(img)
        self.model.cpu()

        handler.remove()
        for module,shape in zip(self.modules, self.mShape):
            self.flops.append(2 * Profiler(ForwardModule(module), {'x':torch.randn(shape)}).profile().macs)
    
    def flops_hook(self):      
        def forward_pre_hook(module, data_input):
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                self.module_num += 1
                assert len(data_input) == 1
                self.modules.append(module)
                self.mShape.append(data_input[0].shape)
                start = time_sync()
                module.cuda()
                end = time_sync()
                self.mlt.append(1000*(end-start))
        return forward_pre_hook


        
    
    




