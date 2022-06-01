
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
    def __init__(self, model) -> None:
        self.model = model.eval()
        self.module_pair = list()
        self.flops = []
        self.mlt = [] # module loading time
        
        self.model._apply(self.pinned_memory)
    
    def __call__(self, img):
        self.run(img)
        # self.model.cpu()
    
    def pinned_memory(self, tensor):
        return tensor.pin_memory()
        
    def run(self, img):
        handler = torch.nn.modules.module.register_module_forward_pre_hook(self.flops_hook())
        self.model(img)
        self.model.cpu()

        handler.remove()
        for module,shape in self.module_pair:
            self.flops.append(2 * Profiler(ForwardModule(module), {'x':torch.randn(shape)}).profile().macs)
    
    def flops_hook(self):      
        def forward_pre_hook(module, data_input):
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                assert len(data_input) == 1
                self.module_pair.append([module, data_input[0].shape])
                start = time_sync()
                module.cuda()
                end = time_sync()
                self.mlt.append(1000*(end-start))
        return forward_pre_hook


        
    
    




