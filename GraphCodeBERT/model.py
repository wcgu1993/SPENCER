import torch.nn as nn
import torch    
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            return self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
