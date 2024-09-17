# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)    
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)        
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]  
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]