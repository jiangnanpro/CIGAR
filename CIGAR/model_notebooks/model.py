# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, CosineSimilarity

import logging
logger = logging.getLogger(__name__)


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        
        super(Model, self).__init__()
        self.encoder=encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        

    def forward(self, name_user_ids=None,official_ids=None): 
        
        name_user_ids = name_user_ids
        name_official_ids = official_ids[:,0:self.args.block_size]
        description_official_ids = official_ids[:,self.args.block_size:]
              
        name_user_ids=name_user_ids.view(-1,self.args.block_size)
        name_official_ids=name_official_ids.view(-1,self.args.block_size)
        description_official_ids=description_official_ids.view(-1,self.args.block_size)

        name_user_outputs = self.encoder(input_ids= name_user_ids,attention_mask=name_user_ids.ne(1))[0]
        name_official_outputs = self.encoder(input_ids= name_official_ids,attention_mask=name_official_ids.ne(1))[0]
        description_official_outputs = self.encoder(input_ids= description_official_ids,attention_mask=description_official_ids.ne(1))[0]
        
        name_user_outputs = name_user_outputs[:,0,:]
        name_official_outputs = name_official_outputs[:,0,:]
        description_official_outputs = description_official_outputs[:,0,:]

        
        # try different way to aggregate the two official vectors.
        
        combined_official_outputs = torch.div((name_official_outputs+description_official_outputs),2) # -> Arithmetic mean
        
        #combined_official_outputs = torch.sqrt(torch.square(name_official_outputs)+torch.square(description_official_outputs)) # -> L2-norm


        # train CIGAR with different features combinations.
        
        #return name_user_outputs, name_official_outputs
        #return name_user_outputs, description_official_outputs
        return name_user_outputs, combined_official_outputs
    
      
    
    
    def encode(self, input_ids=None):
        
        return self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(1))[0]