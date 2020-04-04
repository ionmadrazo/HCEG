
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import sys
import psutil
import os
import gc

def memReport():
  for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


class UnsupervisedInitFreq(nn.Module):

    def __init__(self, config, lang_source, lang_target, vectors_source, vectors_target, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger

        self.logger.info("Computing Unsupervised Initialization for languages: {} {}...".format(lang_source,lang_target))

        self.lang_source= lang_source
        self.lang_target= lang_target

        self.vectorsL1= vectors_source
        self.vectorsL2= vectors_target

        self.vectorsL1.vectors=self.vectorsL1.vectors.to(self.config.device)
        #self.vectorsL1.vectors=self.vectorsL1.vectors.to(torch.device("cpu"))
        torch.cuda.empty_cache()

        #self.vectorsL1.vectors=self.vectorsL1.vectors.to(self.config.device)

        #self.embeddings = nn.Embedding.from_pretrained(vectors.vectors)

        #norm = self.embeddings.weight.norm(p=2, dim=1, keepdim=True)
        #self.embeddings.weight.data = self.embeddings.weight.div(norm.expand_as(self.embeddings.weight))
        #norms = torch.norm(self.embeddings.weight, p=2, dim=1).data
        self.logger.info("Computing word2word L1...")
        w2wL1= torch.matmul(self.vectorsL1.vectors,self.vectorsL1.vectors.t())

        self.logger.info("Sorting word2word L1...")
        w2wL1,_ = torch.sort(w2wL1,dim=1)

        del _
        #torch.cuda.empty_cache()

        self.logger.info("Normalizing word2word L1...")
        norm = w2wL1.norm(p=2, dim=1, keepdim=True)
        w2wL1 = w2wL1.div(norm.expand_as(w2wL1))


        self.vectorsL1.vectors=self.vectorsL1.vectors.to(torch.device("cpu"))

        w2wL1=w2wL1.to(torch.device("cpu"))
        #torch.cuda.empty_cache()

        self.vectorsL2.vectors=self.vectorsL2.vectors.to(self.config.device)

        self.logger.info("Computing word2word L2...")
        w2wL2= torch.matmul(self.vectorsL2.vectors,self.vectorsL2.vectors.t())

        self.logger.info("Sorting word2word L2...")
        w2wL2,_ = torch.sort(w2wL2,dim=1)
        del _
        self.logger.info("Normalizing word2word L2...")
        norm = w2wL2.norm(p=2, dim=1, keepdim=True)
        w2wL2 = w2wL2.div(norm.expand_as(w2wL2))



        w2wL2=w2wL2.to(torch.device("cpu"))
        torch.cuda.empty_cache()


        w2wL1=w2wL1.to(self.config.device)
        w2wL2=w2wL2.to(self.config.device)
        self.logger.info("Computing similarities...")
        self.similarities = torch.matmul(w2wL1,w2wL2.t())
        del norm
        del w2wL1
        del w2wL2
    
        #for i in tqdm(range(100)):
        #    sourceWord= vectorsL1.itos[i]
        #    targetIdx=torch.argmax(similarities[i])
        #    targetWord= vectorsL2.itos[targetIdx]
        #    print(sourceWord,targetWord)
        self.logger.info("Computed Unsupervised Initialization for languages: {} {}".format(lang_source,lang_target))





    def forward(self, src_token):

        if src_token not in self.vectorsL1.stoi:
            return None
        else:
            src_idx =self.vectorsL1.stoi[src_token]
            start_idx= max(0,src_idx-self.config.retrieval_window)
            end_idx=min(len(self.vectorsL1.vectors),src_idx+self.config.retrieval_window)
            targetIdx=torch.argmax(self.similarities[src_idx][start_idx:end_idx])
            targetWord= self.vectorsL2.itos[targetIdx+start_idx]
            return targetWord
