
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import random

class TransformedEmbeddings(nn.Module):

    def __init__(self, config, lang, vectors, matrixes = [] ,logger=None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.lang= lang
        self.logger.info("Initializing TransformedEmbeddings({})...".format(lang))




        self.embeddings = nn.Embedding.from_pretrained(vectors.vectors)

        if self.config.prepocessing_lengthNormalization:
            norm = self.embeddings.weight.norm(p=2, dim=1, keepdim=True)
            self.embeddings.weight.data = self.embeddings.weight.div(norm.expand_as(self.embeddings.weight))
        #norms = torch.norm(self.embeddings.weight, p=2, dim=1).data
        if self.config.prepocessing_meanCentering:
            means= self.embeddings.weight.mean(dim=0)
            self.embeddings.weight.data = self.embeddings.weight - means

        if self.config.prepocessing_SecondlengthNormalization:
            norm = self.embeddings.weight.norm(p=2, dim=1, keepdim=True)
            self.embeddings.weight.data = self.embeddings.weight.div(norm.expand_as(self.embeddings.weight))


        self.embeddings.weight.requires_grad=False
        self.embeddings = self.embeddings.to(self.config.device)
        self.stoi= vectors.stoi
        self.itos= vectors.itos
        self.matrixes = matrixes
        #print(self.stoi)
        #print(embedding.weight.size())
        #print(vectors.dim, vectors.vectors.size() )

        self.logger.info("TransformedEmbeddings initialized with {} embeddings of dim {} and {} transformation matrixes({}) ".format( self.embeddings.weight.size(0),
                                                                                                                                        self.embeddings.weight.size(1),
                                                                                                                                        len(self.matrixes)
                                                                                                                                        ,lang))





    def forward(self, inputs, maxMatrixName=None):
        maxMatrixIdx=None
        for idx, mName in enumerate(self.config.lang2Matrix[self.lang]):
            if mName==maxMatrixName:
                maxMatrixIdx=idx
                break
        idxList = []
        for token in inputs:
            if token in self.stoi:
                idxList.append(self.stoi[token])
            else:
                #random is used to avoid any posibility of hubbing
                #TODO: should be modified during evaluations
                idxList.append(random.randint(0, len(self.stoi)-1))

        embs = self.embeddings(torch.LongTensor(idxList).to(self.config.device))

        result = embs
        for idx, matrix in enumerate(self.matrixes):
            if maxMatrixIdx is None or idx <= maxMatrixIdx:
                result = torch.matmul(result,matrix)
        return result

    def getTransformedEmbeddingMatrix(self, maxMatrixName=None):
        maxMatrixIdx=None
        for idx, mName in enumerate(self.config.lang2Matrix[self.lang]):
            if mName==maxMatrixName:
                maxMatrixIdx=idx
                break
        result = self.embeddings.weight
        resultM= torch.eye(self.config.embedding_dim,device=self.config.device)
        for idx, matrix in enumerate(self.matrixes):
            if maxMatrixIdx is None or idx <= maxMatrixIdx:
                resultM = torch.matmul(resultM,matrix)
        return torch.matmul(result,resultM)
    def getClosestTokens(self, emb, n):
        transformedEmbs= self.getTransformedEmbeddingMatrix()
        cosineSims = torch.matmul(transformedEmbs,emb.view(self.config.embedding_dim,-1))
        _, order = torch.sort(-cosineSims)
        #return self.itos[order[0]]
        return self.itos[order[0]]
        #return cosineSims[:10], order[:10]
