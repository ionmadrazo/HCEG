
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from graphs.models.TransformedEmbeddings import TransformedEmbeddings
from utils.dirs import create_dirs
from tqdm import tqdm
import os
class MHELearning(nn.Module):

    def __init__(self, config, embeddingLoader, logger=None,languages=None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.embeddingLoader= embeddingLoader
        self.lang2TransformedEmbeddings={}
        self.logger.info("Initializing MHELearning model...")
        self.identity = torch.eye(self.config.embedding_dim).to(self.config.device)
        self.langsToConsider=self.config.languages
        if languages is not None:
            self.logger.info("Considering a restricted number of languages.")
            self.langsToConsider=languages
        #Initialize all matrixes
        self.name2Matrix = {}
        checkpointDir="experiments/{}/checkpoints/best/matrixes".format(self.config.exp_name)
        for lang in self.langsToConsider:
            for name in self.config.lang2Matrix[lang]:
                if name not in self.name2Matrix:
                    if self.config.loadCheckpointIfAvailable and os.path.isdir(checkpointDir):
                        #print("########################LOADING CHECKPOINT MATRIX##################")
                        self.name2Matrix[name]=torch.nn.Parameter(torch.load("{}/{}.pytorchmatrix".format(checkpointDir,name)))
                        #print(self.name2Matrix[name])
                    elif self.config.initialization == "random":
                        self.name2Matrix[name]= torch.nn.Parameter(torch.rand(self.config.embedding_dim,self.config.embedding_dim).to(self.config.device),requires_grad=True)
                    elif self.config.initialization == "orthogonal":
                        tmpMatrix = torch.empty(self.config.embedding_dim,self.config.embedding_dim).to(self.config.device)
                        torch.nn.init.orthogonal_(tmpMatrix)
                        self.name2Matrix[name]= torch.nn.Parameter(tmpMatrix,requires_grad=True)
                    elif self.config.initialization == "identity":
                        self.name2Matrix[name]= torch.nn.Parameter(torch.eye(self.config.embedding_dim).to(self.config.device),requires_grad=True)
                    elif self.config.initialization == "load":
                        assert "preload_Matrixes_dir" in self.config, "preload_Matrixes_dir must be defined when load used as initialization"
                        self.name2Matrix[name]=torch.nn.Parameter(torch.load("{}/{}.pytorchmatrix".format(self.config.preload_Matrixes_dir,name)))
                    else:
                        self.name2Matrix[name]= torch.nn.Parameter(torch.rand(self.config.embedding_dim,self.config.embedding_dim).to(self.config.device),requires_grad=True)


                    #self.name2Matrix[name] = self.name2Matrix[name].to(self.config.device)

        self.myparameters= []
        for name in self.name2Matrix:
            self.myparameters.append(self.name2Matrix[name])

        self.logger.info("Initialized {} matrixes with names: [{}].".format(len(self.name2Matrix),",".join(self.name2Matrix.keys())))



        #Initialize transformed embedding obects
        for lang in self.langsToConsider:
            matrixlist = []
            for matrixName in self.config.lang2Matrix[lang]:
                matrixlist.append(self.name2Matrix[matrixName])
            self.lang2TransformedEmbeddings[lang]= TransformedEmbeddings(config=self.config, lang=lang, vectors=self.embeddingLoader.vectors[lang], matrixes=matrixlist, logger=self.logger)

        self.logger.info("MHELearning initialized.")




    def commonRootMatrixName(self,langs=[]):
        assert len(langs)==2, "Finding common root among more than 2 languages not supported yet"
        lang1,lang2=langs[0],langs[1]
        mNames1=self.config.lang2Matrix[lang1]
        mNames2=self.config.lang2Matrix[lang2]
        lenmNames1=len(mNames1)
        lenmNames2=len(mNames2)

        for name1 in mNames1:
            for name2 in mNames2:
                if name1==name2:
                    return name1
        return None


    def forward(self, inputs):
        loss = 0
        for key in inputs:
            embsPerLang={}
            if self.config.useMinimumCommonRoot:
                commonRootMatrixName= self.commonRootMatrixName([lang for lang in inputs[key]])

                for lang in inputs[key]:
                    embsPerLang[lang]=self.lang2TransformedEmbeddings[lang](inputs[key][lang],commonRootMatrixName)
            else:
                for lang in inputs[key]:
                    embsPerLang[lang]=self.lang2TransformedEmbeddings[lang](inputs[key][lang])

            for lang1 in embsPerLang:
                for lang2 in embsPerLang:
                    if lang1!=lang2:
                        loss = loss + torch.nn.MSELoss()(embsPerLang[lang1],embsPerLang[lang2])
                        #loss = loss + -torch.mean(torch.matmul(embsPerLang[lang1].t(), embsPerLang[lang2]))

        separatedLosses= {}
        separatedLosses["pred"] = loss.data
        #regularization of matrixes
        totalMatrixWeightRegularizationLoss=0
        for mName in self.name2Matrix:
           regLoss = torch.norm(self.name2Matrix[mName],p=2)
           totalMatrixWeightRegularizationLoss = totalMatrixWeightRegularizationLoss + regLoss
        totalMatrixWeightRegularizationLoss= totalMatrixWeightRegularizationLoss/len(self.name2Matrix)

        #loss = loss + self.config.matrixRegularizerWeight*totalMatrixWeightRegularizationLoss
        #induce orthogonality on matrixes
        totalOrthLoss=0
        for mName in self.name2Matrix:
            m = self.name2Matrix[mName]
            #identity = torch.eye(self.name2Matrix[mName].size(1))
            orthLoss =torch.norm(torch.matmul(m.t(),m)-self.identity,p=1)
            #print(mName,orthLoss.data)
            totalOrthLoss = totalOrthLoss + orthLoss
            #loss = loss + 0.0001*orthLoss
        totalOrthLoss= totalOrthLoss/len(self.name2Matrix)
        #print("Dist:",loss.data,"Orth",totalOrthLoss.data)
        separatedLosses["orth"] = totalOrthLoss.data
        loss = loss + self.config.orthLossWeight*totalOrthLoss
        return loss, separatedLosses

    def retrievalError(self,sourceTokens,targetTokens, sourceLanguage, targetLangauge):
        trasnfomedEmb = self.lang2TransformedEmbeddings[sourceLanguage](sourceTokens)
        return self.lang2TransformedEmbeddings[sourceLanguage].getClosestTokens(trasnfomedEmb,n)

    def retrieveClosest(self,sourceToken, sourceLanguage, targetLangauge, n):
        trasnfomedEmb = self.lang2TransformedEmbeddings[sourceLanguage](sourceToken)
        return self.lang2TransformedEmbeddings[targetLangauge].getClosestTokens(trasnfomedEmb,n)

    def getEmbedding(self,sourceToken, sourceLanguage):
        return self.lang2TransformedEmbeddings[sourceLanguage](sourceToken)

    def vocabInductionAccuracy(self,inputs):
        accuracyPerKey={}
        for key in tqdm(inputs):
            embsPerLang={}
            commonRootMatrixName=None
            if self.config.useMinimumCommonRoot:
                commonRootMatrixName= self.commonRootMatrixName([lang for lang in inputs[key]])

            for lang in inputs[key]:
                embsPerLang[lang]=self.lang2TransformedEmbeddings[lang](inputs[key][lang],commonRootMatrixName)
            for lang1 in embsPerLang:
                #lang1="en"
                correct = 0
                total =0
                sourceEmbs=embsPerLang[lang1].to(self.config.deviceEval)
                for lang2 in embsPerLang:
                    #lang2="de" #83.5 #84.6 theirs  #84.6401 #72.64
                    if lang1!=lang2:
                        targeAllEmbds = self.lang2TransformedEmbeddings[lang2].getTransformedEmbeddingMatrix(commonRootMatrixName).to(self.config.deviceEval)


                        #norm = self.embeddings.weight.norm(p=2, dim=1, keepdim=True)
                        #self.embeddings.weight.data = self.embeddings.weight.div(norm.expand_as(self.embeddings.weight))
                        cosineSims = torch.matmul(sourceEmbs,targeAllEmbds.t())
                        predictedIdx = torch.argmax(cosineSims,1)

                        src2tgtList={}
                        for i in range(len(inputs[key][lang1])):
                            sourceToken = inputs[key][lang1][i]
                            targetToken = inputs[key][lang2][i]

                            if sourceToken not in src2tgtList:
                                src2tgtList[sourceToken]=[]

                            src2tgtList[sourceToken].append(targetToken)
                        #print(src2tgtList)
                        for i in range(len(inputs[key][lang1])):
                            sourceToken = inputs[key][lang1][i]
                            targetTokens = src2tgtList[sourceToken]
                            predToken = self.lang2TransformedEmbeddings[lang2].itos[predictedIdx[i]]
                            if predToken in targetTokens:
                                correct=correct+1

                            total = total+1
                            #print(sourceToken, targetToken,predToken)

                        #assert False
            accuracyPerKey[key]=correct/total


        return accuracyPerKey

    def saveMatrixes(self, folder):
        create_dirs([folder])


        for name in self.name2Matrix:
            torch.save(self.name2Matrix[name].data,"{}/{}.pytorchmatrix".format(folder,name))
