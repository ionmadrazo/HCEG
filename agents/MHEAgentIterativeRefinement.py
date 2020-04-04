import numpy as np

from tqdm import tqdm
import shutil
import random
import json
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import sys

from agents.base import BaseAgent
import utils.wordPairInference
# import your classes here

#from tensorboardX import SummaryWriter
#from utils.metrics import AverageMeter, AverageMeterList
#from utils.misc import print_cuda_statistics

cudnn.benchmark = True

from graphs.models.MHELearning import MHELearning

from datasets.embeddings import EmbeddingDataLoader
from datasets.wordPairs import WordPairDataLoader
from graphs.losses.cross_entropy import   CrossEntropyLoss
import utils.wordPairInference
class MHEAgentIterativeRefinement(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("Initializing Read2VecAgent...")
        self.config= config

        if "useGPU" in self.config and self.config.useGPU and torch.cuda.is_available():
            self.config.device= torch.device("cuda")
            self.logger.info("Using CUDA as device...")
        else:
            self.config.device= torch.device("cpu")
            self.logger.info("Using CPU as device...")

        if "useGPUeval" in self.config and self.config.useGPUeval and torch.cuda.is_available():
            self.config.deviceEval= torch.device("cuda")
            self.logger.info("Using CUDA as Eval device...")
        else:
            self.config.deviceEval= torch.device("cpu")
            self.logger.info("Using CPU as Eval device...")

        #This is a fake vector so that the cluster does not kick us (if we use more than X GB or ram before using the GPU it will automatically kick us out)
        if "useGPU" in self.config and self.config.useGPU:
            self.fakeTensor = torch.cuda.FloatTensor(10, 10).fill_(0)
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()
        self.embedding_loader = globals()[config.embeddingDataLoader](config=config)
        self.wordPairs_loader = globals()[config.wordPairDataLoader](config=config)


        #using only embeddings that are appearing on the dataset

        self.model = globals()[config.model](config= config,embeddingLoader=self.embedding_loader, logger= self.logger)

        self.loss = globals()[config.loss](config=config)

        self.model = self.model.to(self.config.device)
        # define loss
        #self.loss = nn.NLLLoss()

        # define optimizer
        #print(self.model.parameters())
        #self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.myparameters), lr=self.config.learning_rate)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.myparameters), lr = config.learning_rate)
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.current_metric = 0

        self.logger.info("MHEAgent initialized.")


    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, isbest=False, saveWordPairs=False ):
        folder=""
        if isbest:
            #If the check point is best we also want it to be saved as last so repeate the call
            self.save_checkpoint(isbest=False,saveWordPairs=saveWordPairs)
            folder = "experiments/{}/checkpoints/best/".format(self.config.exp_name)
        else:
            folder = "experiments/{}/checkpoints/last/".format(self.config.exp_name)
        matrixFolder=folder+"matrixes/"
        wordPairsFolder=folder+"wordpairs/"
        self.model.saveMatrixes(matrixFolder)
        if saveWordPairs:
            self.wordPairs_loader.saveWordPairs(wordPairsFolder)

        with open('{}config.json'.format(folder), 'w') as fp:
            json.dump(self.config, fp, default= lambda o: '<not serializable>',indent=4)

    def run(self):
        """
        The main operator
        :return:
        """
        self.bestIRLoss=sys.float_info.max
        self.bestIRiteration = 0
        self.currentIRiteration=0
        try:
            if self.config.IR:
                while(self.currentIRiteration-self.bestIRiteration<=self.config.IR_itersWithoutImprovement):
                    currentIRLoss=self.train()

                    #currentIRLoss = self.validateByLoss()
                    if currentIRLoss<self.bestIRLoss:
                        self.bestIRLoss= currentIRLoss
                        self.bestIRiteration=self.currentIRiteration

                    self.infer()
                    self.save_checkpoint(saveWordPairs=True)
                    self.currentIRiteration=self.currentIRiteration+1
                self.logger.info("Iterative refinement stopping criteria reached.")
                self.save_checkpoint(saveWordPairs=True)
                self.validate()
            else:
                self.train()
                self.validate()



        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        lastLoss = 0
        bestLoss = sys.float_info.max
        bestLossEpoch = 0

        for epoch in tqdm(range(1, self.config.max_epoch + 1)):
            lastLoss=self.train_one_epoch()
            self.save_checkpoint()
            if self.current_epoch % self.config.validate_every ==0 and self.current_epoch!=0 :
                self.validate()
            if lastLoss< bestLoss:
                bestLoss=lastLoss
                bestLossEpoch=self.current_epoch

                #print(self.current_epoch-bestLossEpoch,self.current_epoch,bestLossEpoch)
            if self.current_epoch-bestLossEpoch > self.config.itersWithoutImprovement:
                self.logger.info("Train stopping criteria reached.")
                break
            self.logger.info("Best epoch: {}, Current epoch: {}, Optimization will stop in {} epochs if no improvement is found.".format(bestLossEpoch,self.current_epoch,(self.config.itersWithoutImprovement)-(self.current_epoch-bestLossEpoch)))

            self.current_epoch += 1
        return lastLoss

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        epochError = 0
        for i in tqdm(range(self.config.batches_per_epoch)):

            error,separatedLosses = self.model(self.wordPairs_loader.getBatch())

            self.optimizer.zero_grad()
            error.backward()
            #self.logger.info("Error: " + str(error.data.tolist()))
            self.optimizer.step()
            epochError = epochError + error.data.tolist()

        self.logger.info("Loss for epoch {}: {}".format(self.current_epoch,epochError/self.config.batches_per_epoch))
        print(separatedLosses)
        return epochError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        #self.validateByLoss()
        self.current_metric=self.validateByVocabInductionAccuracy()
        #self.validateByCosineSimilarity()
        #self.validateByRetrieval()
        if(self.current_metric>=self.best_metric):

            self.best_metric=self.current_metric
            self.save_checkpoint(isbest=True,saveWordPairs=True)
        return 0

    def validateByLoss(self):
        epochError = 0
        for i in range(self.config.batches_per_epoch):

            error = self.model(self.wordPairs_loader.getBatchEval())

            #self.optimizer.zero_grad()
            #error.backward()
            #self.logger.info("Error: " + str(error.data.tolist()))
            #self.optimizer.step()
            epochError = epochError + error.data.tolist()

        self.logger.info("Loss for EVAL epoch {}: {}".format(self.current_epoch,epochError/self.config.batches_per_epoch))
        return epochError
    def validateByVocabInductionAccuracy(self):
        accuracyPerKey={}
        for i in tqdm(range(self.config.eval_batches_per_epoch)):

            tmpAccuracyPerKey = self.model.vocabInductionAccuracy(self.wordPairs_loader.getBatchEval())
            for key in tmpAccuracyPerKey:
                if key not in accuracyPerKey:
                    accuracyPerKey[key]=[]
                accuracyPerKey[key].append(tmpAccuracyPerKey[key])

        accList = []
        for key in tmpAccuracyPerKey:
            acc = np.mean(accuracyPerKey[key])
            accList.append(acc)
            self.logger.info("VIA (epoch {}): {} {}".format(self.current_epoch,acc,key))
        mean = np.mean(accList)

        self.logger.info("VIA (epoch {}): {} (Mean)".format(self.current_epoch,np.mean(accList)))
        return mean

    def validateByCosineSimilarity(self):
        pass

    def validateByRetrieval(self):

        for file in self.wordPairs_loader.allPairsByFileEval:
            langs = [lang for lang in self.wordPairs_loader.allPairsByFileEval[file]]
            npairs = self.wordPairs_loader.file2length[file]
            for lang1 in langs:
                for lang2 in langs:
                    if lang1!=lang2:
                        #for i in range(npairs):
                        for i in range(10):
                            sourceToken=self.wordPairs_loader.allPairsByFileEval[file][lang1][i]
                            targetToken=self.wordPairs_loader.allPairsByFileEval[file][lang2][i]
                            #print(sourceToken,targetToken,self.model.retrieveClosest(sourceToken, lang1, lang2,10))

                            #print(sourceToken)
                            sEmb = self.model.getEmbedding([sourceToken],lang1)
                            tEmb = self.model.getEmbedding([targetToken],lang2)
                            print(sourceToken,targetToken,torch.nn.functional.cosine_similarity(sEmb, tEmb))

                        sourceToken="gato"
                        targetToken="math"
                        #print(sourceToken)
                        sEmb = self.model.getEmbedding([sourceToken],"es")
                        tEmb = self.model.getEmbedding([targetToken],"en")
                        print(sourceToken,targetToken,torch.nn.functional.cosine_similarity(sEmb, tEmb))
                            #print(targetToken,tEmb)

                            #break
                            #print(sourceToken,self.model.retrieveClosest([sourceToken], lang1, lang2,10))

    def infer(self):
        """
        Used to infer the word pairs for next iteration of Iterative Refinement (IR)
        :return:
        """
        if not self.config.IR_keepInitialDictionary:
            self.wordPairs_loader.allPairsByFile={} #Remove all pairs
            #self.wordPairs_loader.file2length={}
        for langPair in tqdm(self.config.IR_langsToinfer):
            lang1,lang2=langPair.split()

            commonRootMatrixName=None
            if self.config.useMinimumCommonRoot:
                commonRootMatrixName= self.model.commonRootMatrixName([lang1,lang2])

            sourceEmbs=self.model.lang2TransformedEmbeddings[lang1].getTransformedEmbeddingMatrix(commonRootMatrixName).to(self.config.deviceEval)
            targetEmbs=self.model.lang2TransformedEmbeddings[lang2].getTransformedEmbeddingMatrix(commonRootMatrixName).to(self.config.deviceEval)

            if self.config.IR_cutoff is not None:
                sourceEmbs=sourceEmbs[:self.config.IR_cutoff]
                targetEmbs=targetEmbs[:self.config.IR_cutoff]

            pairedIdx=[]
            if self.config.IR_inferAlgorithm=="NN":
                pairedIdx=utils.wordPairInference.NN(sourceEmbs,targetEmbs,self.config.inferenceChunkSize,self.config.IR_retrievalWindow)
            elif self.config.IR_inferAlgorithm=="CSLS":
                pairedIdx=utils.wordPairInference.CSLS(sourceEmbs,targetEmbs,self.config.inferenceChunkSize,self.config.IR_retrievalWindow)
            else:
                pairedIdx=utils.wordPairInference.NN(sourceEmbs,targetEmbs,self.config.inferenceChunkSize,self.config.IR_retrievalWindow)

            self.wordPairs_loader.allPairsByFile[langPair]={}
            self.wordPairs_loader.allPairsByFile[langPair][lang1]=[]
            self.wordPairs_loader.allPairsByFile[langPair][lang2]=[]
            for idx1,idx2 in pairedIdx:
                token1= self.model.lang2TransformedEmbeddings[lang1].itos[idx1]
                token2= self.model.lang2TransformedEmbeddings[lang2].itos[idx2]
                self.wordPairs_loader.allPairsByFile[langPair][lang1].append(token1)
                self.wordPairs_loader.allPairsByFile[langPair][lang2].append(token2)
                #print(token1,token2)
            self.wordPairs_loader.file2length[langPair]=len(pairedIdx)

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
