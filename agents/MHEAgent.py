import numpy as np

from tqdm import tqdm
import shutil
import random
import json
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent

# import your classes here

#from tensorboardX import SummaryWriter
#from utils.metrics import AverageMeter, AverageMeterList
#from utils.misc import print_cuda_statistics

cudnn.benchmark = True

from graphs.models.MHELearning import MHELearning

from datasets.embeddings import EmbeddingDataLoader
from datasets.wordPairs import WordPairDataLoader
from graphs.losses.cross_entropy import   CrossEntropyLoss

class MHEAgent(BaseAgent):

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

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """

        #for batch in self.data_loader.train_iter:
        #    print(batch.text[2])
        #    print(batch.text_POS[2])
        #    print(batch.label)
        #    print()
            #break
        #print(self.data_loader.TEXT_POS.vocab.stoi)

        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """

        for epoch in tqdm(range(1, self.config.max_epoch + 1)):
            self.train_one_epoch()
            if epoch % self.config.validate_every ==0:
                self.validate()

            self.current_epoch += 1

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
            bestFolder = "experiments/{}/checkpoints/best/".format(self.config.exp_name)
            self.model.saveMatrixes(bestFolder)

            with open('{}config.json'.format(bestFolder), 'w') as fp:
                json.dump(self.config, fp, default= lambda o: '<not serializable>',indent=4)

    def validateByLoss(self):
        epochError = 0
        for i in range(self.config.batches_per_epoch):

            error = self.model(self.wordPairs_loader.getBatchEval())

            self.optimizer.zero_grad()
            error.backward()
            #self.logger.info("Error: " + str(error.data.tolist()))
            self.optimizer.step()
            epochError = epochError + error.data.tolist()

        self.logger.info("Loss for EVAL epoch {}: {}".format(self.current_epoch,epochError/self.config.batches_per_epoch))

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



    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
