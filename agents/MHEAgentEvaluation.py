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

class MHEAgentEvaluation(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("Initializing Evaluatior agent...")
        self.config= config

        self.config.device= torch.device("cpu")
        self.config.deviceEval= torch.device("cpu")
        self.logger.info("Using CPU as device...")



        #self.embedding_loader = globals()[config.embeddingDataLoader](config=config)
        self.wordPairs_loader = globals()[config.wordPairDataLoader](config=config)



        #self.model = globals()[config.model](config= config,embeddingLoader=self.embedding_loader, logger= self.logger)


        #self.model = self.model.to(self.config.device)

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
            self.evaluate()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def evaluate(self):
        for wordPairFile in self.wordPairs_loader.allPairsByFileEval:
            langPair=self.config.wordPairsEval[wordPairFile]
            lang1,lang2=langPair.split(" ")
            self.evaluateFile(wordPairFile,lang1,lang2)

    def evaluateFile(self,wordPairFile,lang1,lang2):

        self.embedding_loader = globals()[self.config.embeddingDataLoader](config=self.config,languages=[lang1,lang2])
        self.model = globals()[self.config.model](config= self.config,embeddingLoader=self.embedding_loader, logger= self.logger,languages=[lang1,lang2])
        self.model = self.model.to(self.config.device)


        acc=self.validateByVocabInductionAccuracy(wordPairFile,lang1,lang2)
        print(wordPairFile,lang1,lang2,acc)

    def train(self):
        """
        Main training loop
        :return:
        """



    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """




    def validate(self):
        """
        One cycle of model validation
        :return:
        """


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

    def validateByVocabInductionAccuracy(self,filename,lang1,lang2):

        #for i in tqdm(range(self.config.eval_batches_per_epoch)):
        batch= self.wordPairs_loader.getBatchEvalForFile(filename)

        tmpAccuracyPerKey = self.model.vocabInductionAccuracy(batch)


        #self.logger.info("VIA (epoch {}): {} (Mean)".format("ep",np.mean(accList)))

        return tmpAccuracyPerKey[filename]

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
