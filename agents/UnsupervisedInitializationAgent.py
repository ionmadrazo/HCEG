import numpy as np

from tqdm import tqdm
import shutil
import random
import json
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import os
from agents.base import BaseAgent

# import your classes here

#from tensorboardX import SummaryWriter
#from utils.metrics import AverageMeter, AverageMeterList
#from utils.misc import print_cuda_statistics

cudnn.benchmark = True

from graphs.models.MHELearning import MHELearning
from graphs.models.UnsupervisedInitArtetxe import UnsupervisedInitArtetxe
from graphs.models.UnsupervisedInitFreq import UnsupervisedInitFreq

from datasets.embeddings import EmbeddingDataLoader
from datasets.wordPairs import WordPairDataLoader
from graphs.losses.cross_entropy import   CrossEntropyLoss

class UnsupervisedInitializationAgent(BaseAgent):

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
        #self.embedding_loader = globals()[config.embeddingDataLoader](config=config)
        self.wordPairs_loader = globals()[config.wordPairDataLoader](config=config)



        #self.model = globals()[config.model](config= config,embeddingLoader=self.embedding_loader, logger= self.logger)



        self.logger.info("UnsupervisedInitializationAgent initialized.")


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



    def run(self):
        """
        The main operator
        :return:
        """
        langPair2Accuracy={}

        if self.config.languagePairs2Generate== "allCombinations":
            self.config.languagePairs2Generate=[]
            for lang1 in self.config.languages:
                for lang2 in self.config.languages:
                    if lang1!=lang2:
                        self.config.languagePairs2Generate.append("{} {}".format(lang1,lang2))

        for langPair in self.config.languagePairs2Generate:

                src_lang, tgt_lang= langPair.split(" ")
                filename="{}/{}-{}.{}.window{}.txt".format(self.config.store_folder,src_lang,tgt_lang,self.config.embedding_max,self.config.retrieval_window)
                if os.path.isfile(filename) and self.config.doNotCreateIfExists:
                    continue
                embedding_loader = globals()[self.config.embeddingDataLoader](config=self.config,languages=[src_lang,tgt_lang])
                model = globals()[self.config.model](self.config,
                                                    src_lang,
                                                    tgt_lang,
                                                    embedding_loader.vectors[src_lang],
                                                    embedding_loader.vectors[tgt_lang],
                                                    self.logger)

                if self.config.evaluate:
                    correct= 0
                    count =0
                    if langPair in self.config.wordPairsEval.values():
                        for key in self.wordPairs_loader.allPairsByFileEval:

                            if  self.config.wordPairsEval[key] == langPair:
                                for i in tqdm(range(len(self.wordPairs_loader.allPairsByFileEval[key][src_lang]))):
                                    src_tok= self.wordPairs_loader.allPairsByFileEval[key][src_lang][i]
                                    tgt_tok= self.wordPairs_loader.allPairsByFileEval[key][tgt_lang][i]
                                    predicted_tgt_tok=model(src_tok)
                                    #print(src_tok,tgt_tok,predicted_tgt_tok==tgt_tok)
                                    if predicted_tgt_tok==tgt_tok:
                                        correct= correct+1
                                        #print(src_tok,predicted_tgt_tok)
                                    if predicted_tgt_tok!= None:
                                        count = count+1
                    langPair2Accuracy[langPair]=correct/count
                    self.logger.info("Accuracy ({}): {}".format(langPair,correct/count))

                if self.config.store_folder is not None:

                    with open(filename,"w",encoding="utf-8") as fout:
                        for src_tok in embedding_loader.vectors[src_lang].stoi:
                            fout.write("{} {}\n".format(src_tok,model(src_tok)))
                del embedding_loader
                del model

        if self.config.evaluate:
            for langPair in langPair2Accuracy:
                self.logger.info("Accuracy ({}): {}".format(langPair,langPair2Accuracy[langPair]))

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
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
