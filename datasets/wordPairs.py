"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import imageio
import torch
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging
import os
import re
import os.path
import wget
from torchtext.vocab import Vectors
from torchtext import data
from tqdm import tqdm
from torchtext.data import Iterator, BucketIterator
import random
from utils.dirs import create_dirs
import numpy as np
from utils.dirs import create_dirs
class WordPairDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("WordPairDataLoader")
        self.logger.info("Initializing WordPairDataLoader...")
        self.allPairsByFile = {}
        self.allPairsByFileEval = {}
        self.file2length= {}
        for file in self.config.wordPairs:
            with open(file, mode='r', encoding="utf-8") as f:
                count = 0
                self.allPairsByFile[file]= {}
                languages = self.config.wordPairs[file].split(self.config.wordPairSeparator)
                for language in languages:
                    self.allPairsByFile[file][language]= []
                for line in f.readlines():

                    words = line.rstrip("\r\n").split(self.config.wordPairSeparator)

                    if len(languages)== len(words):
                        count= count +1
                        for i in range(len(languages)):
                            self.allPairsByFile[file][languages[i]].append(words[i])
                    else:
                        words = line.rstrip("\r\n").split("\t")

                        if len(languages)== len(words):
                            count= count +1
                            for i in range(len(languages)):
                                self.allPairsByFile[file][languages[i]].append(words[i])
            self.file2length[file]=count
            self.logger.info("{} pairs loaded from file {}".format(count,file))


        for file in self.config.wordPairsEval:
            with open(file, mode='r', encoding="utf-8") as f:
                count = 0
                self.allPairsByFileEval[file]= {}
                languages = self.config.wordPairsEval[file].split(self.config.wordPairSeparator)
                for language in languages:
                    self.allPairsByFileEval[file][language]= []
                for line in f.readlines():
                    words = line.rstrip("\r\n").split(self.config.wordPairSeparator)
                    if len(languages)== len(words):
                        count= count +1
                        for i in range(len(languages)):
                            self.allPairsByFileEval[file][languages[i]].append(words[i])
                    else:
                        words = line.rstrip("\r\n").split("\t")
                        if len(languages)== len(words):
                            count= count +1
                            for i in range(len(languages)):
                                self.allPairsByFileEval[file][languages[i]].append(words[i])
            self.file2length[file]=count
            self.logger.info("{} pairs loaded from Eval file {}".format(count,file))





        self.logger.info("WordPairDataLoader initialized.")

    def getBatch(self):
        self.config.batch_size
        batch={}
        selectedPairfileIdx= np.random.randint(low=0, high=len(self.allPairsByFile), size=(self.config.diff_dict_per_batch,))
        for i in range(len(selectedPairfileIdx)):
            key = list(self.allPairsByFile.keys())[selectedPairfileIdx[i]]
            batchName = "{}-{}".format(key,i)
            batch[batchName]={}
            selectedPairs = np.random.randint(low=0, high=self.file2length[key], size=(self.config.batch_size,))


            for lang in self.allPairsByFile[key]:
                batch[batchName][lang]=[]
                for pairIdx in selectedPairs:
                    batch[batchName][lang].append(self.allPairsByFile[key][lang][pairIdx])


        return batch


    def getBatchEval(self):
        self.config.batch_size
        batch={}
        #selectedPairfileIdx= np.random.randint(low=0, high=len(self.allPairsByFileEval), size=(self.config.diff_dict_per_batch,))

        for key in self.allPairsByFileEval:
            try:
                #key = list(self.allPairsByFileEval.keys())[selectedPairfileIdx[i]]
                batchName = key
                batch[batchName]={}
                selectedPairs = np.random.randint(low=0, high=self.file2length[key], size=(self.config.batch_size,))


                for lang in self.allPairsByFileEval[key]:
                    batch[batchName][lang]=[]
                    for pairIdx in selectedPairs:
                        batch[batchName][lang].append(self.allPairsByFileEval[key][lang][pairIdx])
            except:
                self.logger.info("Could not generate a batch for eval file: {}".format(key))


        return batch

    def getBatchEvalForFile(self,filename):
        batch={}
        key=filename
        #selectedPairfileIdx= np.random.randint(low=0, high=len(self.allPairsByFileEval), size=(self.config.diff_dict_per_batch,))

        #key = list(self.allPairsByFileEval.keys())[selectedPairfileIdx[i]]
        batchName = key
        batch[batchName]={}
        selectedPairs = range(0,self.file2length[key])

        for lang in self.allPairsByFileEval[key]:
            batch[batchName][lang]=[]
            for pairIdx in selectedPairs:
                batch[batchName][lang].append(self.allPairsByFileEval[key][lang][pairIdx])


        return batch

    #    def getEvalPairIterator(self):
    #        for file in self.allPairsByFileEval:
    #            langs = []
    #            assert len(self.allPairsByFileEval[file])
    #            for lang in self.allPairsByFileEval[file]:
    #                langs=

    def saveWordPairs(self,folder):
        create_dirs([folder])
        for filename in self.allPairsByFile:
            lang1,lang2= [lang for lang in self.allPairsByFile[filename]]

            filepath= folder+os.path.basename(filename)+".txt"
            filepath= filepath.replace(".txt.txt",".txt")
            with open(filepath,"w") as f:
                for i in range(self.file2length[filename]):
                    token1 = self.allPairsByFile[filename][lang1][i]
                    token2 = self.allPairsByFile[filename][lang2][i]
                    f.write("{} {}\n".format(token1,token2))
    def loadAllWordPairsInFolder(self,folder):
        #for file in os.listdir("/mydir"):
        #    if file.endswith(".txt"):
        #        pass
        pass

    def finalize(self):
        pass
