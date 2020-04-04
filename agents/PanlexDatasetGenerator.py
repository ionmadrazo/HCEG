import numpy as np

from tqdm import tqdm
import shutil
import random
import json
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import random
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
from utils.panlex import panlexAPI
from utils.languageCodes import languageCodeTranslator
import time
import os

class PanlexDatasetGeneratorAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("Initializing PanlexDatasetGeneratorAgent...")
        self.config= config
        self.langTrans= langTrans=languageCodeTranslator("data/languageCodes/ISO-639-2_utf-8.txt")
        self.panlex= panlex= panlexAPI(self.langTrans)




        self.logger.info("PanlexDatasetGeneratorAgent initialized.")





    def run(self):
        """
        The main operator
        :return:
        """
        if "allCombinations" == self.config.languagePairs2Generate:
            for lang1 in self.config.languages:
                for lang2 in self.config.languages:
                    if lang1!=lang2:
                        srcLang, tgtLang= lang1,lang2
                        self.logger.info("Generating for language pair: {} {}".format(srcLang, tgtLang))
                        self.generateWordPairsForLangPair(srcLang,tgtLang)
                        self.logger.info("Finished generating for language pair: {} {}".format(srcLang, tgtLang))
        else:
            for langPair in self.config.languagePairs2Generate:
                srcLang, tgtLang= langPair.split(" ")
                self.logger.info("Generating for language pair: {} {}".format(srcLang, tgtLang))
                self.generateWordPairsForLangPair(srcLang,tgtLang)
                self.logger.info("Finished generating for language pair: {} {}".format(srcLang, tgtLang))

    def generateWordPairsForLangPair(self, srcLang, tgtLang):
        filename= "{}/{}-{}.{}.embmax{}.full.txt".format(self.config.store_folder,srcLang,tgtLang,self.config.sampleNumber,self.config.embedding_max)
        filenameTrain= "{}/{}-{}.{}.embmax{}.train.txt".format(self.config.store_folder,srcLang,tgtLang,self.config.sampleNumber,self.config.embedding_max)
        filenameTest= "{}/{}-{}.{}.embmax{}.test.txt".format(self.config.store_folder,srcLang,tgtLang,self.config.sampleNumber,self.config.embedding_max)
        if os.path.isfile(filename) and self.config.doNotCreateIfExists:
            self.logger.info("Word pair file already exists: {}".format(filename))
            return
        embedding_loader = globals()[self.config.embeddingDataLoader](config=self.config,languages=[srcLang])
        translatedPairs={}
        nonesInARow = 0
        itersWithoutImprovement=0
        lastLen=0
        while len(translatedPairs)<self.config.sampleNumber:
            selectedIdx = random.sample(range(0, len(embedding_loader.vectors[srcLang].itos)), self.config.numberOfsentElementsAPI)
            selectedTokens = [ embedding_loader.vectors[srcLang].itos[idx] for idx in selectedIdx]

            retrievedPairs= self.panlex.translateMultiple(srcLang,tgtLang,selectedTokens)
            if retrievedPairs is None:
                retrievedPairs={}
            for srcToken in retrievedPairs:
                tgtToken= retrievedPairs[srcToken]
                if " " in srcToken or " " in tgtToken:
                    continue

                if tgtToken is not None and len(translatedPairs)<self.config.sampleNumber and " " not in tgtToken :
                    translatedPairs[srcToken]=tgtToken
                    #print(srcLang,tgtLang,srcToken,tgtToken, len(translatedPairs))
            if lastLen== len(translatedPairs):
                itersWithoutImprovement=itersWithoutImprovement+1
            else:
                lastLen= len(translatedPairs)
                itersWithoutImprovement=0

            if itersWithoutImprovement>25:
                break
            self.logger.info("Generated {} pairs so far for language pair: {} {}".format(len(translatedPairs),srcLang, tgtLang ))

        with open(filename, "w", encoding="utf-8") as f, open(filenameTrain, "w", encoding="utf-8") as ftrain, open(filenameTest, "w", encoding="utf-8") as ftest:
            for srcToken in translatedPairs:
                tgtToken = translatedPairs[srcToken]
                f.write("{} {}\n".format(srcToken,tgtToken))
                if random.random()< self.config.split_ratio:
                    ftrain.write("{} {}\n".format(srcToken,tgtToken))
                else:
                    ftest.write("{} {}\n".format(srcToken,tgtToken))
        return translatedPairs


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
