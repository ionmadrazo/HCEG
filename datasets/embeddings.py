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

class EmbeddingDataLoader:
    def __init__(self, config, languages=None):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.logger.info("Initializing EmbeddingDataLoader...")

        create_dirs([self.config.embedding_folder])

        langs2Load= []
        if languages is None:
            langs2Load=  self.config.languages
        else:
            langs2Load=languages

        for lang in langs2Load:


            url=self.config.embedding_download_url.format(lang)
            path=self.config.embedding_folder+"/"+self.config.embedding_file.format(lang)
            pathPortion="{}_{}.pt".format(path,self.config.embedding_max)
            if not os.path.isfile(path) and not  os.path.isfile(pathPortion):
                self.logger.info("Download embeddings for lang={} ".format(lang))

                wget.download(url, path)
            else:
                self.logger.info("Embeddings for lang={} found.".format(lang))

        self.vectors = {}
        for lang in langs2Load:
            if "embedding_max" in self.config and self.config.embedding_max>0:

                self.vectors[lang] = Vectors(name=self.config.embedding_file.format(lang), cache=self.config.embedding_folder, max_vectors=self.config.embedding_max)
            else:
                self.vectors[lang] = Vectors(name=self.config.embedding_file.format(lang), cache=self.config.embedding_folder)


        self.logger.info("SingleLanguageDataLoader initialized.")


    def finalize(self):
        pass
