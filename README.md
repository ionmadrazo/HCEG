# Hierarchical  Mapping for Cross-lingual  Word Embedding Alignment

HCEG is the result of a research collaboration between [Ion Madrazo Azpiazu](https://ionmadrazo.github.io/) and [Maria Soledad Pera](https://solepera.github.io/).

Please cite this work as follows:

```
    @article{madrazoHCEG:2020,
	author = {Ion Madrazo Azpiazu and Maria Soledad Pera},
	year = "2020",
	title = {Hierarchical  Mapping for Cross-lingual  Word Embedding Alignment},
	journal = {Transactions of the Association for Computational Linguistics},
	volume = "1",
	number = "1",
	pages = "1--1"
}
```
## Abstract
The alignment of word embedding spaces in different languages into a common cross-lingual space has recently been in vogue. Strategies that do so compute pairwise alignments and then map multiple languages to a single pivot language (most often English). These strategies, however, are biased towards the choice of the pivot language, given that language proximity and the linguistic characteristics of the target language can strongly impact the resultant cross-lingual space in detriment of topologically distant languages. We present a strategy that eliminates the need for a pivot language by learning the mappings across languages in a hierarchical way. Experiments demonstrate that our strategy significantly improves vocabulary induction scores in all existing benchmarks, as well as in a new non-English centered benchmark we built, which we make publicly available. 

![Example tree decription](https://github.com/ionmadrazo/HCEG/blob/master/LanguageTree.png)


## Requirements
In order to run this code you will first need to:

- [Python 3.7](https://www.python.org/downloads/)
- [Anaconda 4.5.11](https://www.anaconda.com/download/)
- [Pytorch 1.0.0](https://pytorch.org/)


In addition, you will also need to install libraries described in requirements.txt file located on the root directory of this repository. You can install the by running the following command:
```
pip install -r requirements.txt

### Run
To run an experiment use the following command:

```
python main.py configs/<config file name>
```

## License

This software has a free to use license for non commercial use. Check **LICENSE** file for more details.
