# S<sup>3</sup>BERT: *S*emantically *S*tructured *S*entence Embeddings

Code for generating and training sentence embeddings with semantic features. Two main goals:

- increase interpretability of sentence embeddings
- effective aspectual clustering and semantic search (fast and accurate)

For more information and background, please check our [AACL-IJCAI 2022 paper](https://arxiv.org/abs/2206.07023).

## Requirements

Please make sure to have at least the following packages installed:

```
package                 (version tested)
----------------------------------------
sentence-transformers           (2.1.0)
numpy                           (1.21.2)          
python                          (3.8.12)                
scipy                           (1.7.3)        
sentence-transformers           (2.1.0)     
torch                           (1.11.0)
transformers                    (4.16.1)
```

## The basic idea (how to customize)

The basic idea is simple: 

1. Define/use metrics that measure sentence similarity with regard to different interesting aspects (e.g., using dependency tree, AMR, etc.).
2. Train S3BERT on pairs of sentences as inputs and semantic metric scores as targets (only pairs of sentences and a list with metric scores is needed as training data)
3. During training, it learns to partition the sentence vectors into features that express the different metrics

Note that (potentially) costly computation for semantic metrics or generation of sentence parses or trees is **not needed in inference**.

## Full example with AMR 

In our paper, we define metrics between abstract meaning representations (AMRs) such that we can measure, e.g., coreference or quantification similarity of AMR graphs. 

### Get our training data

The data contains the sentences and AMRs with AMR metric scores (note: we only need metric scores and sentences, the AMR graphs are attached only for potential further experimention)

Download and extract data:

```
wget https://cl.uni-heidelberg.de/~opitz/data/amr_data_set.tar.gz
tar -xvzf amr_data_set.tar.gz
```

This is how the format of the traing data should look

```
cd src/
python data_helpers.py
```

### S3BERT embeddings: Train to generate semantic partitioning

Simply run

```
cd src/
python s3bert_train.py
```

Some settings can be adjusted in `config.py`. For other settings, the source code must be consulted.

### S3BERT embeddings: inference

We have prepared an example script:

```
cd src/
python s3bert_infer.py
```

Check out its content for info on how to obtain and use the embeddings.

## Pretrained model:

We provide a pre-trained model here:

```
wget https://cl.uni-heidelberg.de/~opitz/data/s3bert_all-MiniLM-L12-v2.tar.gz
tar -xvzf s3bert_all-MiniLM-L12-v2 -C src/
```

Use pre-trained model: See above (S3BERT embeddings: inference). Set model name in `config.py`.












