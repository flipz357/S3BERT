# S<sup>3</sup>BERT: *S*emantically *S*tructured *S*entence Embeddings

Code for generating and training sentence embeddings with semantic features. Two main goals:

- increase interpretability of sentence embeddings and explain similarity 
- effective aspectual clustering and semantic search 

For more information, background and demonstration, please check our [AACL paper](https://arxiv.org/abs/2206.07023).

## Requirements

Please make sure to have at least the following packages installed:

```
package                 (version tested)
----------------------------------------
torch                           (1.11.0)
transformers                    (4.16.1)
sentence-transformers           (2.1.0)
numpy                           (1.21.2)                         
scipy                           (1.7.3)        
[python                         (3.8.12)]
```

## The basic idea (how to customize)

The basic idea is simple: 

1. Define/apply metrics that measure similarity with regard to aspects or topics that you're interested in. 
2. Assign a specific sub-embedding to each metric
3. During training, it learns to route information into the assigned sub-embeddings so that they can reflect your metrics of interest. The power of the overall embedding is preserved with consistency control.
4. In inference, you are told how the aspects have modulated overall text similarity decision. 

Note that any (possibly costly) computation of metrics from step 1. is **not needed in inference** 

**Rule of thumb for size of feature dimensions**: From experience with different models that use 15 similarity aspect metrics, about 1/3 of the embedding may be reserved for the residual.

- `edim`: size of sentence embedding
- `n`: number of custom metrics
- `feadim`: size of a sentence feature (sentence sub-embedding)

Then `feadim` can be set approximately to `(edim - edim / 3)/n`. 

## Full example with AMR 

In our paper, we define metrics between abstract meaning representations (AMRs) such that we can measure, e.g., coreference or quantification similarity of sentences and see how these sub-similarities modulate the overall similarity. 

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

## Pretrained models and scores:

### Model Download

We provide pre-trained model here:

| Model name               | model link | s3bert config |
| ------------------------ | ---------- | ------------- |
| s3bert_all-mpnet-base-v2 | [model](https://www.cl.uni-heidelberg.de/~opitz/data/s3bert_all-mpnet-base-v2.tar.gz)  | [config](https://www.cl.uni-heidelberg.de/~opitz/data/config_s3bert_all-mpnet-base-v2.py)    |
| s3bert_all-MiniLM-L12-v2        | [model](https://www.cl.uni-heidelberg.de/~opitz/data/s3bert_all-MiniLM-L12-v2.tar.gz)  | [config](https://www.cl.uni-heidelberg.de/~opitz/data/config_s3bert_all-MiniLM-L12-v2.py)    |

Downloaded S3BERT models may be unpacked in src

```
tar -xvzf s3bert_all-MiniLM-L12-v2 -C src/
```

Use pre-trained model: See above (S3BERT embeddings: inference). Use specific `config.py` (see table above), which is needed so that we know which features are assigned a particular metric.

### Scores of pre-trained models

#### Table

All numbers are Spearmanr.

| Model | STSB | SICKR | UKPASPECT | Concepts  | Frames  | Named Ent.  | Negations  | Coreference  | SRL  | Smatch  | Unlabeled  | max_indegree_sim | max_outdegree_sim | max_degree_sim | root_sim | quant_sim | score_wlk | score_wwlk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s3bert_all-mpnet-base-v2 | 83.5     | **81.1** | **57.9** | **79.8** | **73.0** | **54.5** | **34.9** | **54.9** | **69.8** | **74.7** | **72.0** | **36.2** | **49.6** | **35.3** | **52.3** | **75.3** | **80.8** | **80.3** |
| all-mpnet-base-v2        | 83.4     | 80.5     | 56.2     | 74.3     | 41.5     | -12.7    | -0.3     | 9.0      | 42.8     | 57.6     | 52.1     | 23.6     | 21.1     | 17.7     | 22.9     | 10.8     | 68.3     | 66.6     |
| s3bert_all-MiniLM-L12-v2 | **83.7** | 78.9     | 56.6     | 74.3     | 66.3     | 51.0     | 33.4     | 44.1     | 61.4     | 67.5     | 65.1     | 31.9     | 42.4     | 29.5     | 43.6     | 73.6     | 74.6     | 74.2     |
| all-MiniLM-L12-v2        | 83.1     | 78.9     | 54.2     | 76.7     | 37.3     | -12.8    | -3.8     | 7.7      | 42.1     | 56.3     | 51.5     | 23.8     | 19.0     | 19.0     | 20.1    | 9.4       | 66.3     | 63.5     |

#### Table Column names I: basic similarity benchmarking

For both SBERT and S3BERT the similarity for every pair is calculated on the full embeddings (cosine).

- STSB: results on human sentence similarity benchmark STS 
- SICKR: results on human relatedness similarity benchmark SICK 
- UKPA: results on human argument similarity benchmark

#### Table Column names II: aspect similarity of explainable features

For non S3BERT models the aspect similarity is calculated via the full embedding (i.e., it gives the same similarity in every aspect). For S3BERT models the aspect similarities are calculated from the dedicated sub-embeddings.

- Concepts: Similarity w.r.t. to similarity of concepts in sentences
- Frames: Similarity w.r.t. to similarity of predicates in sentences 
- Named Ent: Similarity w.r.t. named entity similarities in sentences 
- Negation: Similarity w.r.t. negation structure of sentences 
- Coreference: Similarity w.r.t. coreference structure of sentences 
- SRL: Similarity w.r.t. semantic role structure of sentences 
- Smatch: Similarity w.r.t. to overall similarity of sentences' semantic meaning structures 
- Unlabeled: Similarity w.r.t. to overall similarity of sentences' semantic meaning structures minus relation label
- (in/out/root)_degree_sim: Similarity w.r.t. to similarity of connected nodes in meaning space ("Focus") 
- quant_sim: Similarity w.r.t.\ quantificational structure similarity of sentences(*three* vs. *four*, *a* vs. *all*, etc.) 
- score_wlk: see Smatch, but measured with contextual Weisfeiler Leman Kernel isntead of Smatch 
- score_wwlk: See Smatch, but measured with Wasserstein Weisfeiler Leman Kernel instead of Smatch 

## Citation

If you find the work interesting, consider citing:

```
@article{opitz2022sbert,
  title={SBERT studies Meaning Representations: Decomposing Sentence Embeddings into Explainable Semantic Features},
  author={Opitz, Juri and Frank, Anette},
  journal={arXiv preprint arXiv:2206.07023},
  year={2022}
}
```
