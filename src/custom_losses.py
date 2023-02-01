import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer, util
import logging
import numpy as np
import model_freeze as freeze

logger = logging.getLogger(__name__)

def dist_sim(reps1: Tensor, reps2: Tensor):
    """ based on manhatten distance """
    diff = torch.abs(reps1 - reps2)
    sim = 1.0 - torch.sum(diff, dim=1)
    return sim

def prod_sim(reps1: Tensor, reps2: Tensor):
    """ dot product """
    diff = reps1 * reps2
    sim = torch.sum(diff, dim=1)
    return sim

def co_sim(reps1: Tensor, reps2: Tensor):
    """ cosinus similarity """
    sim = self.prod_sim(reps1, reps2)
    reps1_norm = torch.sum(reps1 ** 2, dim=1)
    reps2_norm = torch.sum(reps2 ** 2, dim=1)
    reps1_norm = torch.sqrt(reps1_norm)
    reps2_norm = torch.sqrt(reps2_norm)
    sim /= (reps1_norm * reps2_norm)
    return sim

class DistilLoss(nn.Module):
    """
    Distill metrics, Decompose output space

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of features / metric scores. 1 will be added as residual
    :param feature_dim: Dimension of a feature
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.MSELoss()
    :param sim_fct: Optional: Custom similarity function. If not set, uses Manhatten Sim
    Example::
        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=[0.3, 0.2, 0.9]),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=[0.4, 0.1, 0.2])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.DistilLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3, feature_dim=16)
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 feature_dim: int = 16,
                 bias_inits: np.array = None,
                 loss_fct: Callable = nn.MSELoss(),
                 sim_fct: Callable = dist_sim):
        super(DistilLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.sentence_embedding_dimension = sentence_embedding_dimension

        self.loss_fct = loss_fct
        self.sim_fct = sim_fct
        self.feature_dim = feature_dim
        self.residual_dim = sentence_embedding_dimension - self.feature_dim
        
        if bias_inits is None:
            biases = torch.ones(self.num_labels, requires_grad=True)
            self.score_bias = nn.Parameter(biases)
        else:
            
            biases = torch.tensor(bias_inits, requires_grad=True, dtype=torch.float32)
            self.score_bias = nn.Parameter(biases)

        self.score_bias.to(model._target_device)
    

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        
        sims = []
        for i in range(self.num_labels):
            
            # get two subembeddings
            start = i * self.feature_dim
            stop = (i+1) * self.feature_dim
            rep_ax = rep_a[:, start:stop]
            rep_bx = rep_b[:, start:stop]

            # and compute their similariy
            sim = self.sim_fct(rep_ax, rep_bx)
            sims.append(sim)
        
        # sims: (n_features x n_batch)
        # output: (n_batch x n_features)
        outputs = torch.stack(sims).T 
        outputs = self.score_bias * outputs
        
        if labels is not None:
            loss = self.loss_fct(outputs, labels)
            return loss 
        else:
            # not needed
            return reps, outputs
    
    def get_config_dict(self):
        return {'biases': self.score_bias}


class MultipleConsistencyLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        a learner and a teacher model. The loss computes a pairwise similarity matrix on the embeddings from the
        learner model A and for the teacher model B and tunes the Mean squared error (A - B)^2. I.e.,
        the learner is tuned to be consistend with the teacher.
    """
    def __init__(self, 
            model: SentenceTransformer, 
            teacher: SentenceTransformer,
            similarity_fct = util.cos_sim, 
            loss_fct: Callable = nn.MSELoss(),
            scale: float = 5.0):
        """
        :param model: SentenceTransformer model
        :param teacher: SentenceTransformer teacher, will be frozen
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :loss_fct: loss function
        :param scale: Output of similarity function is multiplied by scale value
        """
        super(MultipleConsistencyLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.loss_fct = loss_fct
        
        # freeze teacher
        self.teacher = teacher
        freeze.freeze_all_layers(self.teacher)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        
        teacher_reps = [self.teacher(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        teacher_embeddings_a = teacher_reps[0]
        teacher_embeddings_b = torch.cat(teacher_reps[1:])
        
        # intra model pairwise sims
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        
        # intra teacher pairwise sims
        teacher_scores = self.similarity_fct(teacher_embeddings_a, teacher_embeddings_b) * self.scale
        
        # (teacher_sim - model_sim)^2
        loss = self.loss_fct(scores, teacher_scores)
        return loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}

