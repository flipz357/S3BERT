import numpy as np
from scipy.spatial.distance import cosine


# cosine helper function
def cosine_sim(mat1, mat2):
    prod = mat1 * mat2
    normf = lambda x: np.sqrt(np.sum(x**2, axis=1))
    normx, normy = normf(mat1), normf(mat2)
    return np.sum(prod, axis=1) / (normx * normy)


# mh sim helper function 
def manhattan_sim(mat1, mat2):
    diffs = np.sum(np.abs(mat1 - mat2), axis = 1)
    return 1 - diffs

def get_preds(xsent_encoded, ysent_encoded, globalsim=cosine_sim, subsim=cosine_sim, biases=None, n=15, dim=16):
    """get predictions of model for paired sent vecs

    Params:
        xsent_encoded: matrix of sent vecs
        ysent_encoded: matrix of sent vecs
        biases: score bias coefficients for metrics
        n: number of metrics that are modeled (besides residual)
        dim: feature dimension 

    Returns:
        Matrix with predictions, e.g., [[0.2, 0.3, 0.1, 0.4],
                                        [0.7, ...]
                                        ...]] indicates
        that n=2, and in the first sentence pair 
        0.3 and 0.1 are metric scores derived 
        from decomposed features while 0.2 is global sbert similarity
        derived from full vectors and 0.4 is residual similarity 
    """

    if biases is None:
        biases = np.ones(n)

    #collect preds
    pred = []

    # cosine helper function
    def cosine_sim(mat1, mat2):
        prod = mat1 * mat2
        normf = lambda x: np.sqrt(np.sum(x**2, axis=1))
        normx, normy = normf(mat1), normf(mat2)
        return np.sum(prod, axis=1) / (normx * normy)
    
    # global sbert sims
    simsglobal = globalsim(xsent_encoded, ysent_encoded)
    simsresidual = globalsim(xsent_encoded[:,dim*n:], ysent_encoded[:,dim*n:])
    
    metric_features = []
    for i in range(0, dim*n, dim):    
        metric_features.append((xsent_encoded[:,i:i+dim], ysent_encoded[:,i:i+dim]))
    
    metric_sims = []
    for i in range(n):
        xfea = metric_features[i][0]
        yfea = metric_features[i][1]
        simfea = subsim(xfea, yfea)
        metric_sims.append(simfea)
    
    metric_sims = np.array(metric_sims) * biases[:,np.newaxis]
    metric_sims = metric_sims.T

    preds = np.concatenate((simsglobal[:,np.newaxis], metric_sims, simsresidual[:,np.newaxis]), axis=1)
    return preds
