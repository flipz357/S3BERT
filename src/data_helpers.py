import json
import numpy as np
import config

def jload(path):
    """Load json file"""
    with open(path, "r") as f:
        return json.load(f)


def jsave(dat, path):
    """Save data as json file"""
    with open(path, "w") as f:
        return json.dump(dat, f)


def load_example_data(path, n=None, indicator="pos"):
    
    # load amr json data with amr triples and metric scores
    with open(path, "r") as f:
        dat = f.read().split("\n\n")
    
    samples = []

    # decide if we load positive pairs (similar) 
    # or negative pairs (not similar) 
    if indicator == "pos":
        kx = 1
    elif indicator == "neg":
        kx = 2
    
    # iterate over triples
    for i in range(0, len(dat)-2, 3):

        # get first and second data instance
        samr = dat[i]
        tamr = dat[i+kx]

        # extract and clean sentence strings
        ssnt_pure = samr.split("# ::snt ")[-1].split("\n", 1)[0]
        tsnt_pure = tamr.split("# ::snt ")[-1].split("\n", 1)[0]
        samr_pure = samr.split("# ::")[-1].split("\n", 1)[1].replace("\n", " ")
        tamr_pure = tamr.split("# ::")[-1].split("\n", 1)[1].replace("\n", " ")
        samr_pure = " ".join(samr_pure.split())
        tamr_pure = " ".join(tamr_pure.split())

        # extract AMR metric scores of global metrics
        target_smatch = json.loads(samr.split("::y_{}".format(indicator))[1].split("\n")[0])
        target_stats = json.loads(samr.split("::y_other_{}".format(indicator))[1].split("\n")[0])
        target_wlk = json.loads(samr.split("::y_wl_{}".format(indicator))[1].split("\n")[0])
        target_wwlk = json.loads(samr.split("::y_wwl_{}".format(indicator))[1].split("\n")[0])
        target_wlk["score_wlk"] = target_wlk["score"]
        target_wlk["score_wwlk"] = target_wwlk["score"]
        
        # extract fine grained scores (coref, SRL, etc.)
        feature_score = np.zeros(len(config.FEATURES) - 2)
        for j, fea in enumerate(config.FEATURES[2:]):
            if fea in target_smatch:
                feature_score[j] = target_smatch[fea][2] # index 2 for f1
            if fea in target_stats:
                feature_score[j] = target_stats[fea] 
            if fea in target_wlk:
                feature_score[j] = target_wlk[fea] 
            if fea in target_wwlk:
                feature_score[j] = target_wwlk[fea] 
        
        # build one trainig example and add to data
        # (sent1, sent2, Semantic metric scores)
        sample = [ssnt_pure, tsnt_pure, feature_score.tolist()]
        samples.append(sample)
        
        # in case we want to retrieve not the full data
        if n and i > n:
            return samples
        if i % 1000 == 0:
            print("data loaded:", i, len(dat))

    return samples


def norm_scores(samples):

    scores = [x[2] for x in samples]
    scores = np.array(scores)
    for i in range(scores.shape[1]):
        mi = scores[:,i].min()
        ma = scores[:,i].max()
        scores[:,i] = (scores[:,i] - mi) / (ma - mi)
    
    for i in range(len(samples)):
        samples[i][2] = scores[i].tolist()
    
    return None


def load_example_data_full(include_neg=True):
    
    # load similar pairs and attached AMR metric scores
    trains = load_example_data(config.DATA_PATH + "/amr-triples.train")
    devs = load_example_data(config.DATA_PATH + "/amr-triples.dev")
    tests = load_example_data(config.DATA_PATH + "/amr-triples.test")
    
    if include_neg == True:
        # include negative examples
        trains += load_example_data(config.DATA_PATH + "/amr-triples.train", indicator="neg")
        devs += load_example_data(config.DATA_PATH + "/amr-triples.dev", indicator="neg")
        tests += load_example_data(config.DATA_PATH + "/amr-triples.test", indicator="neg")

    norm_scores(trains)
    norm_scores(devs)
    norm_scores(tests)
    
    return trains, devs, tests


def load_example_data_small(include_neg=True):
    
    # load similar pairs and attached AMR metric scores
    trains = load_example_data(config.DATA_PATH + "/amr-triples.train", n=5000)
    devs = load_example_data(config.DATA_PATH + "/amr-triples.dev", n=5000)
    tests = load_example_data(config.DATA_PATH + "/amr-triples.test", n=5000)
    
    if include_neg == True:
        # include negative examples
        trains += load_example_data(config.DATA_PATH + "/amr-triples.train", indicator="neg", n=5000)
        devs += load_example_data(config.DATA_PATH + "/amr-triples.dev", indicator="neg", n=5000)
        tests += load_example_data(config.DATA_PATH + "/amr-triples.test", indicator="neg", n=5000)

    norm_scores(trains)
    norm_scores(devs)
    norm_scores(tests)
    
    return trains, devs, tests


def test():
    p = "../amr_data_set/amr-triples.dev"
    dat = load_example_data(p, indicator="neg", n=10000)
    print(dat[:10])
    norm_scores(dat)

if __name__ == "__main__":
    test()
