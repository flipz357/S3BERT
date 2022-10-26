import ast
import numpy as np
from sentence_transformers import SentenceTransformer
import config
import prediction_helpers as ph
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# get decomposed model
model = SentenceTransformer("./" + config.SBERT_SAVE_PATH + "/", device="cuda")

# get bias coefficients for sub-metrics (ugly way to load them, todo: find better way)
fp = config.SBERT_SAVE_PATH + "/eval/accuracy_evaluation_results.csv"
with open(fp) as f:
    ll = f.read().split("\n")
losses = [float(l.split("tensor(")[1].split(",")[0]) for l in ll if "tensor" in l]
biasess = [ast.literal_eval(l.split(";")[-1]) for l in ll if "tensor" in l]
amin = np.argmin(losses)
biases = np.array(biasess[amin]) 

# example sentence pairs
xsent = [
        "the man isn't singing", "three man are singing", "two cats are looking at a window", 
        "a cat is looking at a window",
        "rocky and apollo creed are running down the beach", "a man is smoking"] 
ysent = [
        "the man is singing", "two men are singing", "a white cat looking out of a window", 
        "a cat is looking out of a window",
        "the men are jogging on the beach", "a baby is sucking on a pacifier"] 

# encode with s3bert
xsent_encoded = model.encode(xsent)
ysent_encoded = model.encode(ysent)

# get similarity scores of different features
preds = ph.get_preds(xsent_encoded, ysent_encoded, biases=None)

# print similarity scores of different features
features = ["global"] + config.FEATURES[2:] + ["residual"]
for i, x in enumerate(xsent):
    sims = preds[i]
    jl = {k:v for k,v in zip(features, sims)}
    jl["sent_a"] = x 
    jl["sent_b"] = ysent[i]
    print(jl)