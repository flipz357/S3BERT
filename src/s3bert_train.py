import os
import custom_losses
import custom_evaluators as evaluation
import torch
from sentence_transformers import SentenceTransformer, InputExample
import data_helpers as dh
import config

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print(torch.cuda.device_count())
print("Device: ", device)


#--------------
# 1. Model init
#--------------
#--------------
# we need an SBERT model that we can fine-tune
# and another one which we freeze
#--------------
model = SentenceTransformer(config.SBERT_INIT, device="cuda")

# init model for consistency, parameters are frozen 
teacher = SentenceTransformer(config.SBERT_INIT, device="cuda")

freeze.freeze_except_last_layers(model, 2)
freeze.freeze_all_layers(teacher)


#----------------
# 2. Data loading
#----------------
#----------------
# if not preprocessed data available, preprocess and save else just load
# ---------------
# trainx, devx and testx are lists that contain the data samples
# one data example: (sentenceA, sentenceB, list_with_scores)
# list_with_scores teaches the partioning
# in our case list_with_scores contain AMR metric scores, but this can be any metric you want
# --------------

if not "trainx.json" in os.listdir(config.PREPRO_PATH):
    trainx, devx, testx = dh.load_example_data_full()
    dh.jsave(trainx, config.PREPRO_PATH + "/trainx.json")
    dh.jsave(devx, config.PREPRO_PATH + "/devx.json")
    dh.jsave(testx, config.PREPRO_PATH + "/testx.json")
else:
    trainx = dh.jload(config.PREPRO_PATH + "/trainx.json")
    devx = dh.jload(config.PREPRO_PATH + "/devx.json")


# build training input
train_examples = []

for sample in trainx:
    ex = InputExample(texts=[sample[0], sample[1]], label=sample[2])
    train_examples.append(ex)

# build developement input
dev_examples = []

for sample in devx:
    ex = InputExample(texts=[sample[0], sample[1]], label=sample[2])
    dev_examples.append(ex)


#-------------------
# 3. Training S3BERT
#-------------------

#Define dataloader 
train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=config.BATCH_SIZE)
dev_dataloader = torch.utils.data.DataLoader(dev_examples, shuffle=False, batch_size=config.BATCH_SIZE)

# init losses
distill_loss = custom_losses.DistilLoss(model
                                        , sentence_embedding_dimension=model.get_sentence_embedding_dimension()
                                        , num_labels=config.N
                                        , feature_dim=config.FEATURE_DIM
                                        , bias_inits=None)

teacher_loss = custom_losses.MultipleConsistencyLoss(model, teacher)

# init evaluator
evaluator = evaluation.DistilConsistencyEvaluator(dev_dataloader
                                                    , loss_model_distil=distill_loss
                                                    , loss_model_consistency=teacher_loss)

#Tune the model
model.fit(train_objectives=[(train_dataloader, teacher_loss), (train_dataloader, distill_loss)]
                            , optimizer_params={'lr': config.LEARNING_RATE}
                            , epochs=config.EPOCHS
                            , warmup_steps=config.WARMUP_STEPS
                            , evaluator=evaluator
                            , evaluation_steps=config.EVAL_STEPS
                            , output_path=config.SBERT_SAVE_PATH
                            , save_best_model=True)
