from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from sentence_transformers.util import batch_to_device
import os
import csv


logger = logging.getLogger(__name__)


class DistilConsistencyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset
    This requires a model with LossFunction.SOFTMAX
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", loss_model_distil = None, loss_model_consistency = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset
        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.loss_model_distil = loss_model_distil
        self.loss_model_consistency = loss_model_consistency

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "loss distill", "loss consistency", "biases"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        
        model.eval()
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        sum_mse_c = 0.0
        sum_mse_d = 0.0
        bxs = 0
        for _, batch in enumerate(self.dataloader):
            features, labels = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            labels = labels.to(model.device)
            
            
            with torch.no_grad():
                mse = self.loss_model_distil(features, labels=labels)
            sum_mse_d += mse
            with torch.no_grad():
                mse = self.loss_model_consistency(features, labels=labels)
            sum_mse_c += mse
            bxs += 1
         
        accuracy = (sum_mse_d + sum_mse_c) / bxs
        accuracy = 1 - accuracy
        biases = list(self.loss_model_distil.score_bias.detach().cpu().numpy())

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy, sum_mse_d, sum_mse_c, biases])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow([epoch, steps, accuracy, sum_mse_d, sum_mse_c, biases])

        return accuracy
