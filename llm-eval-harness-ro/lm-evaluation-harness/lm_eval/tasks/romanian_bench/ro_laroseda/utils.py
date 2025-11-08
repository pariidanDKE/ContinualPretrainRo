import datasets
import evaluate
from lm_eval.api.metrics import f1_score as f1
from lm_eval.api.registry import register_metric, register_aggregation
def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["rating"] = "pozitivă" if doc["starRating"] > 3 else "negativă" 
        return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

def doc_to_target_bc(doc):
    if doc["rating"] == "pozitivă":
        return 1
    return 0

def doc_to_target_mc(doc):
    return [1, 2, 4, 5].index(doc["starRating"])

def doc_to_target_bc_gen(doc):
    return doc["rating"]

def doc_to_target_mc_gen(doc):
    return str(doc["starRating"])