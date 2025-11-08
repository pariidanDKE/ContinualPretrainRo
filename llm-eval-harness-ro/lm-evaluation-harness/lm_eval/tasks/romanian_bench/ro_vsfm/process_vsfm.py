
import sys

def doc_to_text(doc):
    text = "Întrebare: " + doc["question"] + "\nVariante:\n"
    text += "A. " + doc["answers"][0] + "\nB. " + doc["answers"][1] + "\nC. " + doc["answers"][2] + "\nD. " + doc["answers"][3] +"\n\n"
    text += "Răspuns:"
    return text

def doc_to_choice(doc): 
    return ["A", "B", "C", "D"]


def doc_to_target(doc):
    return chr(ord("A") + doc["answers"].index(doc["correct_answer"]))
    
    