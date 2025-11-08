
import sys

def doc_to_text(doc):

    # temporary hack for dataset
    # also need to handle dummy answers
    if doc["Options"] == "{'a': 'substantivul pană;', 'b': 'substantivul poet;', 'c': prima propoziţie din frază;', 'd': 'substantivul 'memoria'}":
        doc["Options"] = "{'a': 'substantivul pană;', 'b': 'substantivul poet;', 'c': 'prima propoziţie din frază;', 'd': 'substantivul memoria'}"
    if doc["Options"] == "{'a': 'copulativă prin 'şi';', 'b': 'disjunctivă prin 'ori';', 'c': 'disjunctivă prin 'sau';', 'd': 'prin juxtapunere.'}":
        doc["Options"] = "{'a': 'copulativă prin şi;', 'b': 'disjunctivă prin ori;', 'c': 'disjunctivă prin sau;', 'd': 'prin juxtapunere.'}"
    if doc["Options"] == "{'a': 'verbul predicativ 'spune';', 'b': 'substantivul 'omule';', 'c': 'pronumele nehotărât 'tot', 'd': 'pronumele personal 'tu'}":
        doc["Options"] = "{'a': 'verbul predicativ spune;', 'b': 'substantivul omule;', 'c': 'pronumele nehotărât tot;', 'd': 'pronumele personal tu'}"
    # print(type(doc["Options"]), doc["Options"])
    if type(doc["Options"]) == str:
        doc["Options"] = eval(doc["Options"])
    
    text = "Întrebare: " + doc["Question"] + "\nVariante:\n"
    
    text += "A. " + doc["Options"]['a'] + "\nB. " + doc["Options"]['b'] + "\nC. " + doc["Options"]['c']
    if "d" in doc["Options"]:
        text += "\nD. " + doc["Options"]['d'] +"\n\n"
    text += "Răspuns:"
    return text

def doc_to_choice(doc): 
    if doc["Options"] == "{'a': 'substantivul pană;', 'b': 'substantivul poet;', 'c': prima propoziţie din frază;', 'd': 'substantivul 'memoria'}":
        doc["Options"] = "{'a': 'substantivul pană;', 'b': 'substantivul poet;', 'c': 'prima propoziţie din frază;', 'd': 'substantivul memoria'}"
    if doc["Options"] == "{'a': 'copulativă prin 'şi';', 'b': 'disjunctivă prin 'ori';', 'c': 'disjunctivă prin 'sau';', 'd': 'prin juxtapunere.'}":
        doc["Options"] = "{'a': 'copulativă prin şi;', 'b': 'disjunctivă prin ori;', 'c': 'disjunctivă prin sau;', 'd': 'prin juxtapunere.'}"
    if doc["Options"] == "{'a': 'verbul predicativ 'spune';', 'b': 'substantivul 'omule';', 'c': 'pronumele nehotărât 'tot', 'd': 'pronumele personal 'tu'}":
        doc["Options"] = "{'a': 'verbul predicativ spune;', 'b': 'substantivul omule;', 'c': 'pronumele nehotărât tot;', 'd': 'pronumele personal tu'}"

    if type(doc["Options"]) == str:
        doc["Options"] = eval(doc["Options"])
    if len(doc["Options"].keys()) == 3:
        return ["A", "B", "C"]
    return ["A", "B", "C", "D"]


def doc_to_target(doc):
    if doc["Correct_answer"] == "_":
        return "D"
    if doc["Correct_answer"] == "c_":
        return "C"
    return chr(ord("A") + list(doc["Options"].keys()).index(doc["Correct_answer"]))
    
    