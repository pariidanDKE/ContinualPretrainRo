def process_results(doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        predictions = {"id": doc["id"], "prediction_text": results[0],}

        references = {"id": doc["id"], "answers": doc["answers"],}

        return {
            "squad_em": (predictions, references,),  # Exact match (the normalized answer exactly match the gold answer)
            "squad_f1": (predictions, references,),  # The F-score of predicted tokens versus the gold answer
        }

def doc_to_target(doc):
    answer_list = doc["answers"]["text"]
    if len(answer_list) > 0:
        answer = answer_list[0]
    return " " + answer