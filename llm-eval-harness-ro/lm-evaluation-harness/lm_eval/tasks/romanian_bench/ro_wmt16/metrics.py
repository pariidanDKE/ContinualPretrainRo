import evaluate


def bleu(predictions, references):
    return (predictions[0], references[0])


def agg_bleu(items):
    bleu_fn = evaluate.load("bleu")
    predictions, references = zip(*items)
    return bleu_fn.compute(predictions=predictions, references=references)["bleu"]

def wer(predictions, references):
    return (predictions[0], references[0])

def agg_wer(items):
    wer_fn = evaluate.load("wer")
    predictions, references = zip(*items)
    return wer_fn.compute(predictions=predictions, references=references)