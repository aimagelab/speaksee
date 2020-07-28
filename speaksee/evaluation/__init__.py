from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice import Spice
from .recall import recall
from .tokenizer import PTBTokenizer


def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Spice())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        if isinstance(score, list):
            for i, (sc, scs) in enumerate(zip(score, scores)):
                all_score[str(metric) + str(i)] = sc
                all_scores[str(metric) + str(i)] = scs
        else:
            all_score[str(metric)] = score
            all_scores[str(metric)] = scores

    return all_score, all_scores
