import speaksee.evaluation as evaluation


class TestMetrics(object):
    def test_meteor(self):
        metric = evaluation.Meteor()
        gts = {'0': ['these include activities linked to energy and, in particular, energy efficiency.']}
        gen = {'0': ['these are the activities related to energy, and in particular to energy efficiency.']}
        scores = metric.compute_score(gts, gen)[1]
        assert round(scores[0], 3) == 0.440
