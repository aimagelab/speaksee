import speaksee.evaluation as evaluation


class TestMetrics(object):
    def test_meteor(self):
        metric = evaluation.Meteor()
        gts = {'0': ['these include activities linked to energy and, in particular, energy efficiency.'],
               '1': ['then, various videos show us how to properly perform our workout plan.'],
               '2': ['audatex invests 90 million euros a year in developing these databases.']}
        gen = {'0': ['these are the activities related to energy, and in particular to energy efficiency.'],
               '1': ['several videos show us how carried out correctly our programme exercises.'],
               '2': ['audatex invested each year, 90 million to develop these databases.']}
        scores = metric.compute_score(gts, gen)[1]
        assert round(scores[0], 3) == 0.440
        assert round(scores[1], 3) == 0.281
        assert round(scores[2], 3) == 0.393