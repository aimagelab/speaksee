from speaksee.data import ImageField, CocoImageAssociatedDetectionsField, CocoMultipleAssociatedDetectionsField, TextField, RawField
from speaksee.data.dataset import COCOEntities, DictionaryDataset
import torch
import random
import itertools
import argparse
from tqdm import tqdm

random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

# /tmp/fc2k_coco.hdf5
image_field = ImageField(precomp_path='/tmp/fc2k_coco.hdf5')
det_field = CocoImageAssociatedDetectionsField(detections_path='/tmp/coco_det_feats_loc.hdf5',
                                                classes_path='/homes/lbaraldi/bottom-up-attention/data/genome/1600-400-20/objects_vocab.txt')

test_dataset = COCOEntities(image_field, det_field, RawField(),
                            img_root='/tmp/coco/images/',
                            ann_root='/nas/houston/lorenzo/vse/data/coco/annotations',
                            entities_file='/nas/houston/mcornia/coco-entities/coco_entities.json',
                            id_root='/nas/houston/lorenzo/vse/data/coco/annotations')
_, _, test_dataset = test_dataset.splits
test_dataset = DictionaryDataset(test_dataset.examples, test_dataset.fields, 'image')

from speaksee.data import DataLoader
import numpy as np
from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer

dataloader_test = DataLoader(test_dataset, batch_size=100, num_workers=8)


# Test with all reference captions for each sample
if True:
    predictions = []
    gt_captions = []
    lengths = []

    with tqdm(desc='Test', unit='it', total=len(iter(dataloader_test))) as pbar:
        for it, (keys, values) in enumerate(iter(dataloader_test)):
            images = keys
            detections, det_ids, captions = values
            for i in range(images.size(0)):
                for k in range(len(captions[i])):
                    predictions.append(captions[i][k])
                    gt_captions.append(captions[i])
                lengths.append(len(captions[i]))
            pbar.update()

    gen = {}
    gts = {}

    for i in range(len(predictions)):
        gts[i] = list(gt_captions[i])
        gen[i] = [predictions[i]]

        if i <= 10:
            print(gts[i], gen[i])

    gts_t = PTBTokenizer.tokenize(gts)
    gen_t = PTBTokenizer.tokenize(gen)
    lengths = [0, ] + list(itertools.accumulate(lengths))

    scores_dict = dict()

    _, val_bleu = Bleu(n=4).compute_score(gts_t, gen_t)
    method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    for metric, scores in zip(method, val_bleu):
        scores_dict[metric] = scores
        scores = [scores[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]
        scores = [sum(s)/len(s) for s in scores]
        score = sum(scores) / len(scores)
        print(metric, score)

    _, scores = Meteor().compute_score(gts_t, gen_t)
    scores = [scores[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]
    scores = [sum(s) / len(s) for s in scores]
    val_meteor = sum(scores) / len(scores)
    print('METEOR', val_meteor)

    _, scores = Rouge().compute_score(gts_t, gen_t)
    scores_dict['ROUGE_L'] = scores
    scores = [scores[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]
    scores = [sum(s) / len(s) for s in scores]
    val_rouge = sum(scores) / len(scores)
    print('ROUGE_L', val_rouge)

    _, scores = Cider().compute_score(gts_t, gen_t)
    scores_dict['CIDEr'] = scores
    scores = [scores[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]
    scores = [sum(s) / len(s) for s in scores]
    val_cider = sum(scores) / len(scores)
    print('CIDEr', val_cider)

    _, scores = Spice().compute_score(gts_t, gen_t)
    scores_dict['SPICE'] = scores
    scores = [scores[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]
    scores = [sum(s) / len(s) for s in scores]
    val_spice = sum(scores) / len(scores)
    print('SPICE', val_spice)

    # f = open('coco_entities_results_all_train.txt', 'w')
    # for i in range(len(gen)):
    #     f.write('\n')
    #     f.write(str(gts[i]) + '\n')
    #     f.write(str(gen[i][0]) + '\n')
    #     for metric in ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr', 'SPICE']:
    #         f.write('%s: %s\n' % (metric, str(scores_dict[metric][i])))
    # f.close()

