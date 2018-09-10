from speaksee.data import ImageField, CocoImageAssociatedDetectionsField, TextField, RawField
from speaksee.data.dataset import COCOEntities, DictionaryDataset
import torch
import random
import itertools
import argparse
from tqdm import tqdm

random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='entities_attention', type=str, help='experiment name')
parser.add_argument('--test_mod', default=0, type=int, help='0 for all GT captions, 1 for single GT captions')
opt_test = parser.parse_args()

# /tmp/fc2k_coco.hdf5
image_field = ImageField(precomp_path='/tmp/fc2k_coco.hdf5')
det_field = CocoImageAssociatedDetectionsField(detections_path='/tmp/coco_det_feats_loc.hdf5',
                                                classes_path='/homes/lbaraldi/bottom-up-attention/data/genome/1600-400-20/objects_vocab.txt')
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True)

dataset = COCOEntities(image_field, det_field, text_field,
                       img_root='/tmp/coco/images/',
                       ann_root='/nas/houston/lorenzo/vse/data/coco/annotations',
                       entities_file='/nas/houston/mcornia/coco-entities/coco_entities.json',
                       id_root='/nas/houston/lorenzo/vse/data/coco/annotations')

train_dataset, val_dataset, _ = dataset.splits
# text_field.build_vocab(dataset)
text_field.build_vocab(train_dataset, val_dataset, min_freq=5)

saved_data = torch.load('saved_models/%s' % '%s_best.pth' % opt_test.exp_name)
opt = saved_data['opt']

from speaksee.models import EntitiesAttentionImproved, EntitiesAttentionFixed
model = EntitiesAttentionFixed(len(text_field.vocab), h2_fist_lstm=opt.h2_fist_lstm,
                                  det_next_second_lstm=opt.det_next_second_lstm, img_second_lstm=opt.img_second_lstm).to(device)

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

model.eval()
model.load_state_dict(saved_data['state_dict'])


# Test with all reference captions for each sample
if opt_test.test_mod == 0:
    predictions = []
    gt_captions = []
    lengths = []

    with tqdm(desc='Test', unit='it', total=len(iter(dataloader_test))) as pbar:
        for it, (keys, values) in enumerate(iter(dataloader_test)):
            images = keys
            detections, det_ids, captions = values
            for i in range(images.size(0)):
                detections_i, images_i, det_ids_i = detections[i].to(device), images[i].to(device), det_ids[i].to(device)
                images_i = images_i.unsqueeze(0).expand(det_ids_i.size(0), -1)

                if opt.image_features:
                    out = model.test(detections_i, images_i, det_ids_i, 20, text_field.vocab.stoi['<bos>'])
                else:
                    out = model.test(detections_i, None, det_ids_i, 20, text_field.vocab.stoi['<bos>'])

                predictions.append(out.data.cpu().numpy())
                for _ in range(out.shape[0]):
                    gt_captions.append(captions[i])
                lengths.append(out.shape[0])
            pbar.update()

    predictions = np.concatenate(predictions, axis=0)

    gen = {}
    gts = {}

    for i, cap in enumerate(predictions):
        pred_cap = text_field.decode(cap)

        gts[i] = list(gt_captions[i])
        gen[i] = [' '.join(pred_cap)]

        # if i <= 10:
        #     print(gts[i], gen[i])

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


# Test with a single reference captions for each sample
if opt_test.test_mod == 1:
    predictions = []
    gt_captions = []
    with tqdm(desc='Test', unit='it', total=len(iter(dataloader_test))) as pbar:
        for it, (keys, values) in enumerate(iter(dataloader_test)):
            images = keys
            detections, det_ids, captions = values
            for i in range(images.size(0)):
                detections_i, images_i, det_ids_i = detections[i].to(device), images[i].to(device), det_ids[i].to(device)
                images_i = images_i.unsqueeze(0).expand(det_ids_i.size(0), -1)

                if opt.image_features:
                    out = model.test(detections_i, images_i, det_ids_i, 20, text_field.vocab.stoi['<bos>'])
                else:
                    out = model.test(detections_i, None, det_ids_i, 20, text_field.vocab.stoi['<bos>'])

                predictions.append(out.data.cpu().numpy())
                gt_captions.extend(captions[i])
            pbar.update()

    predictions = np.concatenate(predictions, axis=0)

    gen = {}
    gts = {}

    for i, cap in enumerate(predictions):
        pred_cap = text_field.decode(cap)

        gts[i] = [gt_captions[i]]
        gen[i] = [' '.join(pred_cap)]

        # if i <= 10:
        #     print(gts[i], gen[i])

    gts_t = PTBTokenizer.tokenize(gts)
    gen_t = PTBTokenizer.tokenize(gen)

    scores_dict = dict()

    val_bleu, scores = Bleu(n=4).compute_score(gts_t, gen_t)
    method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    for metric, score, ss in zip(method, val_bleu, scores):
        scores_dict[metric] = ss
        print(metric, score)

    val_meteor, _ = Meteor().compute_score(gts_t, gen)
    print('METEOR', val_meteor)

    val_rouge, scores = Rouge().compute_score(gts_t, gen_t)
    scores_dict['ROUGE_L'] = scores
    print('ROUGE_L', val_rouge)

    val_cider, scores = Cider().compute_score(gts_t, gen_t)
    scores_dict['CIDEr'] = scores
    print('CIDEr', val_cider)

    val_spice, _ = Spice().compute_score(gts_t, gen_t)
    scores_dict['SPICE'] = scores
    print('SPICE', val_spice)

    # f = open('coco_entities_results_single.txt', 'w')
    # for i in range(len(gen)):
    #     f.write('\n')
    #     f.write(str(gts[i][0]) + '\n')
    #     f.write(str(gen[i][0]) + '\n')
    #     for metric in ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr', 'SPICE']:
    #         f.write('%s: %s\n' % (metric, str(scores_dict[metric][i])))
    # f.close()
