from speaksee.data import ImageField, CocoImageAssociatedDetectionsField, TextField
from speaksee.data.dataset import COCOEntities
import torch
import random
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='entities_attention', type=str, help='experiment name')
parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--step_size', default=3, type=int, help='learning rate schedule step size')
parser.add_argument('--gamma', default=0.8, type=float, help='learning rate schedule gamma')
parser.add_argument('--image_features', action='store_true', help='for using global image descriptor instead of detection mean')
parser.add_argument('--no_padding', action='store_true', help='padding removing')
parser.add_argument('--sentinel_loss_weight', default=1.0, type=float, help='weight for the sentinel loss')
parser.add_argument('--rnn_det_next_size', default=2048, type=int, help='output size of rnn detection next')
parser.add_argument('--mask_det_next', action='store_true', help='det next and sentinel masking')

parser.add_argument('--h2_fist_lstm', default=1, type=int, help='h2 as input to the first lstm')
parser.add_argument('--det_next_second_lstm', default=1, type=int, help='det_next as input to the second lstm')
parser.add_argument('--img_second_lstm', default=0, type=int, help='img vector as input to the second lstm')

opt = parser.parse_args()
print(opt)

random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda')
exp_name = opt.exp_name

writer = SummaryWriter(log_dir='runs/%s' % exp_name)

# if not os.path.isfile('/tmp/coco_det_feats_loc.hdf5'):
#     copyfile('/nas/houston/lorenzo/coco_det_feats_loc.hdf5', '/tmp/coco_det_feats_loc.hdf5')
# if not os.path.isfile('/tmp/fc2k_coco.hdf5'):
#     copyfile('/nas/houston/lorenzo/fc2k_coco.hdf5', '/tmp/fc2k_coco.hdf5')

image_field = ImageField(precomp_path='/tmp/fc2k_coco.hdf5')
det_field = CocoImageAssociatedDetectionsField(detections_path='/tmp/coco_det_feats_loc.hdf5',
                                               classes_path='/homes/lbaraldi/bottom-up-attention/data/genome/1600-400-20/objects_vocab.txt')
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True)

dataset = COCOEntities(image_field, det_field, text_field,
                       img_root='/tmp/coco/images/',
                       ann_root='/nas/houston/lorenzo/vse/data/coco/annotations',
                       entities_file='/nas/houston/mcornia/coco-entities/coco_entities.json',
                       id_root='/nas/houston/lorenzo/vse/data/coco/annotations')
train_dataset, val_dataset, test_dataset = dataset.splits
text_field.build_vocab(train_dataset, val_dataset, min_freq=5)

from speaksee.models import EntitiesAttentionImproved, EntitiesAttentionFixed
model = EntitiesAttentionFixed(len(text_field.vocab), h2_fist_lstm=opt.h2_fist_lstm,
                               det_next_second_lstm=opt.det_next_second_lstm, img_second_lstm=opt.img_second_lstm).to(device)

from speaksee.data import DataLoader
import numpy as np
from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer

dataloader_train = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
dataloader_val = DataLoader(val_dataset, batch_size=10, num_workers=8)

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn import NLLLoss
optim = Adam(model.parameters(), lr=opt.lr)
scheduler = StepLR(optim, step_size=opt.step_size, gamma=opt.gamma)
if opt.no_padding:
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
else:
    loss_fn = NLLLoss()

loss_fn_att = NLLLoss()

start_epoch = opt.start_epoch
best_cider = .0
patience = 0
if start_epoch > 0:
    saved_data = torch.load('saved_models/%s_epoch_%03d.pth' % (exp_name, start_epoch - 1))
    print("Loading from epoch %d, with validation loss %.02f and validation CIDER %.02f" %
          (saved_data['epoch'], saved_data['val_loss'], saved_data['val_cider']))
    model.load_state_dict(saved_data['state_dict'])
    optim.load_state_dict(saved_data['optimizer'])
    scheduler.load_state_dict(saved_data['scheduler'])
    patience = saved_data['patience']
    best_cider = saved_data['best_cider']
    model.ss_prob = saved_data['ss_prob']
    if patience == 5:
        print('patience ended')
        exit()

if True:
    import math
    for e in range(start_epoch, 50):
        # Training
        model.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(iter(dataloader_train))) as pbar:
            for it, (images, detections, det_ids, captions) in enumerate(iter(dataloader_train)):
                if opt.image_features:
                    detections, images, captions, det_ids = detections.to(device), images.to(device), captions.to(device), det_ids.to(device)
                    out, att_out = model(detections, images, captions, det_ids)
                else:
                    detections, captions, det_ids = detections.to(device), captions.to(device), det_ids.to(device)
                    out, att_out = model(detections, None, captions, det_ids)

                optim.zero_grad()

                captions = captions[:, 1:].contiguous()
                out = out.contiguous()
                loss_cap = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                loss_att = loss_fn_att(att_out.view(-1, 2), (det_ids[:, 1:] == 0).long().view(-1))
                loss = loss_cap + opt.sentinel_loss_weight * loss_att

                # print(it, loss.item())
                if math.isnan(loss.item()):
                    exit()

                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (it+1))
                pbar.update()

        writer.add_scalar('data/train_loss', running_loss / len(dataloader_train), e)
        scheduler.step()

        # Validation loss
        model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - val' % e, unit='it', total=len(iter(dataloader_val))) as pbar:
            with torch.no_grad():
                for it, (images, detections, det_ids, captions) in enumerate(iter(dataloader_val)):
                    if opt.image_features:
                        detections, images, captions, det_ids = detections.to(device), images.to(device), captions.to(
                            device), det_ids.to(device)
                        out, att_out = model(detections, images, captions, det_ids)
                    else:
                        detections, captions, det_ids = detections.to(device), captions.to(device), det_ids.to(device)
                        out, att_out = model(detections, None, captions, det_ids)

                    captions = captions[:, 1:].contiguous()
                    out = out.contiguous()
                    loss_cap = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                    loss_att = loss_fn_att(att_out.view(-1, 2), (det_ids[:, 1:] == 0).long().view(-1))
                    loss = loss_cap + loss_att

                    running_loss += loss.item()
                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        writer.add_scalar('data/val_loss', running_loss / len(dataloader_val), e)

        # Validation with CIDEr
        predictions = []
        gt_captions = []
        max_len = 100
        with tqdm(desc='Test', unit='it', total=len(iter(dataloader_val))) as pbar:
            with torch.no_grad():
                for it, (images, detections, det_ids, captions) in enumerate(iter(dataloader_val)):
                    if opt.image_features:
                        detections, images, captions, det_ids = detections.to(device), images.to(device), captions.to(
                            device), det_ids.to(device)
                        out = model.test(detections, images, det_ids, max_len, text_field.vocab.stoi['<bos>'])
                    else:
                        detections, captions, det_ids = detections.to(device), captions.to(device), det_ids.to(device)
                        out = model.test(detections, None, det_ids, max_len, text_field.vocab.stoi['<bos>'])

                    predictions.append(out.data.cpu().numpy())

                    captions_pad = np.zeros((captions.size(0), max_len - 1))
                    captions_pad[:, :captions.size(1) - 1] = captions[:, 1:].data.cpu().numpy()
                    gt_captions.append(captions_pad)
                    pbar.update()

        predictions = np.concatenate(predictions, axis=0)
        gt_captions = np.concatenate(gt_captions, axis=0)

        gen = {}
        gts = {}
        for i, (cap1, cap2) in enumerate(zip(gt_captions, predictions)):
            gt_cap = []
            for w in cap1:
                word = text_field.vocab.itos[int(w)]
                if word == '<eos>':
                    break
                gt_cap.append(word)

            pred_cap = []
            for w in cap2:
                word = text_field.vocab.itos[int(w)]
                if word == '<eos>':
                    break
                pred_cap.append(word)

            gts[i] = [' '.join(gt_cap)]
            gen[i] = [' '.join(pred_cap)]
            if i <= 10:
                print(gts[i], gen[i])

        gts = PTBTokenizer.tokenize(gts)
        gen = PTBTokenizer.tokenize(gen)

        val_bleu, _ = Bleu(n=4).compute_score(gts, gen)
        method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
        for metric, score in zip(method, val_bleu):
            writer.add_scalar('data/val_%s' % metric, score, e)
            print(metric, score)

        val_meteor, _ = Meteor().compute_score(gts, gen)
        writer.add_scalar('data/val_meteor', val_meteor, e)
        print('METEOR', val_meteor)

        val_rouge, _ = Rouge().compute_score(gts, gen)
        writer.add_scalar('data/val_rouge', val_rouge, e)
        print('ROUGE_L', val_rouge)

        val_cider, _ = Cider().compute_score(gts, gen)
        writer.add_scalar('data/val_cider', val_cider, e)
        print('CIDEr', val_cider)

        # val_spice, _ = Spice().compute_score(gts, gen)
        # print('SPICE', val_spice)

        # Serialize model
        val_loss = running_loss / len(iter(dataloader_val))

        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            torch.save({
                'epoch': e,
                'opt': opt,
                'val_loss': val_loss,
                'val_cider': val_cider,
                'patience': patience,
                'best_cider': best_cider,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ss_prob': model.ss_prob,
            }, 'saved_models/%s_best.pth' % exp_name)
        else:
            patience += 1

        torch.save({
            'epoch': e,
            'opt': opt,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'patience': patience,
            'best_cider': best_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ss_prob': model.ss_prob,
        }, 'saved_models/%s_epoch_%03d.pth' % (exp_name, e))

        if patience == 5:
            break