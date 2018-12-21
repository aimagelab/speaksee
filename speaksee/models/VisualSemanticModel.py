from torch import nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VisualSemanticModel(nn.Module):
    def __init__(self, text_enc, image_enc):
        super(VisualSemanticModel, self).__init__()
        self.text_enc = text_enc
        self.image_enc = image_enc

    def init_weights(self):
        raise NotImplementedError

    def forward(self, images, sentences, *args):
        img_emb, txt_emb = self.forward_emb(images, sentences, *args)
        return self.similarity(img_emb, txt_emb, *args)

    def forward_emb(self, images, sentences, *args):
        img_emb = self.image_enc(images, *args)
        txt_emb = self.text_enc(sentences, *args)
        return img_emb, txt_emb

    def similarity(self, img_emb, txt_emb, *args):
        raise NotImplementedError


class ContrastiveLoss(nn.Module):
    """ Compute contrastive loss"""

    def __init__(self, margin=0, max_violation=True, reduction='sum'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.reduction = reduction

    def forward(self, scores):

        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.reduction == 'sum':
            return cost_s.sum() + cost_im.sum()
        elif self.reduction == 'mean':
            return cost_s.mean() + cost_im.mean()
        else:
            return cost_s + cost_im
