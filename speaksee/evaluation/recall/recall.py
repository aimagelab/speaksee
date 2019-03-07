import numpy as np
import torch


def old_recall(images, captions, model, mode='i2t', npts=None, data='coco', lenghts=None, return_ranks=False):
    step = 5 if data is 'coco' else 1

    if npts is None:
        if isinstance(images, list) or isinstance(images, tuple):
            npts = images[0].shape[0] // step
        else:
            npts = images.shape[0] // step

    ranks = np.zeros(npts) if mode is 'i2t' else np.zeros(step * npts)
    top1 = np.zeros(npts) if mode is 'i2t' else np.zeros(step * npts)

    caps = captions
    if isinstance(images, list) or isinstance(images, tuple):
        ims = [i[::step] for i in images]
    else:
        ims = images[::step]

    if mode is 'i2t':
        d_arr = model.similarity(ims, caps, lenghts).cpu().detach().numpy()
        for index in range(npts):
            d = d_arr[index].flatten()
            inds = np.argsort(d)[::-1]  # indexes that sort similarities from largest to smallest
            # find the best rank position among the 5 captions ground-truth of the image
            rank = 1e20
            for i in range(step * index, step * index + step, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            # ranks array with best rank achieved for index image (among 5 ground-truth captions if coco)
            ranks[index] = rank
            # array with most similar caption retrieved for index image
            top1[index] = inds[0]
    elif mode is 't2i':
        for index in range(npts):
            # Get query captions
            caps = captions[step * index:step * index + step]
            d = model.similarity(ims, caps, lenghts).cpu().detach().numpy().T
            inds = np.zeros(d.shape)
            for i in range(len(inds)):
                inds[i] = np.argsort(d[i])[::-1]
                ranks[step * index + i] = np.where(inds[i] == index)[0][0]
                top1[step * index + i] = inds[i][0]
    else:
        raise ValueError('mode not correct')

    # Compute metrics recall
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr


def recall(images, captions, model, mode='i2t', lenghts=None, return_ranks=False):
    b_s = images.size(0)
    im_flatten = images.view(b_s, -1)
    indexes_b = 1 - torch.all(im_flatten[:-1] == im_flatten[1:], -1) # (b_s, )
    indexes_b = torch.cat([torch.ones(1, ).to(indexes_b.device).byte(), indexes_b, torch.ones(1, ).to(indexes_b.device).byte()])
    npts = torch.sum(indexes_b).item() - 1
    indexes = torch.nonzero(indexes_b).squeeze(-1) # (npts, )

    ranks = np.zeros(npts) if mode is 'i2t' else np.zeros(indexes_b.shape[0] - 1)
    top1 = np.zeros(npts) if mode is 'i2t' else np.zeros(indexes_b.shape[0] - 1)

    caps = captions
    if isinstance(images, list) or isinstance(images, tuple):
        ims = [i[indexes[:-1]] for i in images]
    else:
        ims = images[indexes[:-1]]

    if mode is 'i2t':
        d_arr = model.similarity(ims, caps, lenghts).cpu().detach().numpy()
        for img in range(npts):
            d = d_arr[img].flatten()
            inds = np.argsort(d)[::-1]  # indexes that sort similarities from largest to smallest
            # find the best rank position among the 5 captions ground-truth of the image
            rank = 1e20
            for i in range(indexes[img], indexes[img+1]):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            # ranks array with best rank achieved for index image (among 5 ground-truth captions if coco)
            ranks[img] = rank
            # array with most similar caption retrieved for index image
            top1[img] = inds[0]
    elif mode is 't2i':
        for img in range(npts):
            # Get query captions
            caps = captions[indexes[img]:indexes[img+1]]
            d = model.similarity(ims, caps, lenghts).cpu().detach().numpy().T
            inds = np.zeros(d.shape)
            for i in range(len(inds)):
                inds[i] = np.argsort(d[i])[::-1]
                ranks[indexes[img] + i] = np.where(inds[i] == img)[0][0]
                top1[indexes[img] + i] = inds[i][0]
    else:
        raise ValueError('mode not correct')

    # Compute metrics recall
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr
