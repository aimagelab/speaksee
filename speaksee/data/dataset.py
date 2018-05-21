import os
import json
from collections import defaultdict
from .example import Example


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class PairedDataset(Dataset):
    def __init__(self, examples, image_field, text_field):
        """

        Args:
            examples: a list of tuples
            image_field:
            text_field:
        """
        super(PairedDataset, self).__init__(examples, {'image': image_field,
                                             'text': text_field})
        self.image_field = image_field
        self.text_field = text_field
        self.image_children = defaultdict(set)
        self.text_children = defaultdict(set)
        for e in self.examples:
            self.image_children[e.image].add(e.text)
            self.text_children[e.text].add(e.image)


    def image_set(self):
        return list(self.image_children.keys())

    def text_set(self):
        return list(self.text_children.keys())

    @property
    def splits(self):
        raise NotImplementedError

    def __getitem__(self, i):
        sample = super(PairedDataset, self).__getitem__(i)
        image = self.fields['image'].preprocess(sample.image)
        text = self.fields['text'](sample.text)
        return image, text

    def __len__(self):
        return len(self.examples)


class Flickr(PairedDataset):
    def __init__(self, img_root, ann_file, image_field, text_field):
        dataset = json.load(open(ann_file, 'r'))['images']
        self.train_examples, self.val_examples, self.test_examples = self.get_samples(dataset, img_root)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(Flickr, self).__init__(examples, image_field, text_field)

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.image_field, self.text_field)
        val_split = PairedDataset(self.val_examples, self.image_field, self.text_field)
        test_split = PairedDataset(self.test_examples, self.image_field, self.text_field)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, dataset, img_root):
        train_samples = []
        val_samples = []
        test_samples = []

        for d in dataset:
            for c in d['sentences']:
                filename = d['filename']
                caption = c['raw']
                example = Example.fromdict({'image': os.path.join(img_root, filename),
                                   'text': caption})

                if d['split'] == 'train':
                    train_samples.append(example)
                elif d['split'] == 'val':
                    val_samples.append(example)
                elif d['split'] == 'test':
                    test_samples.append(example)

        return train_samples, val_samples, test_samples
