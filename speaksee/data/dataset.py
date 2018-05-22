import os
import json
import numpy as np
from collections import defaultdict
from .example import Example
from pycocotools.coco import COCO


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):
            transposed = list(zip(*batch))
            tensors = []
            for field, data in zip(self.fields.values(), transposed):
                tensor = field.process(data)
                if isinstance(tensor, tuple):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)

            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]
        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))

        if len(data) > 1:
            return data
        else:
            return data[0]

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield self.fields[attr].preprocess(getattr(x, attr))


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

    def image_dataset(self):
        image_set = list(self.image_children.keys())
        dataset = Dataset(image_set, {'image': self.image_field})
        return dataset

    def text_dataset(self):
        text_set = list(self.text_children.keys())
        dataset = Dataset(text_set, {'text': self.image_field})
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class Flickr(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_file):
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


class COCO(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=False):
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }

        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))
            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (
                ids['train'],
                np.load(os.path.join(id_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCO, self).__init__(examples, image_field, text_field)

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.image_field, self.text_field)
        val_split = PairedDataset(self.val_examples, self.image_field, self.text_field)
        test_split = PairedDataset(self.test_examples, self.image_field, self.text_field)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            if isinstance(roots[split]['cap'], tuple):
                coco_dataset = (COCO(roots[split]['cap'][0]), COCO(roots[split]['cap'][1]))
                root = roots[split]['img']
            else:
                coco_dataset = (COCO(roots[split]['cap']),)
                root = (roots[split]['img'],)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)

            for index in range(len(ids)):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                caption = coco.anns[ann_id]['caption']
                img_id = coco.anns[ann_id]['image_id']
                filename = coco.loadImgs(img_id)[0]['file_name']

                example = Example.fromdict({'image': os.path.join(img_root, filename), 'text': caption})

                if split == 'train':
                    train_samples.append(example)
                elif split == 'val':
                    val_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        return train_samples, val_samples, test_samples


class TabularDataset(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_file_root):
        self.train_examples, self.val_examples, self.test_examples = self.get_samples(ann_file_root, img_root)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(TabularDataset, self).__init__(examples, image_field, text_field)

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.image_field, self.text_field)
        val_split = PairedDataset(self.val_examples, self.image_field, self.text_field)
        test_split = PairedDataset(self.test_examples, self.image_field, self.text_field)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, ann_file_root, img_root):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            captions = [f.strip() for f in open(os.path.join(ann_file_root, '%s_caps.txt' % split), 'r').readlines()]
            filenames = [f.strip() for f in open(os.path.join(ann_file_root, '%s_ims.txt' % split), 'r').readlines()]
            for caption, filename in zip(captions, filenames):
                example = Example.fromdict({'image': os.path.join(img_root, filename), 'text': caption})

                if split == 'train':
                    train_samples.append(example)
                elif split == 'val':
                    val_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        return train_samples, val_samples, test_samples
