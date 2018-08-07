import os
import json
import numpy as np
import re
import itertools
import collections
import torch
import xml.etree.ElementTree
from . import field
from .example import Example
from ..utils import nostdout
from pycocotools.coco import COCO as pyCOCO
from pathos.multiprocessing import ProcessingPool as Pool


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):
            batch = list(zip(*batch))
            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
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

        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)

        self.dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            self.dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = Dataset(value_examples, value_fields)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)

            value_batch_flattened = list(itertools.chain(*value_batch))
            value_tensors_flattened = self.value_dataset.collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in value_batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return key_tensors, value_tensors

        return collate

    def __getitem__(self, i):
        key_data = self.key_dataset[i]

        values_data = []
        for idx in self.dictionary[i]:
            value_data = self.value_dataset[idx]
            values_data.append(value_data)

        return key_data, values_data

    def __len__(self):
        return len(self.key_dataset)


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']
        self.image_children = collections.defaultdict(set)
        self.text_children = collections.defaultdict(set)
        for e in self.examples:
            self.image_children[e.image].add(e.text)
            self.text_children[e.text].add(e.image)

    def image_dataset(self):
        image_set = list(self.image_children.keys())
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_dataset(self):
        text_set = list(self.text_children.keys())
        examples = [Example.fromdict({'text': t}) for t in text_set]
        dataset = Dataset(examples, {'text': self.image_field})
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class Flickr(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_file):
        dataset = json.load(open(ann_file, 'r'))['images']
        self.train_examples, self.val_examples, self.test_examples = self.get_samples(dataset, img_root)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(Flickr, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
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


class FlickrEntities(PairedDataset):
    def __init__(self, image_field, text_field, det_ids_field, img_root, ann_file, entities_root, multiprocess=True):
        self.multiprocess = multiprocess
        self.train_examples, self.val_examples, self.test_examples = self.get_samples(ann_file, img_root, entities_root)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(FlickrEntities, self).__init__(examples, {'image': image_field, 'text': text_field,
                                                        'det_ids': det_ids_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    def get_samples(self, ann_file, img_root, entities_root):
        def _get_sample(d):
            filename = d['filename']
            split = d['split']
            xml_root = xml.etree.ElementTree.parse(os.path.join(entities_root, 'Annotations',
                                                                filename.replace('.jpg', '.xml'))).getroot()
            det_dict = dict()
            id_counter = 1
            for obj in xml_root.findall('object'):
                obj_names = [o.text for o in obj.findall('name')]
                if obj.find('bndbox'):
                    bbox = tuple(int(o.text) for o in obj.find('bndbox'))
                    for obj_name in obj_names:
                        if obj_name not in det_dict:
                            det_dict[obj_name] = {'id': id_counter, 'bdnbox': [bbox]}
                            id_counter += 1
                        else:
                            det_dict[obj_name]['bdnbox'].append(bbox)

            bdnboxes = [[] for _ in range(id_counter - 1)]
            for it in det_dict.values():
                bdnboxes[it['id'] - 1] = tuple(it['bdnbox'])
            bdnboxes = tuple(bdnboxes)

            captions = [l.strip() for l in open(os.path.join(entities_root, 'Sentences',
                                                             filename.replace('.jpg', '.txt'))).readlines()]
            outputs = []
            for c in captions:
                matches = prog.findall(c)
                caption = []
                det_ids = []

                for match in matches:
                    for i, grp in enumerate(match):
                        if i in (0, 2):
                            if grp != '':
                                words = grp.strip().split(' ')
                                for w in words:
                                    if w not in field.TextField.punctuations and w != '':
                                        caption.append(w)
                                        det_ids.append(0)
                        elif i == 1:
                            words = grp[1:-1].strip().split(' ')
                            obj_name = words[0].split('#')[-1].split('/')[0]
                            words = words[1:]
                            for w in words:
                                if w not in field.TextField.punctuations and w != '':
                                    caption.append(w)
                                    if obj_name in det_dict:
                                        det_ids.append(det_dict[obj_name]['id'])
                                    else:
                                        det_ids.append(0)

                caption = ' '.join(caption)
                if caption != '' and np.sum(np.asarray(det_ids)) > 0:
                    example = Example.fromdict({'image': (os.path.join(img_root, filename), bdnboxes),
                                                'text': caption,
                                                'det_ids': det_ids})
                    outputs.append([example, split])

            return outputs

        train_samples = []
        val_samples = []
        test_samples = []

        prog = re.compile(r'([^\[\]]*)(\[[^\[\]]+\])([^\[\]]*)')
        dataset = json.load(open(ann_file, 'r'))['images']

        if self.multiprocess:
            pool = Pool()
            samples = pool.map(_get_sample, dataset)
            samples = [item for sublist in samples for item in sublist]
        else:
            samples = []
            for d in dataset:
                samples.extend(_get_sample(d))

        for example, split in samples:
            if split == 'train':
                train_samples.append(example)
            elif split == 'val':
                val_samples.append(example)
            elif split == 'test':
                test_samples.append(example)

        return train_samples, val_samples, test_samples


class COCO(PairedDataset):
    # TODO check correctness of Karpathy's splits.
    # TODO fix behaviour when use_restval=False
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=True):
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

        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCO, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            if isinstance(roots[split]['cap'], tuple):
                coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
                root = roots[split]['img']
            else:
                coco_dataset = (pyCOCO(roots[split]['cap']),)
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


class CUB200(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, split_root):
        self.train_examples, self.val_examples, self.test_examples = self.get_samples(img_root, ann_root, split_root)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(CUB200, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, img_root, ann_root, split_root):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            if split == 'train':
                split_filename = '%s_noCub.txt' % split
            else:
                split_filename = '%s.txt' % split
            filenames = [f.strip() for f in open(os.path.join(split_root, split_filename), 'r').readlines()]

            for filename in filenames:
                captions = [f.strip() for f in open(os.path.join(ann_root, filename.replace('.jpg', '.txt')), 'r').readlines()]
                for caption in captions:
                    example = Example.fromdict({'image': os.path.join(img_root, filename), 'text': caption})

                    if split == 'train':
                        train_samples.append(example)
                    elif split == 'val':
                        val_samples.append(example)
                    elif split == 'test':
                        test_samples.append(example)

        return train_samples, val_samples, test_samples


class Oxford102(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, split_root):
        self.train_examples, self.val_examples, self.test_examples = self.get_samples(img_root, ann_root, split_root)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(Oxford102, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, img_root, ann_root, split_root):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            classes = [f.strip() for f in open(os.path.join(split_root, '%sclasses.txt' % split), 'r').readlines()]

            for c in classes:
                filenames = [f for f in os.listdir(os.path.join(ann_root, c)) if f.endswith('.txt')]
                for filename in filenames:
                    captions = [f.strip() for f in open(os.path.join(ann_root, c, filename), 'r').readlines()]
                    for caption in captions:
                        example = Example.fromdict({'image': os.path.join(img_root, filename.replace('.txt', '.jpg')),
                                                    'text': caption})

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
        super(TabularDataset, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
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
