import os
import json
from association import Association


class Dataset(object):
    pass


class PairedDataset(Dataset):
    def __init__(self, samples, left_field, right_field):
        """

        Args:
            samples: a list of tuples
            left_field:
            right_field:
        """
        self.examples = Association(samples)
        self.left_field = left_field
        self.right_field = right_field
        super(PairedDataset, self).__init__()

    @property
    def splits(self):
        raise NotImplementedError

    def __getitem__(self, i):
        sample = self.examples[i]
        left = self.left_field.preprocess(sample[0])
        right = self.right_field(sample[1])
        return left, right

    def __len__(self):
        return len(self.examples)


class Flickr(PairedDataset):
    def __init__(self, img_root, ann_file, left_field, right_field):
        dataset = json.load(open(ann_file, 'r'))['images']
        self.train_samples, self.val_samples, self.test_samples = self.get_samples(dataset, img_root)
        samples = self.train_samples + self.val_samples + self.test_samples
        super(Flickr, self).__init__(samples, left_field, right_field)

    @property
    def splits(self):
        train_split = PairedDataset(self.train_samples, self.left_field, self.right_field)
        val_split = PairedDataset(self.val_samples, self.left_field, self.right_field)
        test_split = PairedDataset(self.test_samples, self.left_field, self.right_field)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, dataset, img_root):
        train_samples = []
        val_samples = []
        test_samples = []

        '''
        samples = []
        ids = []
        for i, d in enumerate(dataset):
            if d['split'] == split:
                ids += [(i, x) for x in range(len(d['sentences']))]

        for i in range(len(ids)):
            ann_id = ids[i]
            img_id = ann_id[0]
            caption = dataset[img_id]['sentences'][ann_id[1]]['raw']
            path = dataset[img_id]['filename']

            samples.append((os.path.join(img_root, path), caption))
        '''

        for d in dataset:
            for c in d['sentences']:
                filename = d['filename']
                caption = c['raw']

                if d['split'] == 'train':
                    train_samples.append((os.path.join(img_root, filename), caption))
                elif d['split'] == 'val':
                    val_samples.append((os.path.join(img_root, filename), caption))
                elif d['split'] == 'test':
                    test_samples.append((os.path.join(img_root, filename), caption))

        return train_samples, val_samples, test_samples
