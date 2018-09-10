# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
from itertools import chain
import six
import torch
from tqdm import tqdm
import numpy as np
import h5py
import os

from .dataset import Dataset
from ..vocab import Vocab
from .utils import get_tokenizer
from torchvision.datasets.folder import default_loader
from torchvision import transforms


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class ImageField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, precomp_path=None):
        super(ImageField, self).__init__(preprocessing, postprocessing)
        self.precomp_path = precomp_path
        self.precomp_index = None
        self.precomp_data = None

        if self.precomp_path and os.path.isfile(self.precomp_path):
            precomp_file = h5py.File(self.precomp_path, 'r')
            self.precomp_index = list(precomp_file['index'][:])
            if six.PY3:
                self.precomp_index = [s.decode('utf-8') for s in self.precomp_index]
            # self.precomp_data = dict()
            # for x in self.precomp_index:
            #     self.precomp_data[x] = precomp_file['data'][self.precomp_index.index(x)]

    def preprocess(self, x, avoid_precomp=False):
        """
        Loads a single example using this field.

        Args:
            x:
            avoid_precomp:

        Returns:

        """
        if self.precomp_path and not avoid_precomp:
            precomp_file = h5py.File(self.precomp_path, 'r')
            precomp_data = precomp_file['data']
            return precomp_data[self.precomp_index.index(x)]
            # return self.precomp_data[x]
        else:
            x = default_loader(x)
            if self.preprocessing is not None:
                x = self.preprocessing(x)
            else:
                x = transforms.ToTensor()(x)
            return x

    def precomp(self, *args):
        sources = []

        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        xs = []
        for data in sources:
            xs.extend(data)
        xs = list(set(xs))

        with h5py.File(self.precomp_path, 'w') as out:
            example = self.preprocess(xs[0], avoid_precomp=True)
            shape = [len(xs), ] + list(example.shape)
            dset_data = out.create_dataset('data', shape, example.dtype, chunks=True)
            out.create_dataset('index', data=np.array(xs, dtype='S'))
            for i, x in enumerate(tqdm(xs, desc='Building precomputed data')):
                dset_data[i] = self.preprocess(x, avoid_precomp=True)

        self.precomp_index = xs


class ImageDetectionsField(RawField):
    def __init__(self, postprocessing=None, detections_path=None):
        self.max_detections = 100
        self.detections_path = detections_path
        self.detections_file = h5py.File(self.detections_path, 'r')
        self.detections = dict()
        for key in self.detections_file.keys():
            self.detections[key] = self.detections_file[key][()]

        super(ImageDetectionsField, self).__init__(None, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        # image_id = int(x.split('_')[-1].split('.')[0])
        image_id = int(x.split('/')[-1].split('.')[0])
        # det_file = h5py.File(self.detections_path, 'r')
        try:
            precomp_data = h5py.File(self.detections_path, 'r')['%d_features' % image_id][()]
            # precomp_data = det_file['%d' % image_id]
        except:
            precomp_data = np.random.rand(10, 2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return precomp_data.astype(np.float32)


class ImageAssociatedDetectionsField(RawField):
    def __init__(self, postprocessing=None, detections_path=None, image_features_path=None):
        self.max_detections = 100
        self.detections_path = detections_path
        self.image_features_path = image_features_path
        self.detections_file = h5py.File(self.detections_path, 'r')
        self.detections_dict = dict()
        for key in self.detections_file.keys():
            self.detections_dict[key] = self.detections_file[key][()]

        img_precomp_file = h5py.File(self.image_features_path, 'r')
        self.precomp_index = list(img_precomp_file['index'][:])
        if six.PY3:
            self.precomp_index = [s.decode('utf-8') for s in self.precomp_index]

        img_precomp_data = img_precomp_file['data']
        self.image_features_precomp = dict()
        for key in self.precomp_index:
            self.image_features_precomp[key] = img_precomp_data[self.precomp_index.index(key)]

        super(ImageAssociatedDetectionsField, self).__init__(None, postprocessing)

    @staticmethod
    def _bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    def preprocess(self, x, avoid_precomp=False):
        image = x[0]
        gt_bboxes_set = x[1]

        id_image = image.split('/')[-1].split('.')[0]
        # f = h5py.File(self.detections_path, 'r')
        # det_bboxes = f['%s_boxes' % id_image]
        # det_features = f['%s_features' % id_image]

        det_bboxes = self.detections_dict['%s_boxes' % id_image]
        det_features = self.detections_dict['%s_features' % id_image]
        feature_dim = det_features.shape[-1]
        features = np.zeros((self.max_detections, feature_dim))

        for i, bboxes in enumerate(gt_bboxes_set[:self.max_detections]):
            overall_bbox = [min(b[0] for b in bboxes),
                      min(b[1] for b in bboxes),
                      max(b[2] for b in bboxes),
                      max(b[3] for b in bboxes)]

            id_bbox = -1
            iou_max = 0
            for k, det_bbox in enumerate(det_bboxes):
                iou = self._bb_intersection_over_union(overall_bbox, det_bbox)
                if iou_max < iou:
                    id_bbox = k
                    iou_max = iou

            features[i] = np.asarray(det_features[id_bbox])

        return features.astype(np.float32), self.image_features_precomp[image]


class CocoImageAssociatedDetectionsField(RawField):
    def __init__(self, postprocessing=None, detections_path=None, classes_path=None,
                 padding_idx=0, fix_length=None, pad_init=True, pad_eos=True, dtype=torch.long,
                 max_detections=100, max_length=100):
        self.max_detections = max_detections
        self.max_length = max_length
        self.detections_path = detections_path
        self.padding_idx = padding_idx
        self.fix_length = fix_length
        self.init_token = padding_idx if pad_init else None
        self.eos_token = padding_idx if pad_eos else None
        self.dtype = dtype

        # self.detections_file = h5py.File(self.detections_path, 'r')
        # self.detections_dict = dict()
        # for key in self.detections_file.keys():
        #     self.detections_dict[key] = self.detections_file[key][()]

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        super(CocoImageAssociatedDetectionsField, self).__init__(None, postprocessing)

    def preprocess(self, x, avoid_precomp=False):

        image = x[0]
        det_classes = x[1]

        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        f = h5py.File(self.detections_path, 'r')
        det_cls_probs = f['%s_cls_prob' % id_image][()]
        det_features = f['%s_features' % id_image][()]
        # det_cls_probs = self.detections_dict['%s_cls_prob' % id_image]
        # det_features = self.detections_dict['%s_features' % id_image]

        feature_dim = det_features.shape[-1]
        features = np.zeros((self.max_detections, feature_dim))
        det_ids = np.zeros((len(det_classes), ))

        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:])+1] for i in range(len(det_cls_probs))]
        probas = [np.max(det_cls_probs[i][1:]) for i in range(len(det_cls_probs))]

        det_seq = list(set([c for c in det_classes if c is not None]))

        for i, cls in enumerate(det_seq):
            max_prob = 0
            id_det = -1
            for k in range(len(selected_classes)):
                if cls == selected_classes[k]:
                    if probas[k] > max_prob:
                        max_prob = probas[k]
                        id_det = k
            features[i] = det_features[id_det]

        for i, cls in enumerate(det_classes):
            if cls is not None:
                det_ids[i] = [i for i in range(len(det_seq)) if det_seq[i] == cls][0] + 1

        # det_ids = det_ids[:self.max_length]

        return features.astype(np.float32), det_ids

    def process(self, minibatch, device=None):
        minibatch = list(zip(*minibatch))

        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch[1])
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch[1]:
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                list(x[:max_len]) +
                ([] if self.eos_token is None else [self.eos_token]) +
                [self.padding_idx] * max(0, max_len - len(x)))

        minibatch[1] = torch.tensor(padded, dtype=self.dtype, device=device)
        return default_collate(list(zip(*minibatch)))


class CocoMultipleAssociatedDetectionsField(RawField):
    def __init__(self, postprocessing=None, detections_path=None, classes_path=None,
                 padding_idx=0, fix_length=None, pad_init=True, pad_eos=True, dtype=torch.long,
                 max_detections=100, max_length=100):
        self.max_detections = max_detections
        self.max_length = max_length
        self.detections_path = detections_path
        self.padding_idx = padding_idx
        self.fix_length = fix_length
        self.init_token = padding_idx if pad_init else None
        self.eos_token = padding_idx if pad_eos else None
        self.dtype = dtype

        # self.detections_file = h5py.File(self.detections_path, 'r')
        # self.detections_dict = dict()
        # for key in self.detections_file.keys():
        #     self.detections_dict[key] = self.detections_file[key][()]

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        super(CocoMultipleAssociatedDetectionsField, self).__init__(None, postprocessing)

    def get_detections_inside(self, det_boxes, query):
        cond1 = det_boxes[:, 0] >= det_boxes[query, 0]
        cond2 = det_boxes[:, 1] >= det_boxes[query, 1]
        cond3 = det_boxes[:, 2] <= det_boxes[query, 2]
        cond4 = det_boxes[:, 3] <= det_boxes[query, 3]
        cond = cond1 & cond2 & cond3 & cond4
        return np.nonzero(cond)[0]

    def preprocess(self, x, avoid_precomp=False):
        image = x[0]
        det_classes = x[1]

        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        f = h5py.File(self.detections_path, 'r')
        det_cls_probs = f['%s_cls_prob' % id_image][()]
        det_features = f['%s_features' % id_image][()]
        det_boxes = f['%s_boxes' % id_image][()]
        # det_cls_probs = self.detections_dict['%s_cls_prob' % id_image]
        # det_features = self.detections_dict['%s_features' % id_image]
        # det_boxes = self.detections_dict['%s_boxes' % id_image]

        feature_dim = det_features.shape[-1]
        features = np.zeros((self.max_detections, feature_dim))
        features[:min(len(det_features), self.max_detections)] = det_features[:min(len(det_features), self.max_detections)]

        det_ids = np.full((len(det_classes), self.max_detections), -1)
        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:])+1] for i in range(len(det_cls_probs))]

        for t, cls in enumerate(det_classes):
            if cls is None:
                det_ids[t, 0] = 0
            else:
                seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]
                seed_detections_probs = np.asarray([np.max(det_cls_probs[i][1:]) for i in seed_detections])
                detections = np.concatenate([np.asarray([seed_detections[np.argmax(seed_detections_probs)]]), ] + [self.get_detections_inside(det_boxes, d) for d in seed_detections]) + 1
                det_ids[t, :min(len(detections), self.max_detections)] = detections[:min(len(detections), self.max_detections)]

        return features.astype(np.float32), det_ids

    def process(self, minibatch, device=None):
        minibatch = list(zip(*minibatch))

        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch[1])
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch[1]:
            this_pad = x[:max_len]
            dim = x.shape[-1]
            if self.init_token is not None:
                conc = np.full((1, dim), -1)
                conc[0, 0] = self.init_token
                this_pad = np.concatenate([conc, this_pad])
            if self.eos_token is not None:
                conc = np.full((1, dim), -1)
                conc[0, 0] = self.eos_token
                this_pad = np.concatenate([this_pad, conc])

            conc = np.full((max(0, max_len - len(x)), dim), -1)
            conc[:, 0] = self.padding_idx
            this_pad = np.concatenate([this_pad, conc])
            padded.append(this_pad)

        minibatch[1] = torch.tensor(padded, dtype=self.dtype, device=device)
        return default_collate(list(zip(*minibatch)))


class PadField(RawField):
    def __init__(self, padding_idx, fix_length=None, pad_init=True, pad_eos=True, dtype=torch.long):
        self.padding_idx = padding_idx
        self.fix_length = fix_length
        self.init_token = padding_idx if pad_init else None
        self.eos_token = padding_idx if pad_eos else None
        self.dtype = dtype
        super(PadField, self).__init__()

    def process(self, minibatch, device=None):
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                list(x[:max_len]) +
                ([] if self.eos_token is None else [self.eos_token]) +
                [self.padding_idx] * max(0, max_len - len(x)))

        var = torch.tensor(padded, dtype=self.dtype, device=device)
        return var


class TextField(RawField):
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded


    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, cap):
        words = []
        for w in cap:
            word = self.vocab.itos[int(w)]
            if word == '<eos>':
                break
            words.append(word)
        return words