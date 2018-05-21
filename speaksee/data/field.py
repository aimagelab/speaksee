# coding: utf8
from collections import Counter, OrderedDict
from itertools import chain
import six
import torch
from tqdm import tqdm
import numpy as np
import pickle as pkl
import os

from .dataset import Dataset
from ..vocab import Vocab
# from .pipeline import Pipeline
from .utils import get_tokenizer
# from ..vocab import Vocab, SubwordVocab
from torchvision.datasets.folder import default_loader


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
        return batch


class ImageField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, precomp_path=None):
        super(ImageField, self).__init__(preprocessing, postprocessing)
        self.precomp_path = precomp_path
        self.precomp_data = None
        self.precomp_index = None
        if os.path.exists(precomp_path):
            self.precomp_data = pkl.load(open(precomp_path, 'rb'))

    def preprocess(self, x, avoid_precomp=False):
        """
        Loads a single example using this field.

        Args:
            x:
            avoid_precomp:

        Returns:

        """
        if self.precomp_data and not avoid_precomp:
            index = self.precomp_data[0]
            precomp_data = self.precomp_data[1]
            return precomp_data[index.index(x)]
        else:
            x = default_loader(x)
            if self.preprocessing is not None:
                return self.preprocessing(x)
            else:
                return x

    def precomp(self, xs):
        precomp_data = []
        xs = set(xs)
        for x in tqdm(xs, desc='Building precomputed data'):
            precomp_data.append(self.preprocess(x, avoid_precomp=True))

        precomp_data = np.concatenate(precomp_data, 0)
        self.precomp_data = [xs, precomp_data]
        pkl.dump(self.precomp_data, open(self.precomp_path, 'wb'), 2)


class TextField(RawField):
    vocab_cls = Vocab

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 include_lengths=False, batch_first=False, pad_token="<pad>", unk_token="<unk>", pad_first=False,
                 truncate_first=False):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
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
        if isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = six.text_type.lower(x)
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

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
