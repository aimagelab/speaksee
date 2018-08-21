# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import unittest
import speaksee.data as data
import numpy as np
import torch


'''class TestImageField(object):
    def test_preprocessing(self):
        field = data.ImageField()
        image = ''
        expected_image = ''
        assert field.preprocess(image) == expected_image
'''

class TestTextField(object):
    def test_pad(self):
        # Default case.
        field = data.TextField()
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [["a", "sentence", "of", "data", "."],
                                     ["yet", "another", "<pad>", "<pad>", "<pad>"],
                                     ["one", "last", "sent", "<pad>", "<pad>"]]
        expected_lengths = [5, 2, 3]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.TextField(include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)

        # Test fix_length properly truncates and pads.
        field = data.TextField(fix_length=3)
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [["a", "sentence", "of"],
                                     ["yet", "another", "<pad>"],
                                     ["one", "last", "sent"]]
        expected_lengths = [3, 2, 3]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.TextField(fix_length=3, include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)
        field = data.TextField(fix_length=3, truncate_first=True)
        expected_padded_minibatch = [["of", "data", "."],
                                     ["yet", "another", "<pad>"],
                                     ["one", "last", "sent"]]
        assert field.pad(minibatch) == expected_padded_minibatch

        # Test init_token is properly handled.
        field = data.TextField(fix_length=4, init_token="<bos>")
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [["<bos>", "a", "sentence", "of"],
                                     ["<bos>", "yet", "another", "<pad>"],
                                     ["<bos>", "one", "last", "sent"]]
        expected_lengths = [4, 3, 4]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.TextField(fix_length=4, init_token="<bos>", include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)

        # Test init_token and eos_token are properly handled.
        field = data.TextField(init_token="<bos>", eos_token="<eos>")
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [
            ["<bos>", "a", "sentence", "of", "data", ".", "<eos>"],
            ["<bos>", "yet", "another", "<eos>", "<pad>", "<pad>", "<pad>"],
            ["<bos>", "one", "last", "sent", "<eos>", "<pad>", "<pad>"]]
        expected_lengths = [7, 4, 5]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.TextField(init_token="<bos>", eos_token="<eos>", include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)

    def test_decode(self):
        def test_all_dtypes(word_idxs, expected_output):
            assert field.decode(word_idxs) == expected_output
            assert field.decode(np.asarray(word_idxs)) == expected_output
            assert field.decode(torch.from_numpy(np.asarray(word_idxs))) == expected_output

        class MyVocab(object):
            def __init__(self, eos_token):
                self.itos = {0: 'a',
                        1: 'b',
                        2: eos_token,
                        3: 'c'}

        field = data.TextField()
        field.vocab = MyVocab(field.eos_token)

        # Empty captions (not tested for PyTorch tensors)
        word_idxs = []
        expected_output = ''
        assert field.decode(word_idxs) == expected_output
        assert field.decode(np.asarray(word_idxs)) == expected_output

        word_idxs = [[]]
        expected_output = ['', ]
        assert field.decode(word_idxs) == expected_output
        assert field.decode(np.asarray(word_idxs)) == expected_output

        # Single caption
        word_idxs = [0, 3, 2, 1]
        expected_output = 'a c'
        test_all_dtypes(word_idxs, expected_output)

        # Batch of captions
        word_idxs = [[0, 3, 2, 1],
                     [3, 3, 2, 1],
                     [2, 1, 1, 1]]
        expected_output = ['a c', 'c c', '']
        test_all_dtypes(word_idxs, expected_output)
