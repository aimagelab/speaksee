# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import unittest
import speaksee.data as data


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
        