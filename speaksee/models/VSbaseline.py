import torch
from torch import nn
import numpy as np
from .VisualSemanticModel import VisualSemanticModel
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2norm(X):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class BaseImageEncoder(nn.Module):
    def __init__(self, embed_size, finetune=True):
        super(BaseImageEncoder, self).__init__()
        self.embed_size = embed_size

        self.cnn = models.resnet18(pretrained=True).to(device)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        self.fc = nn.Linear(self.cnn.fc.in_features, embed_size).to(device)
        self.cnn.fc = nn.Sequential()
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, *args):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        features = l2norm(features)

        return features


class BaseTextEncoder(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers):
        super(BaseTextEncoder, self).__init__()
        self.embed_size = embed_size
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        # caption embedding
        self.rnn = nn.LSTM(word_dim, embed_size, num_layers, batch_first=True).to(device)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions"""
        # Embed word ids to vectors
        x = self.embed(x)

        lengths_sorted, index = torch.sort(lengths, descending=True)
        x_sorted = x[index]

        packed = pack_padded_sequence(x_sorted, lengths_sorted, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths_sorted.cpu()).view(-1, 1, 1).to(device)
        I = I.expand(x.size(0), 1, self.embed_size)-1  # index for taking the last element of each sequence with gather
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # back to the original sequence order
        original_index = torch.zeros(lengths.size(0)).long()
        for i in range(len(index)):
            original_index[index[i]] = i

        # lengths = lengths_sorted[original_index]
        out = out[original_index]

        # normalization in the joint embedding space
        out = l2norm(out)

        return out


class VSbaseline(VisualSemanticModel):
    def __init__(self, embed_size, vocab_size, word_dim, num_layers_lstm):

        image_enc = BaseImageEncoder(embed_size=embed_size)
        text_enc = BaseTextEncoder(vocab_size=vocab_size, word_dim=word_dim, embed_size=embed_size, num_layers=num_layers_lstm)

        super(VSbaseline, self).__init__(text_enc, image_enc)

    def similarity(self, img_emb, txt_emb, *args):
        return torch.mm(img_emb, txt_emb.t())

