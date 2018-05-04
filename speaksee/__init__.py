def foo():
    print 'Funziona.'

if False:
    import argparse
    from models.fc import FC
    import torch
    from torch.autograd import Variable

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_encoding_size', type=int, default=512, help='The encoding size of words and of the image.')
    parser.add_argument('--rnn_size', type=int, default=512, help='Hidden size of the RNN')
    parser.add_argument('--dropout_prob_lm', type=float, default=.5, help='Dropout probability of the language model')
    parser.add_argument('--vocabulary_size', type=int, default=1024) # remove
    parser.add_argument('--img_feat_size', type=int, default=4096)
    opt = parser.parse_args()

    model = FC(opt)
    images = Variable(torch.zeros((10, opt.img_feat_size))).float()
    words = Variable(torch.zeros((10, 100))).long()

    outputs = model(images, words)