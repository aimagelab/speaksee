import torch


class EncodeCNN(object):
    def __init__(self, cnn, transforms):
        self.cnn = cnn
        self.transforms = transforms

    def __call__(self, x):
        if self.transforms is not None:
            x = self.transforms(x)

        x = x.unsqueeze(0)
        x = x.to(next(self.cnn.parameters()).device)

        with torch.no_grad():
            x = self.cnn(x).squeeze(0).data.cpu().numpy()

        return x
