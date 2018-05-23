# Speaksee

Speaksee is a Python package that provides utilities for working with Visual-Semantic data, developed at AImageLab.

## Installation
Speaksee is under development. 

To have a working installation, make sure you have Python 2.7 or 3.5+. You can then install speaksee from source with:

```
git clone https://github.com/aimagelab/speaksee
cd speaksee
pip install -e .
```

and obtain fresh upgrades without reinstalling it, simply running:

```
git pull
```

## Example(s)
``` python
from speaksee.data import ImageField, TextField
from speaksee.data.pipeline import EncodeCNN
from speaksee.data.dataset import Flickr
from torchvision.models import resnet152
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch import nn

# Define an ImageField to pre-process images with some fancy cnn, and cache all vectors to disk
cnn = resnet152(pretrained=True)
cnn = cnn.cuda(1)
cnn.avgpool.forward = lambda x : x.mean(-1).mean(-1)
cnn.fc = nn.Sequential()

transforms = Compose([
    Resize(224),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

prepro_pipeline = EncodeCNN(cnn, transforms)
image_field = ImageField(preprocessing=prepro_pipeline, precomp_path='/raid/lbaraldi/flickr8k.pkl') 

# Define a TextField to pre-process captions (see class for a full list of options)
text_field = TextField(init_token='<bos>', eos_token='<eos>', include_lengths=True)

# Create a paired image-captions dataset
dataset = Flickr(image_field, text_field, '/nas/houston/lorenzo/vse/data/f8k/images/', '/nas/houston/lorenzo/vse/data/f8k/dataset_flickr8k.json')
train_dataset, val_dataset, test_dataset = dataset.splits

image_field.precomp(dataset.image)  # do this once to populate the disk cache, or to refresh it
text_field.build_vocab(train_dataset, val_dataset)

# And finally, iterate over data pairs
from speaksee.data import DataLoader
dataloader = DataLoader(dataset, batch_size=10)

for image, caption, caption_length in iter(dataloader):
    pass

# ... or just over the image/text set
dataset_imgs = dataset.image_dataset()
dataloader = DataLoader(dataset_imgs, batch_size=10)

for image in iter(dataloader):
    pass

```