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

### Pre-processing visual data
``` python
from speaksee.data import ImageField, TextField
from speaksee.data.pipeline import EncodeCNN
from speaksee.data.dataset import COCO
from torchvision.models import resnet101
from torchvision.transforms import Compose, Normalize
from torch import nn
import torch
from tqdm import tqdm

device = torch.device('cuda')

# Preprocess with some fancy cnn and transformation
cnn = resnet101(pretrained=True).to(device)
cnn.avgpool.forward = lambda x : x.mean(-1).mean(-1)
cnn.fc = nn.Sequential()

transforms = Compose([
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

prepro_pipeline = EncodeCNN(cnn, transforms)
image_field = ImageField(preprocessing=prepro_pipeline, precomp_path='/nas/houston/lorenzo/fc2k_coco.hdf5')
```

### Pre-processing textual data
``` python
# Pipeline for text
text_field = TextField(eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True)
```

### Calling a dataset
``` python
# Create the dataset
dataset = COCO(image_field, text_field, '/tmp/coco/images/',
               '/nas/houston/lorenzo/vse/data/coco/annotations',
               '/nas/houston/lorenzo/vse/data/coco/annotations')
train_dataset, val_dataset, test_dataset = dataset.splits
#image_field.precomp(dataset)  # do this once, or to refresh cache (we might change this in the near future)
text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
```

### Training a model
``` python
from speaksee.models import FC
model = FC(len(text_field.vocab), 2048, 512, 512, dropout_prob_lm=0).to(device)

from speaksee.data import DataLoader
dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=16)

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import NLLLoss
optim = Adam(model.parameters(), lr=5e-4)
scheduler = StepLR(optim, step_size=3, gamma=.8)
loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])

for e in range(50):
    # Training
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader_train)) as pbar:
        for it, (images, captions )in enumerate(dataloader_train):
            images, captions = images.to(device), captions.to(device)
            out = model(images, captions)
            optim.zero_grad()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
            loss.backward()
            optim.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it+1))
            pbar.update()

    if e % 3 == 0 and model.ss_prob < .25:
        model.ss_prob += .05

    # Validation
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - val' % e, unit='it', total=len(dataloader_val)) as pbar:
        for it, (images, captions )in enumerate(dataloader_val):
            images, captions = images.to(device), captions.to(device)
            out = model(images, captions)
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it+1))
            pbar.update()

    # Serialize model
    torch.save({
        'epoch': e,
        'val_loss': running_loss / len(iter(dataloader_val)),
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
    }, '/nas/houston/lorenzo/fc_epoch_%03d.pth' % e)
```

### Model zoo
| Model        | CIDEr | Download   |
|--------------|-------|------------|
| FC-2k (beam) | 93.78 | [Download](http://aimagelab.ing.unimore.it/speaksee/model_zoo/fc_epoch_029.pth)        |
