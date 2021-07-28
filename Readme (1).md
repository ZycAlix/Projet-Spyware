# GAIF_DATABASE_PROCESSORS

It's a readme file for using Gaif framework. In order to prepare the database you need in <strong>Text, Image, Audio</strong> and choose your data augmentation to apply, please look at this one. xD

# Installation 

<font color='red'>
<p>git clone https://github.com/nico-ri/generic-AI-framework
<p>cd generic-AI-framework
<p>pip install --editable .</font>

This command line will download all dependencies needed and setup up the environment for Gaif framework. 

# Datasets

The Gaif Framework has installed these datasets below for testing:

## Text

#### *Classification*

<font color='blue'>
    <p>Yahooanswers
</font>

### Image

#### *Detection*

<font color='blue'>
    <p>balloon (COCO)
    <p>poc_mons (YOLO)
</font>

#### *Classification*

<font color='blue'>
    <p>cifar10
    <p>omniglot
</font>

### Audio

#### *Classification*

<font color='blue'>
    <p>gtzan
</font>

Besides, Gaif could also download the dataset of MMF, Torchvision, Torchtext and Torchaudio to train these databases. It will have an exemple in section <strong>Usage</strong>

# Processors

The Gaif Framework use all of transforms of MMF, Torchvision, Torchtext and Torchaudio to apply the data augmentation and method to pre-prapare dataset. It will have an example in section <strong>Usage</strong>.
<p> We also have a simple script to show all transforms disponible to facilite the usage. 
<p><font color='red'>Python3 projects/data_augmentation/run_processors.py</font>

# Usage_Dataset

### *MMF*

MMF is an API in Gaif that provides a few datasets interessant. Here is an example below:


```
from mmf.utils.build import build_dataset


from gaif.utils.env import setup_imports
from mmf.common.registry import registry


if __name__ == "__main__":

    setup_imports()

    dataset_key = "coco"
    dataset = build_dataset(dataset_key=dataset_key)
    print(dataset.__getitem__(6))
```


### *Torchvision*

Torchvision works also. Here is an example below:

```
    ... in Builder:
    
    def load(self, config, dataset, *args, **kwargs):
        # Load the dataset using the CIFAR10Dataset class
        self.dataset = ClassificationOMNIGLOTDataset(
            config, dataset, data_folder=self.data_folder
        )
        return self.dataset
        
    ... in Dataset:
    
    def _load(self):
        # Background and Download is optional
        self.omniglot_dataset = torchvision.datasets.Omniglot(
            self._data_dir, background=True, download=True
        )
    
    ....
    
```

# Usage_Processors

The processors are used in dataset for data augmentation and pre-prepare dataset. All of transforms in Augly, Torchvision, Torchtext, Torchaudio could be added by choosing their function's name to config file. 
<p> For example: 

```
    ... in .yaml
    
    dataset_config:
    # You can specify any attributes you want, and you will get them as attributes
    # inside the config passed to the dataset. Check the Dataset implementation below.
    classification_cifar10_augly:
        # Where your data is stored
        data_dir: ${env.data_dir}
        processors:
        # The processors will be assigned to the datasets automatically by GAIF
        # For example if key is text_processor, you can access that processor inside
        # dataset object using self.text_processor
          augly_image_transforms:
            type: augly_image_transforms
            params: 
              transforms:
                - type: Brightness (Augly)
                  params: 
                    factor: 0.5
                - type: RandomBlur (Augly)
                - ToTensor (Torchvision)



```

If we want to have a specifique processor or to use processors at a different moment, we could also change a method to define processors. 
<p>For example:

```
    ... in . yaml
    
    dataset_config:
    # You can specify any attributes you want, and you will get them as attributes
    # inside the config passed to the dataset. Check the Dataset implementation below.
    classification_yahooanswers:
        # Where your data is stored
        data_dir: ${env.data_dir}
        processors:
        # The processors will be assigned to the datasets automatically by MMF
        # For example if key is text_processor, you can access that processor inside
        # dataset object using self.text_processor
          augly_text_transforms:
            type: augly_text_transforms
            params: 
              transforms:
                # - type: ReplaceBidirectional
                #   params: 
                #       p: 1.0
                - type: ReplaceSimilarUnicodeChars
                  params:
                    aug_word_p: 0.6
                - type: InsertPunctuationChars
                  params: 
                    {}
                # - type: vocab
                #   params:
                #       max_length: 10
                #       vocab:
                #           type: random
                #           vocab_file: yahoo_answers_csv/words.txt
          text_processor:
                type: vocab
                params:
                    max_length: 10
                    vocab:
                        type: random
                        vocab_file: yahoo_answers_csv/words.txt




```

# API

MMF(https://mmf.sh/)

Torchvision(https://pytorch.org/vision/stable/index.html)

Torchtext(https://pytorch.org/text/stable/index.html)

Torchaudio(https://pytorch.org/audio/stable/index.html)

Augly(https://github.com/facebookresearch/AugLy)

# License


```python

```
