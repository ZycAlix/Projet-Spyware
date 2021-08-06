# generic-AI-framework (GAIF)

GAIF is an extension of [MMF](https://github.com/facebookresearch/mmf) which brings other models, datasets, optimizer and trainer
Please check the documentation of [MMF](https://mmf.sh/).
For a quick presentation of MMF, please check this [notebook](https://colab.research.google.com/github/facebookresearch/mmf/blob/notebooks/notebooks/mmf_hm_example.ipynb)

<!-- toc -->

- [Prerequisties](#prerequisties)
- [Installation](#installation)
- [How to use](#how-to-use)
  - [Dataset Step](#dataset-step)
  - [Model Step](#model-step)
- [Dataset](#dataset)
- [Processor](#processor)
- [Models](#models)
- [License](#license)

<!-- tocstop -->


## Prerequisties

GAIF was tested with **Ubuntu 20.04** and **python 3.8.10**

## Installation

```sh
git clone https://github.com/nico-ri/generic-AI-framework
cd generic-AI-framework
pip install .
```

## How to use
For training from scratch vgg16
### Dataset Step

### Model Step
**Step 1**: create a yaml file *config.yaml* to store the configuration of the training 

```yaml
dataset: classification_cifar10

model_config:
  vgg16:
    model: vgg16
    losses: 
      - type: cross_entropy
  
optimizer:
  type: Adam
  params:
    lr: 1e-3
    weight_decay: 1e-5

evaluation:
  metrics:
    - accuracy

# trainer lightning 
trainer:
  params:
    gpus: 1
    max_epochs: 100
    logger: true
    progress_bar_refresh_rate: 1
    val_check_interval: 100
    checkpoint_callback: true

training:
  trainer: lightning_gaif
  seed: 1
  batch_size: 32
  max_epochs: 20
  tensorboard: true
```

**Step 2**: Create a python file *train.py* to run the training

```python
from mmf_cli.run import run
from mmf.common.registry import registry
from gaif.utils.env import setup_imports

setup_imports()

registry.mapping["state"] = {}

opts = [
    "config='PATH_TO/config.yaml'",
    "model=vgg16",
    "dataset=classification_cifar10",
]

run(opts=opts)
```

## [Dataset](./doc/Dataset_README.md)
Inside our GAIF framework, the dataset part is mainly loaded with various types of data, such as COCO, Yolo, to facilitate the extraction of information and tags. Generally, all of dataset are stored in this location : **".cache/torch/mmf/data"** by default. 

GAIF provides different methods of **Data Augmentation** to make the dataset more robust. Of course, we also provide the function of visualizing the dataset samples to facilitate a more intuitive understanding of this type of data. 

Generally, Dataset consists of three parts: **Builder**, **Datasets**, and configuration file **.yaml**. 
  - Each Builder will have a key word which is recorded in **Register**. Then GAIF can find the Dataset and configuration files through the Builder
  - The Dataset is mainly to build a data class, while providing sample output and sample visualization. **Data Augmentation** also happens here.
  - Each dataset has his own configuration file which provides parameters for **Data Augmentation** realised by Processors, could be override by the config of Model.  

Finally, For more details about Dataset, please click on the title of this section. 


## [Processor](./doc/Proc_README.md)

Processors are generally used in the GAIF framework to implement various Data Augmentation and to tersorize data from datasets. But there are also some special Processor, such as TextProcessor, can be used to tokenize text data and build vocab dictionary. 

Finally, For more details about Processor, please click on the title of this section. 

## Models
| Name_Model | type | THEME | Key_Word_Registry | Description |
| ---- | --- | ---- | --- | ---- |
| [**All models from timm**](https://github.com/rwightman/pytorch-image-models) | Image | Classification | model_timm | A large collection of image models |
| [**All classification models from torchvision**](https://pytorch.org/vision/stable/models.html) | Image | Classification | model_torchvision |  A large collection of image models | 
| [**Efficientnet**](https://github.com/lukemelas/EfficientNet-PyTorch) | Image| Classification | efficientnet |Implementation of a Efficientnet by @lukemelas |
| [**Simclr from lightly**](https://github.com/lightly-ai/lightly)| Image| Self-Supervised Learning| simclr| Implementation of simclr training by [lightly](https://github.com/lightly-ai/lightly)| 


* [timm](https://github.com/rwightman/pytorch-image-models)
Generic class model model_timm to call timm models by specifying in the __model_library__ field the desired architecture (e.g. a resnet 50 of timm)
```yaml
model_config:
  # key of model
  model_timm:
    # key of model
    model: model_timm
    # name of the architecture in library timm
    model_library: resnet50 
    # flag to use pretraining of timm
    pretrained: false
    losses:
      # key of a loss function
      - type: cross_entropy
```
* [torchvision](https://pytorch.org/vision/stable/models.html)
Generic class model model_torchvision to call timm models by specifying in the __model_library__ field the desired architecture (e.g. a resnet 50 of timm)
```yaml
model_config:
  # key of model
  model_torchvision:
    # key of model
    model: model_torchvision
    # name of the architecture in library torchvision
    model_library: resnet50 
    # flag to use pretraining of torchvision
    pretrained: false
    losses:
      # key (from gaif or mmf) of a loss function
      - type: cross_entropy
```
* [torchaudio](https://pytorch.org/audio/stable/models.html)

## Weights
* [CIFAR-10/CIFAR-100](https://github.com/chenyaofo/pytorch-cifar-models) (supervised)
* [Places 365](https://github.com/CSAILVision/places365) (supervised)
* [lightly](https://github.com/lightly-ai/lightly) (self-supervised) (Not implemented yet)

Note: If you create a custom model, you can create your own fields for the configuration file. Otherwise, you have to check the documentation of the gaif/mmf model to get to check the fields to be filled in.

__Tutoriels from MMF__:

* [How to add a custom model ?](https://mmf.sh/docs/tutorials/concat_bert_tutorial)
* [How to use pretrained weighs ?](https://mmf.sh/docs/tutorials/checkpointing)
# Backbones/Encoders

We have added to the encoders of mmf the encoders of timm

| Name_Backbone | type | Key_Word_Registry | Description |
| ---- | --- | ---- | --- | 
| [**All encoders from timm**](https://github.com/rwightman/pytorch-image-models) | Image | timm_encoder | A large collection of image models which can be used as encoders |

Config of a timm_encoder:
```yaml
type: timm_encoder 
params:
  model_library: resnet18
  pretrained: false 
  # set fpn to true to get a FPN 
  fpn: False
  # set forward_features to true to get the embeddings
  forward_features: true

  # To change the tensor shape of the ouput like this Batch_sizex2048x7x7 -> Batch_sizexnum_output_featuresx2048
  ppol: true
  pool_type: avg # avg or max 
  num_output_features: 1

  # Specify the path of the pre-training weights file if needed, otherwise put the value null
  pretrained_model: PATH_TO/pretrained_weights.pth 
  freeze:
    use_freeze: true 
    # layers to freeze have to be declared in a list (e.g: [5,6,7]), -1 value will freeze the whole backbone
    layers_to_freeze: -1

```

Notes: FPN is not available to all timm encoders (if FPN is set to true, a NotImplementedError whill be raised)
# Self-Supervised training
Self-supervised learning in GAIF is based on [lightly](https://github.com/lightly-ai/lightly)

For instance with simclr:
```yaml
simclr:
   model: simclr
   num_ftrs: 512
   losses:
     - type: NTXentLoss
       params: 
         temperature: 0.5
   
   image_encoder:
     type: timm_encoder 
     params:
       model_library: resnet18
       pretrained: false 
       fpn: False
       forward_features: true
       pool_type: avg
       num_output_features: 1
       pretrained_model: weights/resnet18_simclr_lighlty.pth
      freeze:
        use_freeze: true 
        layers_to_freeze: -1
```
## Trainer

GAIF proposes a new trainer lightning_gaif which is based on the trainer lightning of mmf with new features (model summary with torchinfo, inference,...)

| Name_Trainer | type| Key_Word_Registry | Description |
| ---- | --- | ---- | --- | 
| GAIF_lightning_trainer | pytorch_lightning | lightning_gaif | GAIF proposes a new trainer lightning_gaif which is based on the trainer lightning of mmf with new features (model summary with torchinfo, inference,...) |



Extensions like deepspeed will be available.

## License
## Citations
GAIF use API of 
* models:
[timm](https://github.com/rwightman/pytorch-image-models), 
[torchvision](https://pytorch.org/vision/0.8/index.html), 
[torchaudio](https://pytorch.org/audio/stable/index.html),
[transformers](https://huggingface.co/transformers/)

* weights:
[pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
* training: 
[lightly](https://github.com/lightly-ai/lightly)

Authors of MMF
```bibtex
@misc{singh2020mmf,
  author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                 Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  title =        {MMF: A multimodal framework for vision and language research},
  howpublished = {\url{https://github.com/facebookresearch/mmf}},
  year =         {2020}
}
```

Authors of Augly
``` bibtex
@misc{bitton2021augly,
    author       = {Joanna Bitton and Zoe Papakipos},
    title        = {AugLy: A data augmentations library for audio, image, text, and video.},
    year         = {2021},
    howpublished = {\url{https://github.com/facebookresearch/AugLy}},
    doi          = {10.5281/zenodo.5014032}
    }
```
