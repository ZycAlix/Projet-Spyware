# generic-AI-framework (GAIF)

GAIF is an extension of [MMF](https://github.com/facebookresearch/mmf) which brings other models, datasets, optimizer and trainer.

GAIF brings more vision models, diversifies training techniques (self-supervised learning, distillation, ...) by incorporating api from other frameworks.

Please check the documentation of [MMto add GAIF functionality to MMFF](https://mmf.sh/).

<!-- toc -->

- [Prerequisties](#prerequisties)
- [Installation](#installation)
- [How to use](#how-to-use)
- [How to visualize dataset](#how-to-visualize-dataset)
- [How to use processor](#how-to-use-processor)
- [Models](#models)
- [License](#license)

<!-- tocstop -->

## Installation

GAIF was tested with **Ubuntu 20.04** and **python 3.8.10**

```sh
git clone https://github.com/nico-ri/generic-AI-framework
cd generic-AI-framework
pip install .
```

## How to use

GAIF use the core of MMF to operate. Please check this [notebook](https://colab.research.google.com/github/facebookresearch/mmf/blob/notebooks/notebooks/mmf_hm_example.ipynb) to have a quick introduction of MMF.

GAIF bring new models, datasets, optimizers, trainers, processors which can be called directely by MMF API. You have just to add this two following lines in you main.py to launch mmf training to to add GAIF functionality to MMF:

```python
from gaif.utils.env import setup_imports

setup_imports()
```

## How to visualize dataset

GAIF has scripts for some datasets, which can be viewed by running the following code directly:
```python
from gaif.utils.build import build_dataset
from gaif.utils.env import setup_imports


if __name__ == "__main__":

    setup_imports()

    dataset_key = "classification_cifar10"
    dataset = build_dataset(dataset_key=dataset_key)
    print(dataset.__getitem__(6))

```

*Finally, For more details about Dataset, please click [here](./doc/Dataset_README.md).* 

## How to use processor

For GAIF, If you wish to use a processor to process the data, this is an example: 

```python
transform_out = self.augly_image_transforms({"image": img})
img = transform_out["image"]
```
Here is config file: 
```yaml
dataset_config:
    # You can specify any attributes you want, and you will get them as attributes
    # inside the config passed to the dataset. Check the Dataset implementation below.
    classification_raf_basic:
        # Where your data is stored
        data_dir: ${env.data_dir}
        method: svp
        processors:
        # The processors will be assigned to the datasets automatically by GAIF
        # For example if key is text_processor, you can access that processor inside
        # dataset object using self.text_processor
          augly_image_transforms:
            type: augly_image_transforms
            params:
              transforms:
                - ToTensor
```
*Finally, For more details about Processor, please click [here](./doc/Proc_README.md).*


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


### Weights
* [CIFAR-10/CIFAR-100](https://github.com/chenyaofo/pytorch-cifar-models) (supervised)
* [Places 365](https://github.com/CSAILVision/places365) (supervised)
* [lightly](https://github.com/lightly-ai/lightly) (self-supervised) 

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

Notes: FPN is not available to all timm encoders (if FPN is set to true, a NotImplementedError will be raised) (please check the documenation of [timm](https://rwightman.github.io/pytorch-image-models/feature_extraction/))

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

## For Developpers
Please note that we patch some modules of MMF to merge features of GAIF with MMF API: 
* [batch_collator](./gaif/common/patches_batch_collator.py)
* [registry](./gaif/common/patches_batch_collator.py)


## Licenseo
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
