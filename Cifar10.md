# Generic AI Framework (GAIF)

GAIF is an extension of [MMF](https://github.com/facebookresearch/mmf) which brings additional models, datasets, optimizers and trainers

GAIF brings more vision models, diversifies training techniques (self-supervised learning, distillation, ...) by incorporating api from other frameworks.

Please check the documentation of [MMF](https://mmf.sh/).
For a quick introduction to MMF, please check this [notebook](https://colab.research.google.com/github/facebookresearch/mmf/blob/notebooks/notebooks/mmf_hm_example.ipynb)

<!-- toc -->

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

registry.mapping["state"] = {}

opts = [
    "config='PATH_TO/config.yaml'",
    "model=vgg16", 
    "dataset=classification_cifar10",
]

run(opts=opts)
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
=======
# Models (for supervised and self-supervised learning)
| Name Model | Type | Theme | Key Word Registry | Description |
>>>>>>> 0085f6dc490202d9db226aec992dadeb66620ef8
| ---- | --- | ---- | --- | ---- |
| [**All models from timm**](https://github.com/rwightman/pytorch-image-models) | Image | Classification | model_timm | A large collection of image models |
| [**All classification models from torchvision**](https://pytorch.org/vision/stable/models.html) | Image | Classification | model_torchvision |  A large collection of image models | 
| [**Efficientnet**](https://github.com/lukemelas/EfficientNet-PyTorch) | Image| Classification | efficientnet |Implementation of a Efficientnet by @lukemelas |
| [**Simclr from lightly**](https://github.com/lightly-ai/lightly)| Image| Self-Supervised Learning| simclr| Implementation of simclr training by [lightly](https://github.com/lightly-ai/lightly)| 

[More details and how to use these models](./gaif/models/README.md)

# Encoders

We have added to the encoders of mmf the encoders of timm

| Name_ENcoder | type | Key_Word_Registry | Description |
| ---- | --- | ---- | --- | 
| [**All encoders from timm**](https://github.com/rwightman/pytorch-image-models) | Image | timm_encoder | A large collection of image models which can be used as encoders |

[More details and how to use these encoders](./gaif/modules/README.md)

# Self-Supervised training
Self-supervised learning in GAIF is based on [lightly](https://github.com/lightly-ai/lightly)

We use lightly models (simclr, byol,...) which will wrap an encoder of MMF/GAIF to be train with Self-SUpervised Learning

For instance to training an encoder resnet 18 (from timm_encoder) with simclr:
```yaml
simclr:
   model: simclr
   num_ftrs: 512
   losses:
     - type: NTXentLoss
       params: 
         temperature: 0.5

   # encoder
   image_encoder:
     type: timm_encoder 
     params:
       model_library: resnet18
       pretrained: false 
       fpn: False
       forward_features: true
       pool_type: avg
       num_output_features: 1
       pretrained_model: PATH_TO/weights.pth
      freeze:
        use_freeze: false 
        layers_to_freeze: null
```
## Trainer

GAIF proposes a new trainer lightning_gaif which is based on the trainer lightning of mmf with new features (model summary with torchinfo, inference,...)

| Name_Trainer | type| Key_Word_Registry | Description |
| ---- | --- | ---- | --- | 
| GAIF_lightning_trainer | pytorch_lightning | lightning_gaif | GAIF proposes a new trainer lightning_gaif which is based on the trainer lightning of mmf with new features (model summary with torchinfo, inference,...) |

## Examples
###Training from scratch a model 

For training from scratch vgg16

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
  trainer: lightning # key of the trainer 
  seed: 1
  batch_size: 32
  max_epochs: 100
  tensorboard: false
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
## For Developpers
Please note that we patch some modules of MMF to merge features of GAIF with MMF API: 
* [batch_collator](./gaif/common/patches/patch_batch_collator.py)
* [registry](./gaif/common/patches/patch_batch_collator.py)
* [base_model](./gaif/models/patches/patch_base_model.py)
* [encoder](./gaif/modules/patch_encoder.py)
* [logger](./gaif/utils/patches/patch_logger.py)
* [mmf_cli](./gaif_cli/patches/patch_cli.py)

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
