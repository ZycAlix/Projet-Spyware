# Dataset
A number of datasets have been installed for testing and experience with the GAIF framework, as listed below:
| Name_Dataset | Type | THEME | Key_Word_Registry | Download | Description |
| ---- | --- | ---- | --- | ---- | ---- |
| [**YahooAnswers**](https://paperswithcode.com/sota/text-classification-on-yahoo-answers) | TEXT | Classification | classification_yahooanswers | :heavy_check_mark: | A dataset with Question and Answers in 10 Types Topic |
| [**IMDB**](https://paperswithcode.com/dataset/imdb-movie-reviews) | TEXT | Classification | classification_imdb | :heavy_check_mark: | A dataset with critics positive and negative for movies | 
| [**Balloon**](https://github.com/matterport/Mask_RCNN) | IMAGE | Detection | detection_balloon | :heavy_check_mark: | A dataset of photos with a lot of balloon to find out |
| **Poc_mons** | IMAGE | Detection | detection_poc_mons | :x: | A dataset prive of photos in Mons to look for mans and vehicules |
| [**Cifar10**](https://paperswithcode.com/dataset/cifar-10) | IMAGE | Classification | classification_cifar10 | :heavy_check_mark: | A dataset with 10 differents category objects |
| [**Omniglot**](https://paperswithcode.com/dataset/omniglot-1) | IMAGE | Classification | classification_omniglot | :heavy_check_mark: | A dataset of hand-written characters with 1623 characters and 20 examples for each character |
| [**Gtzan**](https://paperswithcode.com/dataset/gtzan) | Audio | Classification | classification_gtzan | :heavy_check_mark: | A dataset in Musical genre classification of audio signals |

In addition to the above datasets, we can also find the required datasets from MMF, Torchvision, Torchtext and Torchaudio. MMF, as the main API of our framework, we have all the datasets recorded in the registry and just need to load the key_word of this required dataset by a simple script. For the other three APIs, including custom datasets need to create a special Builder and Dataset, and create a key_word private saved into the registry. Of course, all datasets require the configuration file .yaml to provide the required parameters. There will be specific examples in the Usage Section.

## Usage
As mentioned in the previous section, all datasets except MMF need to create a key_word and save it in the registry of GAIF framework. For example:
```python
@registry.register_builder("classification_yahooanswers")
class YAHOOANSWERSBuilder(BaseDatasetBuilder): 
....
```
Then, the GAIF framework will know that this key_word *classification_yahooanswers* is the class of the corresponding dataset and thus create the relevant object. Also, there is a function *config_path* (custom) in the Builder, which will provide the location of the dataset configuration file.yaml, so that GAIF can find the relevant parameters. Something like this:
``` python
@classmethod
    def config_path(cls):
        return "configs/datasets/yahooanswers/classification.yaml"
```

### MMF
  For MMF, it provides a number of datasets that can be used directly, including automatically downloaded data, and are registed in the GAIF framework. We give an example here:
```python
    from mmf.utils.build import build_dataset
    from gaif.utils.env import setup_imports
    from mmf.common.registry import registry
    
    setup_imports()

    dataset_key = "coco"
    dataset = build_dataset(dataset_key=dataset_key)
    print(dataset.__getitem__(6))
```
  The samples of this dataset are obtained directly by calling MMF's Builder and Dataset with the key_word of the MMF dataset. For more informations, please look at this : 
  - [*MMF_Dataset*](https://github.com/facebookresearch/mmf/tree/master/mmf/datasets/builders)
  - [*MMF_Colab*](https://colab.research.google.com/github/facebookresearch/mmf/blob/notebooks/notebooks/kdd_tutorial.ipynb#scrollTo=wu5o2DbhHp8M)

### Torchvision
  For the three sisters of the Torch series, we can use the datasets they generate directly, but we have to build Builders and Datasets for them separately, as described in the previous section. Here is an example: 
```python
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
    
    .... in Config.yaml:
    
    dataset_config:
    # You can specify any attributes you want, and you will get them as attributes
    # inside the config passed to the dataset. Check the Dataset implementation below.
    classification_yahooanswers:
        # Where your data is stored
        data_dir: ${env.data_dir}
        processors:
        
    .... 
    
```
In the example above, Torchvision provides a dataset class for Omniglot. We need to create the Builder, Dataset and profile.yaml for them separately.  *Notice*: The key_word in the configuration file and the key_word of the dataset should be consistent.
