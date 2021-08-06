# Processor

<p>With the GAIF framework's three custom processors: <strong>augly_image_transforms</strong>, <strong>augly_audio_transforms</strong>, <strong>augly_text_transforms</strong> , GAIF can add all transforms function of the augly and the torch series.
<p>The processors will implement the required Data Augmentation by simply adding the name of the transforms and the required parameters to the config. Here is an example: 

```yaml
    
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
Of course, in addition to Data Augmentation, we can also use some special processors, such as some of MMF's processors or custom processors, or use multiple processors at the same time.
For example:
```yaml
    
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
          text_processor:
                type: vocab
                params:
                    max_length: 10
                    vocab:
                        type: random
                        vocab_file: yahoo_answers_csv/words.txt
```
For more information about MMF's processors and their uses, please click here: [MMF_Processors](https://github.com/facebookresearch/mmf/tree/master/mmf/datasets/processors)

If you want to see all valid transforms, We offer two approaches:
  - Use this custom commande to find out it:
  ```sh
    python projects/data_augmentation/run_processors.py
  ```
  ![Transform illustration](./transform_illustration.png)
  - Or look at these two lien for more informations: [Augly_transforme](https://github.com/facebookresearch/AugLy) and [Torch_transforme](https://pytorch.org/vision/stable/transforms.html)
  
For those who wish to learn more about the dataset part of the GAIF framework or have unanswered questions, we have a notebook that shows the process of building a custom dataset from start to finish. Please look at here: [Dataset_Detail](/Dataset_Example.ipynb)
