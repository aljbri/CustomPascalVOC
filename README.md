# Custom Pascal VOC PyTorch Module
This PyTorch module serves as a flexible  dataset reader for custom datasets created using the Pascal VOC format. It is a modified version of the original [Pascal VOC dataset reader](https://github.com/anthony-cabacungan/vision/blob/main/torchvision/datasets/voc.py) provided by PyTorch's *torchvision* package. The primary enhancement is the removal of folder structure restrictions, allowing users to read datasets organized in any manner while still leveraging the Pascal VOC annotation structure.

# Classes

The module includes two classes: `VOCDetection` and `VOCSegmentation`.

## **VOCDetection** : 
This class is designed for reading datasets following [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) Detection format. 

```Python
VOCDetection(root: str, image_set: str = 'train', transform: Optional[Callable] = None, 
             target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None)
```
**Parameters:**
* **root** (string): Root directory of the VOC Dataset.
* **image_set** (string, optional): Select the image_set to use, ``"train"``, ``"val"`` or ``"test"``.
* **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``.
* **target_transform** (callable, required): A function/transform that takes in the target and transforms it.
* **transforms** (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.

## **VOCSegmentation** 
This class is tailored for reading datasets structured in the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) Segmentation format.

```Python
VOCSegmentation(root: str, image_set: str = 'train', transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None)
```
**Parameters:**
* root (string): Root directory of the VOC Dataset.
* image_set (string, optional): Select the image_set to use, ``"train"``, ``"val"`` or ``"test"``.
* transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``.
* target_transform (callable, optional): A function/transform that takes in the target and transforms it.
* transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.


# Installing

You can install the package from [pypi](https://pypi.org/project/customVOC) using the following code:

    pip install --update customVOC

# Using customVOC

After installing the module, you can import either of the two classes: 'VOCDetection' and 'VOCSegmentation'.

```Python
from customVOC import VOCDetection
from customVOC import VOCSegmentation
```
Once imported, you can use these classes similarly to the ones from the torchvision package, without passing the year. When initializing, provide the root folder containing the dataset with one of the following subfolders: ``"train"``, ``"val"``, or ``"test"`` depending on the dataset portion you are using.

---
The following example loads the training dataset and converts the image to a tensor using ``ToTensor()``:
```Python
from customVOC import VOCDetection
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Create an instance of VOCDetection dataset.
voc_dataset = VOCDetection(root="datset_path",image_set="train", transform=ToTensor())

# Create a DataLoader for the VOCDetection dataset.
data_loader = DataLoader(voc_dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
```
> [!TIP]
>  *The collate function is used to customize how batches are created from individual samples in the dataset.*
---
> [!NOTE]  
> * The package is an absolute copy of the original code, with modifications to the way of reading image and annotation files locations.
> * The images and annotations files should be stored in the same folder, either train, val, or test, depending on what part of the dataset you have.
> * Only ***VOCDetection*** has been tested on reading datasets.
