import collections
import os

from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse

from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image


class _Pascal_Voc(VisionDataset):
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        valid_image_sets = ["train", "val", "test"]
       
        if image_set not in valid_image_sets:
            raise ValueError("Unknown value '{image_set}' for argument image_set. Valid values are {{{valid_image_sets}}}.")
        
        self.image_set = image_set

        voc_root = os.path.join(self.root, self.image_set)
        
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted.") # You can use download=True to download it
        
        self.images, self.targets = self._getFil(voc_root)
        
        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)
    
    def _getFil(self, voc_root):
        image_files = [ f for f in os.listdir(voc_root) if f.endswith('.jpg')]
        annotation_files = [f.replace(os.path.splitext(f)[1], self._TARGET_FILE_EXT) for f in image_files]
        
        for fil in annotation_files:
            if not os.path.isfile(os.path.join(voc_root, fil)):
                raise FileNotFoundError(f"Annotation file not found: {fil}")
                
        image_files = [os.path.join(voc_root, img) for img in image_files]
        annotation_files = [os.path.join(voc_root, anto) for anto in annotation_files]
        
        return image_files, annotation_files

class VOCSegmentation(_Pascal_Voc):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"val"`` or ``"test"``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    _TARGET_FILE_EXT = ".png"

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

class VOCDetection(_Pascal_Voc):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"val"`` or ``"test"``.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    _TARGET_FILE_EXT = ".xml"

    @property
    def annotations(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = List(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict