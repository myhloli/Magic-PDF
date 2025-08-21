from typing import Dict, Union, Optional, List

from torch import TensorType
from transformers import DonutImageProcessor, DonutProcessor
from transformers.image_processing_utils import BatchFeature, BaseImageProcessor
from transformers.image_transforms import pad
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, \
    valid_images, to_numpy_array, is_scaled_image, get_image_size
import numpy as np
import PIL
import logging

logger = logging.getLogger()

IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_MEAN = [0.485, 0.456, 0.406]


class VariableDonutImageProcessor(DonutImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def numpy_resize(self, image: np.ndarray, size, resample):
        image = PIL.Image.fromarray(image)
        resized = self.pil_resize(image, size, resample)
        resized = np.array(resized, dtype=np.uint8)
        resized_image = resized.transpose(2, 0, 1)

        return resized_image

    def pil_resize(self, image: PIL.Image.Image, size, resample):
        width, height = image.size
        max_width, max_height = size["width"], size["height"]
        if width != max_width or height != max_height:
            # Shrink to fit within dimensions
            width_scale = max_width / width
            height_scale = max_height / height
            scale = min(width_scale, height_scale)

            new_width = min(int(width * scale), max_width)
            new_height = min(int(height * scale), max_height)

            image = image.resize((new_width, new_height), resample)

        image.thumbnail((max_width, max_height), resample)

        assert image.width <= max_width and image.height <= max_height

        return image

    def process_inner(self, images: List[List], train=False):
        # This will be in list of lists format, with height x width x channel
        assert isinstance(images[0], (list, np.ndarray))

        # convert list of lists format to array
        if isinstance(images[0], list):
            # numpy unit8 needed for augmentation
            np_images = [np.array(img, dtype=np.uint8) for img in images]
        else:
            np_images = [img.astype(np.uint8) for img in images]

        assert np_images[0].shape[2] == 3  # RGB input images, channel dim last

        # This also applies the right channel dim format, to channel x height x width
        np_images = [self.numpy_resize(img, self.max_size, self.resample) for img in np_images]
        assert np_images[0].shape[0] == 3  # RGB input images, channel dim first

        # Convert to float32 for rescale/normalize
        np_images = [img.astype(np.float32) for img in np_images]

        # Pads with 255 (whitespace)
        # Pad to max size to improve performance
        max_size = self.max_size
        np_images = [
            self.pad_image(
                image=image,
                size=max_size,
                random_padding=train,  # Change amount of padding randomly during training
                input_data_format=ChannelDimension.FIRST,
                pad_value=255.0
            )
            for image in np_images
        ]

        # Rescale and normalize
        np_images = [
            self.rescale(img, scale=self.rescale_factor, input_data_format=ChannelDimension.FIRST)
            for img in np_images
        ]
        np_images = [
            self.normalize(img, mean=self.image_mean, std=self.image_std, input_data_format=ChannelDimension.FIRST)
            for img in np_images
        ]

        return np_images

    def preprocess(
            self,
            images: ImageInput,
            do_resize: bool = None,
            size: Dict[str, int] = None,
            resample: PILImageResampling = None,
            do_thumbnail: bool = None,
            do_align_long_axis: bool = None,
            do_pad: bool = None,
            random_padding: bool = False,
            do_rescale: bool = None,
            rescale_factor: float = None,
            do_normalize: bool = None,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
            **kwargs,
    ) -> PIL.Image.Image:
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # Convert to numpy for later processing steps
        images = [to_numpy_array(image) for image in images]

        images = self.process_inner(images, train=False)

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def pad_image(
            self,
            image: np.ndarray,
            size: Dict[str, int],
            random_padding: bool = False,
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
            pad_value: float = 0.0,
    ) -> np.ndarray:
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        assert delta_width >= 0 and delta_height >= 0

        if random_padding:
            pad_top = np.random.randint(low=0, high=delta_height + 1)
            pad_left = np.random.randint(low=0, high=delta_width + 1)
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format,
                   constant_values=pad_value)


class VariableDonutProcessor(DonutProcessor):
    def __init__(self, image_processor=None, tokenizer=None, train=False, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False
        self.train = train

    def __call__(self, *args, **kwargs):
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None:
            raise ValueError("You need to specify images to process.")

        inputs = self.image_processor(images, *args, **kwargs)
        return inputs

import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision.transforms.functional import resize
from omegaconf import OmegaConf
import albumentations as alb
from albumentations.pytorch import ToTensorV2


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)
    
class FormulaImageBaseProcessor(BaseProcessor):

    def __init__(self, image_size):
        super(FormulaImageBaseProcessor, self).__init__()
        self.input_size = [int(_) for _ in image_size]
        assert len(self.input_size) == 2

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def prepare_input(self, img: Image.Image, random_padding: bool = False):
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return

        if img.height == 0 or img.width == 0:
            return

        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return ImageOps.expand(img, padding)
    
class FormulaImageEvalProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size):
        super().__init__(image_size)

        self.transform = alb.Compose(
            [
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.transform(image=np.array(image))['image'][:1]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [384, 384])

        return cls(image_size=image_size)