from typing import *

import tensorflow as tf


class Resizer:
    """
    Resizes a given image to target dimensions using one of the provided methods. If
    labels are passed with the input image, they are appended in a tuple with the image
    upon return.

    This is blatantly copied from: https://github.com/mokahaiku/toai/

    # Arguments
        img_dims: target image dimensions (channel last)
        resize: "stretch"/"crop"/"random_crop" - methods of resizing, passed to the Resizer class
        crop_adjustment: pre-stretches/pre-shrinks image before cropping by height*alpha and width*alpha.
    """
    def __init__(self,
        img_dims: Tuple[int, int, int],
        resize: str,
        crop_adjustment: float = 1.0):

        self.img_dims = img_dims
        self.resize = resize
        self.crop_adjustment = crop_adjustment

    def __call__(self,
        image: tf.Tensor,
        label: Optional[Any] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, Any]]:

        height, width, _ = self.img_dims
        if self.resize == "stretch":
            image = tf.image.resize(images=image, size=(height, width))
        elif self.resize == "crop":
            crop_height, crop_width = [
                int(x * self.crop_adjustment) for x in (height, width)
            ]
            image = tf.image.resize(
                images=image, size=(crop_height, crop_width), preserve_aspect_ratio=True
            )
            image = tf.image.resize_with_crop_or_pad(image, height, width)
        elif self.resize == "random_crop":
            crop_height, crop_width = [
                int(x * self.crop_adjustment) for x in (height, width)
            ]
            image = tf.image.resize(image, (crop_height, crop_width))
            image = tf.image.random_crop(image, self.img_dims)
        if label is not None:
            return (image, label)
        else:
            return image
