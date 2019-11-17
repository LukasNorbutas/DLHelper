import imgaug.augmenters as iaa
import tensorflow as tf
from typing import *

class Augmentor:
    """
    This class is used to define the image augmentation pipeline. Augmentor class instance
    can be passed to DataSplit "aug" argument to apply it to the train dataset.

    # Arguments
    h_flip: probability of applying horizontal flip to a batch image.
    v_flip: probability of applying vertical flip to a batch image.
    brightness: (probability, alpha) - tuple defines probability and the amount of brightness (+- alpha%) to apply to an image.
    contrast: (probability, alpha) - tuple defines probability and the amount of contrast (+- alpha%) to apply to an image.
    blur: probability of applying blur and the amount of blur to have in a tuple (from sigma, to sigma), e.g. 50% images get
        blurred between 0 and 3 sigma, blur = (0.5, (0., 3.))
    hue: probability of applying hue shift (tf implementation) and alpha (amount of the shift).
    warp: the amount of random warp to apply
    pipeline: custom pipeline (currently not supported)
    """
    def __init__(self,
        h_flip: float = 0.,
        v_flip: float = 0.,
        brightness: Tuple[float, float] = (0., 0.),
        contrast: Tuple[float, float] = (0., 0.),
        hue: Tuple[float, float] = (0., 0.),
        sharpness: Tuple[float, float] = (0., 0.),
        blur: Tuple[float, Tuple[float, float]] = (0., 0.),
        warp: float = 0.,
        zoom: Tuple[float, float] = (0., 0.),
        rotate: Tuple[float, float] = (0., 0.),
        dropout: Tuple[float, float] = (0., 0.),
        pipeline: Optional[List[Any]] = None):

        self.h_flip = h_flip
        self.v_flip = v_flip
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.sharpness = sharpness
        self.blur = blur
        self.warp = warp
        self.zoom = zoom
        self.rotate = rotate
        self.dropout = dropout
        self.pipeline = pipeline

    def __call__(self,
        image: tf.Tensor,
        labels: Optional[Any] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, Any]]:
        """
        Upon calling takes image and applies transformation. If labels are passed in the input tuple,
        those are appended in a tuple with the augmented image upon return.

        Note: Custom pipelines not yet implemented.
        """

        # Tensorflow augmentors
        image = tf.cond(tf.random.uniform([], 0, 1) < self.h_flip,
                        lambda: tf.image.flip_left_right(image),
                        lambda: image)

        image = tf.cond(tf.random.uniform([], 0, 1) < self.v_flip,
                        lambda: tf.image.flip_up_down(image),
                        lambda: image)

        image = tf.cond(tf.random.uniform([], 0, 1) < self.brightness[0],
                        lambda: tf.image.random_brightness(image, max_delta=self.brightness[1]),
                        lambda: image)

        image = tf.cond(tf.random.uniform([], 0, 1) < self.contrast[0],
                        lambda: tf.image.random_contrast(image, lower=0.99999-self.contrast[1], upper=1+self.contrast[1]),
                        lambda: image)

        # Hue
        image = tf.cond(tf.random.uniform([], 0, 1) < self.hue[0],
                       lambda: tf.image.random_hue(image, self.hue[1], seed=None),
                       lambda: image)


        # Third-party module augmentors. Applied using py_func, require to get_shape of the
        # input, set_shape before returning.

        input_shape = image.get_shape()

        # Blur
        if self.blur[0]:
            augmenter = iaa.Sometimes(self.blur[0], iaa.GaussianBlur(self.blur[1]))
            img_exp = tf.expand_dims(image, axis=0)
            image = tf.py_function(augmenter.augment_images, [img_exp], Tout=[tf.float32])
            image = tf.squeeze(image)

        # Sharpen
        if self.sharpness[0]:
            augmenter = iaa.Sometimes(self.sharpness[0], iaa.Sharpen(alpha=(0, self.sharpness[1]), lightness=(0.75, 1.5)))
            img_exp = tf.expand_dims(image, axis=0)
            image = tf.py_function(augmenter.augment_images, [img_exp], Tout=[tf.float32])
            image = tf.squeeze(image)

        # Perspective transform
        if self.warp:
            def augmenter(image):
                return iaa.PerspectiveTransform(scale=(0, self.warp), keep_size=True).augment_images(image)
            img_exp = tf.expand_dims(image, axis=0)
            image = tf.py_function(augmenter, [img_exp], Tout=[tf.float32])
            image = tf.squeeze(image)

        # Zoom
        if self.zoom[0]:
            augmenter = iaa.Sometimes(self.zoom[0],
                                      iaa.Affine(scale=(1-self.zoom[1],1+self.zoom[1]), mode="edge"))
            img_exp = tf.expand_dims(image, axis=0)
            image = tf.py_function(augmenter.augment_images, [img_exp], Tout=[tf.float32])
            image = tf.squeeze(image)

        # Rotate
        if self.rotate[0]:
            augmenter = iaa.Sometimes(self.rotate[0],
                                      iaa.Affine(rotate=(-self.rotate[1], self.rotate[1]),
                                                mode="edge"))
            img_exp = tf.expand_dims(image, axis=0)
            image = tf.py_function(augmenter.augment_images, [img_exp], Tout=[tf.float32])
            image = tf.squeeze(image)

        # Dropout
        if self.dropout[0]:
            augmenter = iaa.Sometimes(self.dropout[0], iaa.CoarseDropout(p=self.dropout[1], size_percent=0.10))
            img_exp = tf.expand_dims(image, axis=0)
            image = tf.py_function(augmenter.augment_images, [img_exp], Tout=[tf.float32])
            image = tf.squeeze(image)

        image.set_shape(input_shape)

        if self.pipeline:
            print("Not yet implemented")
        image = tf.clip_by_value(image, 0.0, 1.0)
        if labels is not None:
            return (image, labels)
        else:
            return image
