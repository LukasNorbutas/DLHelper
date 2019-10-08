from typing import *
from pathlib import *

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .Resizer import Resizer
from .Augmentor import Augmentor
from .DataRaw import DataRaw

class DataSplit:
    """
    This class takes a DataRaw instance as input and creates a train/validation/test split as tf Datasets, which
    can then be passed to keras.model.fit, or the CNNLearner class of this module.
    The class involves the entire pre-processing pipeline, including augmentation, up/downsampling and resizing.
    The original dataframe, train/val/test dataframes and the new Dataset generators are stored in the DataSplit
    instance.

    # Example
        # Creating a data split based on DataRaw instance:
        my_data = DataRaw(data_dir, filetype="png").init_df().encode()
        my_data_split = DataSplit(my_data, val_test_size=(0.2, 0.1), batch_size=8 ... )

        # Passing an Augmentor to the pipeline:
        my_augmentor = Augmentor(h_flip=0.5, v_flip=0.25, brightness=(0.5, 0.3))
        my_data_split = DataSplit(my_data, val_test_size=(0.1, 0.05), aug=my_augmentor)

        # Resize content images at any time after initialization:
        my_data_split.resize(img_dims=(100,100,3), resize="random_crop")

    # Arguments
        data: a DataRaw instance that contains a dataframe with image locations and labels (DataRaw.df[["id", "label"]]).
        val_test_size: the size of validation and test datasets, expresseed as fractions of the total number of samples.
        aug: Augmentor class instance that contains the image augmention pipeline.
        downsample: downsampling of over-represented label classes during batching. The argument takes a list of floats as
            the target distribution of label classes, where len(list) == n_classes, e.g. 4-classes: [0.25, 0.2, 0.3, 0.25].
        upsample: upsampling of under-represented label classes before batching. Duplicates under-represented classes in
            the dataframe before feeding it to a tf dataset generator. Takes a float value that expresses the % decrease of
            class imbalance in the output dataset (1 = all classes equal, 0 = no change in class distribution).
        batch_size: batch size of the output train, validation and test datasets.
        train_shuffle: enable/disable shuffling of the train dataset generator.
        prefetch: enable batch prefetching (pre-loading the next batch of data while GPU is busy with the previous one)
        num_parallel_calls: number of parallel calls for the augmentation/image reading pipeline, which is executed on CPU
            cores.
    """

    def __init__(self,
        data: DataRaw,
        val_test_size: Tuple[float, float],
        aug: Union[Augmentor, None] = None,
        downsample: Union[List[float], None] = None,
        upsample: float = 0.,
        batch_size: int = 8,
        train_shuffle: bool = True,
        prefetch: int = 1,
        num_parallel_calls: int = -1):

        self.data = data
        self.val_test_size = val_test_size
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.prefetch = prefetch
        self.num_parallel_calls = num_parallel_calls
        self.aug = aug
        self.downsample = downsample
        self.upsample = upsample
        self.img_dims = None

        # Pipeline for regular RGB images
        self._dataset_constructor(self.val_test_size, self.train_shuffle, self.batch_size, self.aug,
                                 self.downsample, self.upsample, self.prefetch, self.num_parallel_calls)


    def _dataset_constructor(self,
        val_test_size: Tuple[float, float],
        train_shuffle: bool,
        batch_size: int,
        aug: Augmentor,
        downsample: List[float],
        upsample: float,
        prefetch: int,
        num_parallel_calls: int) -> None:
        """
        Constructs a tf dataset based on a dataframe in the input DataRaw instance. The inputed dataframe
        is: 1) split into train/validation/test dataframes based on val_test_size; 2) train dataframe upsampled
        if necesary; 3) train/val/test dataframes get routed via df_to_ds() function, where dataframes are
        passed to tf.data.Dataset and train dataset gets augmented + downsampled if necessary; 4) The train
        dataset is shuffled if necessary; 5) Train/val/test datasets are assigned an iterator, batched and
        stored in the instance as DataSplit.train, DataSplit.val, DataSplit.test.
        """
        images = pd.Series(self.data.df.id.unique())
        train_images = images.sample(round(len(images)*(1-sum(val_test_size))))
        val_images = (
            images[~images.isin(train_images)]
            .sample(round((len(images)-len(train_images))*val_test_size[0]/sum(val_test_size)))
        )
        test_images = images[(~images.isin(train_images)) & (~images.isin(val_images))]

        self.train_dataframe = self.data.df[self.data.df.id.isin(train_images)]
        if upsample > 0:
            self.train_dataframe = self._upsampler(self.train_dataframe, upsample)

        self.val_dataframe = self.data.df[self.data.df.id.isin(val_images)]
        self.test_dataframe = self.data.df[self.data.df.id.isin(test_images)]

        self.train = self._df_to_ds(self.train_dataframe, aug=aug, downsample=downsample)
        self.val = self._df_to_ds(self.val_dataframe)
        self.test = self._df_to_ds(self.test_dataframe)

        if train_shuffle:
            self.train = self.train.shuffle(self.batch_size)

        self.train = self.train.repeat().batch(batch_size).prefetch(prefetch)
        self.val = self.val.repeat().batch(batch_size).prefetch(prefetch)
        self.test = self.test.repeat().batch(batch_size).prefetch(prefetch)
        self._steps_calc()


    def _df_to_ds(self,
        data: pd.DataFrame,
        aug: Union[Augmentor, None] = None,
        resize: Union[Resizer, None] = None,
        downsample: Union[List[float], None] = None) -> tf.data.Dataset:
        """
        Converts a dataframe with image paths and labels to a tf.data.Dataset. Maps image reader,
        resizer, augmentor and downsampler.
        """
        img_ds = tf.data.Dataset.from_tensor_slices(data.id.values)
        img_ds = img_ds.map(lambda x: self._img_reader(x), num_parallel_calls=self.num_parallel_calls)
        if resize:
            img_ds = img_ds.map(lambda x: resize(x), num_parallel_calls=self.num_parallel_calls)
        if aug:
            img_ds = img_ds.map(aug, num_parallel_calls=self.num_parallel_calls)
        label_ds = tf.data.Dataset.from_tensor_slices(data.label.values)
        ds = tf.data.Dataset.zip((img_ds, label_ds))
        if downsample:
            resampler = self._downsampler(target_dist=downsample)
            ds = ds.apply(resampler).map(lambda x,y: y)
        return ds


    def _img_reader(self, filename: str) -> tf.Tensor:
        """
        Function that reads **and normalizes /255** an image file.
        """
        image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
        image = tf.cast(image, tf.float32)
        if image.shape[2] == 1:
            image = tf.reshape(image, (image.shape[0], image.shape[1]))
        image /= 255
        return image


    def _steps_calc(self) -> None:
        """
        Function calculates the number of steps in an epoch of training/evaluating, based on the number of examples
        and batch_size, stored in the DataSplit instance.
        """
        self.train.steps = math.ceil(len(self.train_dataframe)/self.batch_size)
        self.val.steps = math.ceil(len(self.val_dataframe)/self.batch_size)
        self.test.steps = math.ceil(len(self.test_dataframe)/self.batch_size)


    def show(self, cols: int = 8, n_batches: int = 1) -> None:
        """
        Displays several example images and labels in the training dataset.
        """
        if cols >= self.batch_size * n_batches:
            cols = self.batch_size * n_batches
            rows = 1
        else:
            rows = math.ceil(self.batch_size * n_batches / cols)
        _, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        i = 0
        for x_batch, y_batch in self.train.take(n_batches):
            for (x, y) in zip(x_batch.numpy(), y_batch.numpy()):
                idx = (i // cols, i % cols) if rows > 1 else i % cols
                ax[idx].axis("off")
                ax[idx].imshow(x)
                ax[idx].set_title(y)
                i += 1


    def _downsampler(self, target_dist: List[float]):
        """
        Takes a target distribution of classes, where len(target_dist) == n_classes in the data.
        Outputs a downsampling function that is applied to the train dataset during df_to_ds().
        During training, the data will be batched based on the target distribution.
        WARNING: ** slows batching down if class imbalance is high **
        """
        resampler = tf.data.experimental.rejection_resample(
            class_func=lambda x, label: label,
            target_dist= tf.constant(target_dist, dtype=tf.float32),
            seed=20
        )
        return resampler


    def _upsampler(self,
        df: pd.DataFrame,
        alpha: float) -> pd.DataFrame:
        """
        Upsamples (duplicates) under-represented label classes by a factor of alpha. 0 <= alpha <= 1,
        where 1 duplicates under-represented cases to uniform class distribution, and 0 does not duplicate
        any cases.
        """
        initial_dist = pd.DataFrame(df.label.value_counts())
        initial_dist["diff_largest"] = np.round((initial_dist.label.max() - initial_dist.label) * alpha).astype(np.int)
        for i in initial_dist.index:
            upsampled_copies = df.loc[df.label == i, :].sample(initial_dist.loc[i,"diff_largest"], replace=True)
            df = df.append(upsampled_copies)
        df = df.sample(frac=1)
        return df


    def reaugment(self,
        aug: Augmentor) -> None:
        """
        This function reaugments the DataSplit.train dataset with a new augmentor, and stores the new dataset
        and augmentor in the class instance as DataSplit.train and DataSplit.aug.
        """
        self.aug = aug
        if self.img_dims:
            resizer = Resizer(self.img_dims, self.resize_mode, self.crop_adjustment)
        else:
            resizer = None
        self.train = self._df_to_ds(self.train_dataframe, aug=aug, resize=resizer,
                                    downsample=self.downsample)
        if self.train_shuffle:
            self.train = self.train.shuffle(self.batch_size)
        self.train = self.train.repeat().batch(self.batch_size).prefetch(self.prefetch)
        self._steps_calc()


    def resize(self,
        img_dims: Tuple[int, int, int],
        resize: str,
        crop_adjustment: float = 1.) -> None:
        """
        Resizes the train/val/test datasets, and pushes it through the original pre-processing pipeline
        (augmentation, downsampling, ...). The resulting datasets are identical to the originally stored train/val/test
        dataset, except for image size. The datasets are stored in the class instance.

        # Arguments
        img_dims: target image dimensions (channel last)
        resize: "stretch"/"crop"/"random_crop" - methods of resizing, passed to the Resizer class
        crop_adjustment: pre-stretches/pre-shrinks image before cropping by height*alpha and width*alpha.
        """
        height, width, _ = img_dims
        self.img_dims = (height, width, 3)
        self.resize_mode = resize
        self.crop_adjustment = crop_adjustment

        resizer = Resizer(img_dims, resize, crop_adjustment)
        self.train = self._df_to_ds(self.train_dataframe, aug=self.aug,
                                   resize=resizer, downsample=self.downsample)
        self.val = self._df_to_ds(self.val_dataframe, resize=resizer)
        self.test = self._df_to_ds(self.test_dataframe, resize=resizer)

        if self.train_shuffle:
            self.train = self.train.shuffle(self.batch_size)

        self.train = self.train.repeat().batch(self.batch_size).prefetch(self.prefetch)
        self.val = self.val.repeat().batch(self.batch_size).prefetch(self.prefetch)
        self.test = self.test.repeat().batch(self.batch_size).prefetch(self.prefetch)
        self._steps_calc()
