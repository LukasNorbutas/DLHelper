import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from pathlib import *
from typing import *

from .LearningRateFinder import LearningRateFinder
from .DataSplit import DataSplit
from .TrainCycle import TrainCycle
from .discriminative_lr import DLR_Adam, get_lr_multipliers, layer_to_param_dict


class CNNLearner:
    """
    This class is used for training and evaluating a convolutional neural network on a DataSplit
    object. The model uses transfer learning, i.e. a base_model architecture from keras.applications
    or other implementations. The top layers contain concatenated GlobalAverage and GlobalMax
    pooling layers (applied to the output of the base_model), BatchNorm and Dropout. Output layer
    is defined by passing the output_layer argument.
    The class contains functionality for one_cycle learning, (un)freezing, discriminative learning
    rates.

    # Example:
        model_1 = CNNLearner(name="resnet50_1",
                        path=str(DATA_DIR),
                        data=my_data,
                        base_model=keras.applications.ResNet50,
                        input_shape=(100,100,3),
                        output_layer=keras.layers.Dense(
                            5,
                            kernel_regularizer=keras.regularizers.l1_l2(1e-6, 1e-3),
                            activation=keras.activations.softmax,
                        ),
                        dropout=0.5,
                        load=None)

        model_1.compile(optimizer="adam", lr=(1e-4,),  # pass a tuple for discr. lr (*0.3 lr for base_model)
                   loss='sparse_categorical_crossentropy',
                   metrics=['sparse_categorical_accuracy'])

        model_1.fit(epochs=1, name="test_run_01")

    # Arguments:
        path: path to the data folder, where folders for weights/architectures and logs will be created.
        name: name for the model's architecture.
        data: DataSplit object, that contains train/val/test generators.
        base_model: base_model for transfer learning.
        input_shape: expected input image shape for the model.
        output_layer: Keras.layer object that contains the final (output) layer of the model.
        dropout: the amount of dropout applied to the concatenated output of the base_model.
        load: pre-load model's weights from file.
    """
    def __init__(self,
        name: str,
        path: Union[Path, str],
        data: DataSplit,
        base_model: keras.Model,
        input_shape: Tuple[int, int, int],
        output_layer: List[keras.layers.Dense],
        dropout: Optional[float] = 0.0,
        load: Optional[bool] = False):

        self.path = Path(str(path))
        self.weights_path = Path(f"{self.path}/weights")
        self.arch_path = Path(f"{self.path}/arch")
        self.logs_path = Path(f"{self.path}/logs")

        self.name = name
        self.data = data
        self.n_classes = len(data.data.label_map)
        self.input_shape = input_shape
        self.output_layer = output_layer
        self.dropout = dropout
        self.transfer_architecture = base_model

        self.path.mkdir(parents=True, exist_ok=True)
        self.weights_path.mkdir(parents=True, exist_ok=True)
        self.arch_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        self._model_creator(name, base_model, input_shape, dropout, output_layer, load)


    def _model_creator(self,
        name: str,
        base_model: keras.Model,
        input_shape: Tuple[int, int, int],
        dropout: float,
        output_layer: List[keras.layers.Dense],
        load: bool) -> keras.Model:
        """
        Create a Keras model based on class initialization arguments and save model's architecture.
        """
        self.base_model = base_model(include_top=False, input_shape=(input_shape))
        concat_layer = keras.layers.concatenate(
            [
                keras.layers.GlobalAvgPool2D()(self.base_model.output),
                keras.layers.GlobalMaxPool2D()(self.base_model.output),
            ]
        )

        branch_1 = keras.layers.BatchNormalization()(concat_layer)
        branch_1 = keras.layers.Dropout(dropout)(branch_1)
        output_1 = self.output_layer[0](branch_1)

        if len(self.output_layer) > 1:
            branch_2 = keras.layers.BatchNormalization()(concat_layer)
            branch_2 = keras.layers.Dropout(dropout)(branch_2)
            output_2 = self.output_layer[1](branch_2)
            self.model = keras.models.Model(inputs=self.base_model.inputs, outputs=[output_1, output_2])

        if len(self.output_layer) == 1:
            self.model = keras.models.Model(inputs=self.base_model.inputs, outputs=output_1)

        if load:
            self.load(load)
            self.previous_weights = load

        self.save(filename=name, arch_only=True)

    def save(self, filename: str, arch_only: bool) -> None:
        """
        Save model's architecture/weights.
        """
        if arch_only == False:
            self.model.save_weights(f"{str(self.weights_path)}/{str(filename)}.h5")
        with open(f"{str(self.arch_path)}/{str(filename)}.json", "w") as f:
            f.write(self.model.to_json())

    def load(self, name: str) -> None:
        """
        Load model's weights.
        """
        self.model.load_weights(f"{str(self.weights_path)}/{str(name)}.h5")
        print(f"Note: {name} are loaded.")

    def compile(self,
        optimizer: keras.optimizers.Optimizer,
        lr: Union[float, Tuple[float], Tuple[float, float]],
        loss: keras.losses.Loss,
        loss_weights: Optional[List[float]] = None,
        metrics: List[keras.metrics.Metric] = None) -> None:
        """
        Initial compilation of the created model. Can later be recompiled usign self.recompile.
        Applies discriminative learning rates for the model if LR is passed as a tuple, and simple
        learning rate if lr is integer. Discriminative LR has a multiplier of * 0.3 for all base_model
        layers if self.lr is a tuple with a single float value. If self.lr is passed as a tuple with 2
        float values, the first half of base_model gets self.lr[0], the second half gets
        (self.lr[0] + self.lr[1] / 2), and the top layers get self.lr[1].

        # Examples:

            # Fixed learning rate
            model_1.compile("adam", lr=1e-4, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            # Discriminative learning rate
            model_1.compile("adam", lr=(1e-4,), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            # Discriminative learning rate with specified minimum value for bottom layers
            model_1.compile("adam", lr=(1e-6, 1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        """
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        self.loss_weights = loss_weights
        if isinstance(lr, float):
            self.model.compile(
                optimizer=optimizer(lr=lr),
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics,
            )
        elif isinstance(lr, tuple):
            multipliers = get_lr_multipliers(self, lr=lr, params=True)
            if len(lr) == 2:
                lr = lr[1]
            else:
                lr = lr[0]
            self.model.compile(
                optimizer=DLR_Adam(param_lrs=multipliers, learning_rate=lr),
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics,
            )


    def recompile(self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        lr: Optional[Union[float, Tuple[float], Tuple[float, float]]] = None,
        loss: Optional[keras.losses.Loss] = None,
        loss_weights: Optional[List[float]] = None,
        metrics: Optional[keras.metrics.Metric] = None,
        dropout: Optional[float] = None,
        load: Optional[str] = None) -> None:
        """
        Recompile model after initial compilation and training with a different optimizer,
        input shape, etc. Updates CNNLearner class variables with newly passed arguments
        (if provided) and compiles it. If some of the options are not provided, previous
        settings are used.

        Note: if weights for the "load" argument are not provided, the most recent weights in
        self.previous_weights are loaded.
        """
        if (input_shape is not None) | (dropout is not None):
            if dropout:
                self.dropout = dropout
            dropout = dropout or self.dropout
            output_layer = self.output_layer
            if not load: load = self.previous_weights
            self._model_creator(base_model=self.transfer_architecture,
                               name=self.name,
                               input_shape=input_shape,
                               output_layer=output_layer,
                               dropout=dropout,
                               load=load)
        if not optimizer: optimizer = self.optimizer
        if not lr: lr = self.lr
        if not loss: loss = self.loss
        if not loss_weights: loss_weights = self.loss_weights
        if not metrics: metrics = self.metrics

        self.compile(
            lr = lr,
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics
        )


    def freeze(self,
        n_layers: int,
        bn_skip: bool = True,
        ):
        """
        Freeze all but n_layers layers of the model (converts to layer.trainable = False).
        If bn_skip == True, all batchnorm layers are not frozen.
        """
        self.unfreeze()
        for layer in self.model.layers[:-n_layers]:
            layer.trainable = False
        if bn_skip:
            batch_norm_layers = ([idx for (idx, layer) in enumerate(self.model._layers) if
                      str(type(layer)) == "<class 'tensorflow.python.keras.layers.normalization.BatchNormalization'>"])
            for batch_norm_layer in batch_norm_layers:
                self.model.layers[batch_norm_layer].trainable = True

    def unfreeze(self):
        """
        Unfreeze all the layers.
        """
        for layer in self.model.layers[:-1]:
            layer.trainable = True

    def class_weight_calc(self,
        train_df: pd.DataFrame,
        strength: Optional[float] = 1.0):
        """
        Calculates class weights based on training dataframe class distribution.
        E.g. for binary classification, if 0 = 900 cases and 1 = 100 cases, class
        weights are {0: 1.0, 1: 9.0}.

        # Params:
        train_df: train dataset dataframe with 'id' and 'label' columns
        strength: optional multiplier for class weight strengths (e.g. 0.5 would keep
            largest class weight at 1, but cut weights of all other classes by 0.5, but
            not lower than 1.)
        """
        class_counts = train_df.groupby('label').count().reset_index()
        class_weights = max(class_counts.id)/class_counts.id
        class_weights = dict(class_weights)
        if strength != 1.0:
            class_weights = {k: max(1.0, strength*v) for (k,v) in class_weights.items() if v > 1}
        return class_weights

    def fit(self,
        epochs: int,
        name: str,
        lr: Optional[Union[float, Tuple[float], Tuple[float, float]]] = None,
        class_weights: Optional[dict] = None,
        verbose: Optional[int] = 1
        ):
        """
        Do N epochs of training. Saves best model weights to an h5 file. Uses early stopping
        and ReduceLROnPlateau callbacks. Stores the output weights file to self.previous_weights.
        """
        if lr != None:
            self.recompile(lr=lr)

        if class_weights:
            if isinstance(class_weights, bool):
                weights = self.class_weight_calc(self.data.train_dataframe)
            elif isinstance(class_weights, float):
                weights = self.class_weight_calc(self.data.train_dataframe, strength=class_weights)
        else:
            weights = None

        reduce_lr_patience = max(2, epochs // 4)
        early_stopping_patience = reduce_lr_patience * 2

        self.history = self.model.fit(
            x=self.data.train,
            steps_per_epoch=self.data.train.steps,
            validation_data=self.data.val,
            validation_steps=self.data.val.steps,
            epochs=epochs,
            class_weight=weights,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    f"{str(self.weights_path)}/{str(name)}.h5", save_best_only=True, save_weights_only=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.3, patience=reduce_lr_patience,
                ),
                keras.callbacks.EarlyStopping(
                    patience=early_stopping_patience, restore_best_weights=True
                ),
            ],
            verbose=verbose,
        )
        self.previous_weights = name
        self.load(name)

    def fit_one_cycle(self,
        lr: Tuple[float, float],
        name: str,
        momentum: Tuple[float, float] = (0.85, 0.95),
        epochs: int = 1,
        dlr: bool = False,
        verbose: int = 1,
        class_weights: Optional[dict] = None):
        """
        Fit N epochs of one cycle learning (inverted-v-shape learning rate, v-shape momentum batch-by-batch).
        Similar to self.fit(), except uses lr_range and momentum_range in a scheduler, that updates the lr
        after each batch. More information on One Cycle Policy: https://arxiv.org/pdf/1506.01186.pdf

        Learning rate and momentum fluctuations can be plotted after fitting by:
        ```python
            my_learner.fit_one_cycle((1e-5, 1e-3), name="test", dlr=True)
            my_learner.scheduler.plot_lr()
            my_learner.scheduler.plot_mtm()
        ```

        # Arguments:
        lr: learning rate slice from min to max
        name: save model's weights with this name
        momentum: momentum range, defaults to (0.85, 0.95). Note: momentum[0] has to be < momentum[1], not
            reversed.
        epochs: epochs to train.
        dlr: use discriminative learning rate optimizer or not.
        class_weights: class_weights for the loss function.
        """
        if dlr:
            self.recompile(lr=lr)
        else:
            if isinstance(lr, tuple):
                print("Note: Discriminative learning rates are disabled, but tuple lr",
                     f"is passed. Use dlr=True for DLR optimizer.")
        scheduler = TrainCycle(lr=lr, momentum=momentum, epochs=epochs,
                              batch_size=self.data.batch_size,
                              train_set_size=self.data.train_dataframe.shape[0])

        if class_weights:
            if isinstance(class_weights, bool):
                weights = self.class_weight_calc(self.data.train_dataframe)
            elif isinstance(class_weights, float):
                weights = self.class_weight_calc(self.data.train_dataframe, strength=class_weights)
        else:
            weights = None

        self.history = self.model.fit(
            x=self.data.train,
            steps_per_epoch=self.data.train.steps,
            validation_data=self.data.val,
            validation_steps=self.data.val.steps,
            epochs=epochs,
            class_weight=weights,
            callbacks=[
                scheduler,
                keras.callbacks.ModelCheckpoint(
                    f"{str(self.weights_path)}/{str(name)}.h5", save_best_only=True, save_weights_only=True
                ),
            ],
            verbose=verbose,
        )
        self.scheduler = scheduler
        self.previous_weights = name
        self.load(name)

    def find_lr(self,
        lr: Tuple[float, float] = (1e-7, 1e-1),
        epochs: Optional[int] = 1,
        class_weights: Union[bool, float] = None,
        verbose: int = 1):
        """
        Runs N epochs with different learning rates within given range for each batch and follows
        model's loss. The returned plot can be used to assess a good range of learning rates for
        training the model.

        # Examples:
        ```python
            my_learner.freeze(1)
            my_learner.find_lr()
            my_learner.plot_lr(skip_begin=10, skip_end=20)
        ```
        """
        if class_weights:
            weights = self.class_weight_calc(self.data.train_dataframe)
            if isinstance(class_weights, float):
                weights = {k: v*class_weights for (k,v) in weights if v > 1}
        else:
            weights = None
        lr_finder = LearningRateFinder(self.model)
        lr_finder.find(x=self.data.train,
                       start_lr=lr[0],
                       end_lr=lr[1],
                       epochs=1,
                       class_weights=weights,
                       steps_per_epoch=self.data.train.steps,
                       batch_size=self.data.batch_size,
                       verbose=verbose)
        self.lr_finder = lr_finder

    def plot_lr(self,
        lr_finder: LearningRateFinder = None,
        skip_begin=5,
        skip_end=20):
        """
        Plots lr_finder's results.
        """
        if lr_finder == None:
            lr_finder = self.lr_finder
        return lr_finder.plot_loss(skip_begin=skip_begin, skip_end=skip_end)

    def acc_report(self):
        """
        Returns classification report for validation data as a dictionary
        """
        preds = self.model.predict_generator(self.data.val, steps=self.data.val.steps)
        preds = preds.argmax(axis=1)
        y_true = self.data.val_dataframe.label
        preds = list(preds)[:len(y_true)]
        return classification_report(y_true=y_true, y_pred=preds, output_dict=True)
