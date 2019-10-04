from typing import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

class TrainCycle(Callback):
    """
    This class defines a callback that changes learning rate according to
    the one-cycle policy (https://arxiv.org/pdf/1803.09820.pdf). This implementation
    exponentially increases learning rate from min_lr to max_lr in the first 25%
    of training (stage 1), and gradually decreases it from max_lr to min_lr with
    a cosine decrease during the latter 75% of training (stage 2).

    This implementation is somewhat inspired by:
    https://docs.fast.ai/callbacks.one_cycle.html#OneCycleScheduler
    See also implementation of, where I borrowed a couple of ideas:
    https://github.com/titu1994/keras-one-cycle

    # Arguments
        lr: tuple of minimum and maximum learning rates for training
        momentum: tuple of minimum (!) and maximum (!) momentums for training (not the
            other way around!)
        epochs: number of epochs to train
        batch_size: batch_size used during training. Used to calculate # steps.
        train_set_size: number of training examples used to calculate # steps
    """

    def __init__(self,
                lr: Tuple[float, float] = (1e-5, 1e-2),
                momentum: Tuple[float, float] = (0.85, 0.95),
                epochs: int = 1,
                batch_size: int = 8,
                train_set_size: int = 1000,
                ):

        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self.iterations = 0
        self.history = {}

    def cosine_between(self, step_standardized, _min, _max):
        cosine = np.cos(step_standardized * np.pi) + 1
        cosine_between = (_min - _max)/2 * cosine + _max
        return cosine_between

    def polynomial_between(self, step_standardized, _min, _max):
        polynomial = (_min - _max) * (step_standardized)**2 + _max
        polynomial_between = -polynomial + (_min + _max)
        return polynomial_between

    def get_lr_mom_lists(self):
        # How many steps are there in total, in stage1 (25%) and stage2
        n_steps = int(round((self.epochs * self.train_set_size) / self.batch_size) + 4)
        n_steps_stage1 = round(n_steps/self.epochs/4)
        n_steps_stage2 = round(n_steps/self.epochs) - n_steps_stage1 + 4
        # Get a step list
        iteration_list = np.array(range(n_steps))
        # Stage 1: exponential increase to highest lr in epoch X
        stage_1_lrs = [self.polynomial_between(i, self.lr[0], self.lr[1]) for \
                       i in np.array(range(n_steps_stage1))/n_steps_stage1]
        # Stage 2: cosine decrease from highest to lowest lr until end of epoch X
        stage_2_lrs = [self.cosine_between(i, self.lr[1], self.lr[0]) for \
                       i in np.array(range(n_steps_stage2))/n_steps_stage2]
        concat_lrs = stage_1_lrs + stage_2_lrs

        # Stages are reversed for momentum (not: self.momentum[1] preceedes self.momentum[0]
        # in polynomial_between of stage 1 (decrease) and vice versa in stage 2 (increase))
        stage_1_moms = [self.polynomial_between(i, self.momentum[1], self.momentum[0]) for \
                       i in np.array(range(n_steps_stage1))/n_steps_stage1]
        stage_2_moms = [self.cosine_between(i, self.momentum[0], self.momentum[1]) for \
                       i in np.array(range(n_steps_stage2))/n_steps_stage2]
        concat_moms = stage_1_moms + stage_2_moms

        # Get lr and mom list for each step. Loops from stage 1 to stage 2 in each epoch
        lr_list = {v: k for (v,k) in zip(iteration_list, concat_lrs*self.epochs)}
        mom_list = {v: k for (v,k) in zip(iteration_list, concat_moms*self.epochs)}
        return lr_list, mom_list


    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.lr_list, self.mom_list = self.get_lr_mom_lists()
        K.set_value(self.model.optimizer.lr, self.lr[0])
        K.set_value(self.model.optimizer.beta_1, self.momentum[1])

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.beta_1))
        self.history.setdefault('iterations', []).append(self.iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.lr_list[self.iterations])
        K.set_value(self.model.optimizer.beta_1, self.mom_list[self.iterations])

    def plot_lr(self):
        plt.xlabel('Training step')
        plt.ylabel('LR')
        plt.title("Learning Rate change during training")
        plt.plot(self.history['iterations'], self.history['lr'])

    def plot_mtm(self):
        plt.xlabel('Training step')
        plt.ylabel('Momentum')
        plt.title("Momentum change during training")
        plt.plot(self.history['iterations'], self.history['momentum'])
