from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import *
from pathlib import *

from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export

def get_lr_multipliers(
    model: Any,
    lr: Union[Tuple[float], Tuple[float, float], Tuple[float, ...]],
    params: bool = False) -> dict:
    """
    This funcion takes an ImageLearner and a learning rate, and creates a dictionary, where each
    layer (or each layer's parameters if params == True) is mapped to an individual
    learning rate multiplier. This dictionary is then passed to DLR optimizer, which applies these
    layer/parameter learning rate multipliers during optimization (a.k.a. discriminative/differential
    learning rates).

    # Arguments:
        model: ImageLearner, where ImageLearner.model is the final model, and ImageLearner.base_model
            is the transfer learning type used in ImageLearner.model (e.g. keras.applications.Xception)
        lr: learning rates to be applied to layers/parameters. Supports a tuple of length 1
            for applying learning rates
    """

    # Dictionary that contains parts of layer names that can be safely used as splits
    # for defining layer groups
    architecture_slices = {
        "inception_v3": "mixed",
        "resnet50": "add",
        "efficientnet-b0": "expand_conv",
        "efficientnet-b0": "expand_conv",
        "xception": "add"
        }

    # FastAI kind of learning rate splits:
    # 1. If input lr is a tuple of length 1, apply no multiplier for top layers and
    # 0.3 multiplier for all base layer (ImageLearner.base_model)
    if len(lr) == 1:
        all_layers = [i.name for i in model.model.layers]
        top_layer = model.base_model.layers[-1].name
        idx_top = all_layers.index(top_layer)
        split_1 = {i: 0.3 for i in all_layers[1:idx_top]}
        split_2 = {i: 1 for i in all_layers[idx_top:]}
        split_1.update(split_2)
        lr_slices = split_1
        if params:
            return layer_to_param_dict(lr_slices, model)
        else:
            return lr_slices
    # 2. If lr's passed as double tuple, the top layers get LR * 1, the base layers
    # get divided into two groups: bottom group (first layers) gets a multiplier lr[0]/lr[1],
    # middle layers get the average of multipliers of top layers and bottom layers.
    elif len(lr) == 2:
        split_candidates = [x.name for x in model.model.layers
                            if architecture_slices[model.base_model.name] in x.name]
        split_layer = split_candidates[round(len(split_candidates)/2)]

        all_layers = [x.name for x in model.model.layers]
        top_layer = model.base_model.layers[-1].name

        idx_split = all_layers.index(split_layer)
        idx_top = all_layers.index(top_layer)

        split_1 = {i: lr[0]/lr[1] for i in all_layers[1:idx_split]}
        split_2 = {i: (1 + (lr[0] + lr[1])) / 2 for i in all_layers[idx_split:idx_top]}
        split_3 = {i: 1 for i in all_layers[idx_top:]}

        split_1.update(split_2)
        split_1.update(split_3)
        lr_slices = split_1
        if params:
            return layer_to_param_dict(lr_slices, model)
        else:
            return lr_slices


def layer_to_param_dict(
    lr_slices: dict,
    model: Any) -> dict:
    """
    This function takes a dictionary of model layers as keys and learning rate multipliers as values,
    the model which it was based on, and converts it to a dictionary of parameters and learning_rates,
    where each parameter's LR corresponds to its layer's LR in the input dictionary.
    """
    layers = {i.name: i for i in model.model.layers}
    parameters = {i.name: i.variables for i in layers.values() if len(i.variables) > 0}
    parameter_lr_rates = {}
    for layer, parameters in parameters.items():
        for param in parameters:
            parameter_lr_rates[param.name] = lr_slices[layer]
    return parameter_lr_rates


class DLR_Adam(keras.optimizers.Optimizer):
    """
    Tensorflow Keras Adam optimizer, edited to support discriminative learning rates. Takes an additional
    input param_lrs, which is a dictionary that contains model's parameters and learning_rate multipliers.
    Multipliers in param_lrs are applied to learning_rates during _resource_apply_dense or
    _resource_apply_sparse, i.e. right before parameters are updated.
    """
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        name='Adam',
        param_lrs=None,
        **kwargs
        ):

        super(DLR_Adam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr',
                        learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad
        self.param_lrs = param_lrs

        initiation_dict = {k: 1 for (k,v) in self.param_lrs.items()}
        self.initiation_dict = initiation_dict

        print("NOTE: Discriminative LR Adam is used.")

    def _create_slots(self, var_list):

    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.

        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _prepare_local(
        self,
        var_device,
        var_dtype,
        apply_state,
        ):
        super(DLR_Adam, self)._prepare_local(var_device, var_dtype,
                apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1',
                var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2',
                var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[(var_device, var_dtype)]['lr_t'] \
            * (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
        apply_state[(var_device, var_dtype)].update(dict(
            lr=lr,
            epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t,
            ))

    def set_weights(self, weights):
        params = self.weights

    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.

        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(DLR_Adam, self).set_weights(weights)

    def _resource_apply_dense(
        self,
        grad,
        var,
        apply_state=None,
        ):
        (var_device, var_dtype) = (var.device, var.dtype.base_dtype)
        coefficients = (apply_state or {}).get((var_device, var_dtype)) \
            or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        if self.initiation_dict[var.name] == 1:
            coefficients['lr_t'] = coefficients['lr_t'] * self.param_lrs[var.name]
            self.initiation_dict[var.name] = 0
        if not self.amsgrad:
            return training_ops.resource_apply_adam(
                var.handle,
                m.handle,
                v.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                coefficients['lr_t'],
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking,
                )
        else:
            vhat = self.get_slot(var, 'vhat')
            return training_ops.resource_apply_adam_with_amsgrad(
                var.handle,
                m.handle,
                v.handle,
                vhat.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                coefficients['lr_t'],
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking,
                )

    def _resource_apply_sparse(
        self,
        grad,
        var,
        indices,
        apply_state=None,
        ):
        (var_device, var_dtype) = (var.device, var.dtype.base_dtype)
        coefficients = (apply_state or {}).get((var_device, var_dtype)) \
            or self._fallback_apply_state(var_device, var_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t

        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices,
                    m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)

        v = self.get_slot(var, 'v')
        v_scaled_g_values = grad * grad \
            * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices,
                    v_scaled_g_values)

        if not self.amsgrad:
            v_sqrt = math_ops.sqrt(v_t)
            if self.initiation_dict[var.name] == 1:
                var_update = state_ops.assign_sub(var,
                        coefficients['lr'] * self.param_lrs[var.name] \
                        * m_t / (v_sqrt + coefficients['epsilon']),
                        use_locking=self._use_locking)
                self.initiation_dict[var.name] = 0
            else:
                var_update = state_ops.assign_sub(var,
                        coefficients['lr'] \
                        * m_t / (v_sqrt + coefficients['epsilon']),
                        use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(v_hat, v_hat_t,
                        use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            if self.initiation.dict[var.name] == 1:
                var_update = state_ops.assign_sub(var,
                        coefficients['lr'] * self.param_lrs[var.name] \
                        * m_t / (v_sqrt + coefficients['epsilon']),
                        use_locking=self._use_locking)
                self.initiation_dict[var.name] = 0
            else:
                var_update = state_ops.assign_sub(var,
                        coefficients['lr'] \
                        * m_t / (v_hat_sqrt + coefficients['epsilon']),
                        use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t,
                    v_hat_t])

    def get_config(self):
        config = super(DLR_Adam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'
                    ),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            })
        return config
