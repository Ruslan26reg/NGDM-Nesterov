import tensorflow as tf 
import numpy as np
import os
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import keras_export

from scipy import special

def parameter(n):
    alpha = []
    for i in range(n):
    	alpha.append(1.4 - 0.005**(i+1)) #0.4 - 0.005**(i+1) Cifar10 
    return alpha

def parameter1(n):
    alpha = []
    for i in range(n):
        alpha.append(1.4 - 0.005**(i+1)) # 0.8 - 0.05**(i+1) 4.8 + 0.005**(i+1)
    beta = []
    for i in range(n):
        beta.append(0.9 - 0.005**(i+1)) # 0.4 - 0.05**(i+1)
    return alpha, beta

def summation(n):
    summ = 0
    alpha = parameter(n)
    for i in range(n):
        summ += alpha[i]
    return summ

def Fisher(n):
    FIM = []
    alpha = parameter(n)
    for i in range(n):
        Row = []
        for j in range(n):
            if i == j: 
                Row.append(special.polygamma(1, alpha[i]) - special.polygamma(1, summation(n)))
            else:
                #Row.append(0)
                Row.append(- special.polygamma(1, summation(n)))
        FIM.append(Row)
    FIM = numpy.array(FIM)
    FIM = numpy.linalg.inv(FIM)
    return FIM

def Fisher2(n):
    FIM = []
    for i in range(n):
        Zero = []
        for j in range(n):
            Zero.append(0)
        FIM.append(Zero)
    alpha, beta = parameter1(n)
    for i in range(0,n,2):
        FIM[i][i] = special.polygamma(1, alpha[i]) - special.polygamma(1, alpha[i]+beta[i])
        FIM[i+1][i] = - special.polygamma(1, alpha[i]+beta[i])
        FIM[i][i+1] = - special.polygamma(1, alpha[i]+beta[i])
        FIM[i+1][i+1] = special.polygamma(1, beta[i]) - special.polygamma(1, alpha[i]+beta[i])
    FIM = numpy.array(FIM)
    FIM = numpy.linalg.inv(FIM)
    return FIM

class NGD_Dirichlet(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        name="NGD_Dirichlet",
        **kwargs):
        super(NGD_Dirichlet, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._momentum = False
        if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError(f"`momentum` must be between [0, 1]. Received: "
                f"momentum={momentum} (of type {type(momentum)}).")
        self._set_hyper("momentum", momentum)
        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(NGD_Dirichlet, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype))

        if self._momentum:
          momentum_var = self.get_slot(var, "momentum")
          return tf.raw_ops.ResourceApplyKerasMomentum(
              var=var.handle,
              accum=momentum_var.handle,
              lr=coefficients["lr_t"],
              grad=grad,
              momentum=coefficients["momentum"],
              use_locking=self._use_locking,
              use_nesterov=self.nesterov)
        else:
          return tf.raw_ops.ResourceApplyGradientDescent(
              var=var.handle,
              alpha=coefficients["lr_t"],
              delta=grad,
              use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs):
        if self._momentum:
          return super(NGD_Dirichlet, self)._resource_apply_sparse_duplicate_indices(
              grad, var, indices, **kwargs)
        else:
          var_device, var_dtype = var.device, var.dtype.base_dtype
          coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype))

        return tf.raw_ops.ResourceScatterAdd(
            resource=var.handle,
            indices=indices,
            updates=np.dot(Fisher(len(grad)),grad.transpose()) * coefficients["lr_t"])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype))

        momentum_var = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)

    def get_config(self):
        config = super(NGD_Dirichlet, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._initial_decay,
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config

class NGD_GeneralizedDirichlet(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        name="NGD_GeneralizedDirichlet",
        **kwargs):
        super(NGD_GeneralizedDirichlet, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._momentum = False
        if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError(f"`momentum` must be between [0, 1]. Received: "
                f"momentum={momentum} (of type {type(momentum)}).")
        self._set_hyper("momentum", momentum)
        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(NGD_GeneralizedDirichlet, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype))

        if self._momentum:
          momentum_var = self.get_slot(var, "momentum")
          return tf.raw_ops.ResourceApplyKerasMomentum(
              var=var.handle,
              accum=momentum_var.handle,
              lr=coefficients["lr_t"],
              grad=grad,
              momentum=coefficients["momentum"],
              use_locking=self._use_locking,
              use_nesterov=self.nesterov)
        else:
          return tf.raw_ops.ResourceApplyGradientDescent(
              var=var.handle,
              alpha=coefficients["lr_t"],
              delta=grad,
              use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs):
        if self._momentum:
          return super(NGD_GeneralizedDirichlet, self)._resource_apply_sparse_duplicate_indices(
              grad, var, indices, **kwargs)
        else:
          var_device, var_dtype = var.device, var.dtype.base_dtype
          coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype))

        return tf.raw_ops.ResourceScatterAdd(
            resource=var.handle,
            indices=indices,
            updates=np.dot(Fisher1(len(grad)/2),-grad.transpose()) * coefficients["lr_t"])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype))

        momentum_var = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)

    def get_config(self):
        config = super(NGD_GeneralizedDirichlet, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._initial_decay,
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config
