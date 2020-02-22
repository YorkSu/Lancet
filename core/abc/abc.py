# -*- coding: utf-8 -*-
"""Abstract Class

  File: 
    /Lancet/core/abc/abc

  Description: 
    Abstract Classes
"""


class ClassA(object):
  """ClassA

    Description:
      Abstract Classes - ClassA class
      Including `get` and `set` method
  """
  def get(self, name, default=None):
    if name not in self.__dict__:
      return default
    return self.__dict__[name]

  def set(self, name, value):
    self.__dict__[name] = value


class Keras_Network(object):
  """Keras Network

    Description:
      Abstract Classes - Keras Network API class
      Including usual keras.Model.functions
      Need to implement `__init__()` and define this Attributes:
        model: tf.keras.models.Model
  """
  def compile(
      self,
      optimizer,
      loss=None,
      metrics=None,
      loss_weights=None,
      sample_weight_mode=None,
      weighted_metrics=None,
      target_tensors=None,
      distribute=None,
      **kwargs):
    """Get compile function"""
    self.model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights,
        sample_weight_mode=sample_weight_mode,
        weighted_metrics=weighted_metrics,
        target_tensors=target_tensors,
        distribute=distribute,
        **kwargs)

  def fit(
      self,
      x=None,
      y=None,
      batch_size=None,
      epochs=1,
      verbose=1,
      callbacks=None,
      validation_split=0.,
      validation_data=None,
      shuffle=True,
      class_weight=None,
      sample_weight=None,
      initial_epoch=0,
      steps_per_epoch=None,
      validation_steps=None,
      validation_freq=1,
      max_queue_size=10,
      workers=1,
      use_multiprocessing=False,
      **kwargs):
    """Get fit function"""
    return self.model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=validation_split,
        validation_data=validation_data,
        shuffle=shuffle,
        class_weight=class_weight,
        sample_weight=sample_weight,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        **kwargs)

  def evaluate(
      self,
      x=None,
      y=None,
      batch_size=None,
      verbose=1,
      sample_weight=None,
      steps=None,
      callbacks=None,
      max_queue_size=10,
      workers=1,
      use_multiprocessing=False):
    """Get evaluate function"""
    return self.model.evaluate(
        x=x,
        y=y,
        batch_size=batch_size,
        verbose=verbose,
        sample_weight=sample_weight,
        steps=steps,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def predict(
      self,
      x,
      batch_size=None,
      verbose=1,
      steps=None,
      max_queue_size=10,
      workers=1,
      use_multiprocessing=False):
    """Get predict function"""
    return self.model.predict(
        x,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def fit_generator(
      self,
      generator,
      steps_per_epoch=None,
      epochs=1,
      verbose=1,
      callbacks=None,
      validation_data=None,
      validation_steps=None,
      class_weight=None,
      max_queue_size=10,
      workers=1,
      use_multiprocessing=False,
      shuffle=True,
      initial_epoch=0):
    """Get fit_generator function"""
    return self.model.fit_generator(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch)

  def evaluate_generator(
    self,
    generator,
    steps=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    verbose=1):
    """Get evaluate_generator function"""
    return self.model.evaluate_generator(
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose,)

  def save(
      self,
      filepath,
      overwrite=True,
      include_optimizer=True):
    """Get save function"""
    self.model.save(
        filepath,
        overwrite=overwrite,
        include_optimizer=include_optimizer)

  def summary(
      self,
      line_length=100,
      positions=None,
      print_fn=None):
    """Get summary function"""
    self.model.summary(
        line_length=line_length,
        positions=positions,
        print_fn=print_fn)

