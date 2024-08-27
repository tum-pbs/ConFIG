The `loss_recorder` module contains classes for recording the loss values during the optimization process.
It is used in the momentum version of the ConFIG algorithm to record the loss values. Not every loss is calculated in a single iteration with the momentum version of the ConFIG algorithm. However, sometimes we need to know the information of all the loss values, e.g., logging and calculating length/weight model. You can create a custom loss recorder by inheriting from the `LossRecorder` class.

## Loss Recorder
::: conflictfree.loss_recorder.LatestLossRecorder
::: conflictfree.loss_recorder.MomentumLossRecorder

## Base Class of Loss Recorder
::: conflictfree.loss_recorder.LossRecorder