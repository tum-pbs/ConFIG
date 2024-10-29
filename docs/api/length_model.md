The `length_model` module contains classes for rescaling the magnitude of the final gradient vector.
The `ProjectionLength` class is the default length model for the ConFIG algorithm. You can create a custom length model by inheriting from the `LengthModel` class.

## Length Model
::: conflictfree.length_model.ProjectionLength
::: conflictfree.length_model.TrackMinimum
::: conflictfree.length_model.TrackMaximum
::: conflictfree.length_model.TrackHarmonicAverage
::: conflictfree.length_model.TrackArithmeticAverage
::: conflictfree.length_model.TrackGeometricAverage
::: conflictfree.length_model.TrackSpecific

## Base Class of Length Model
::: conflictfree.length_model.LengthModel