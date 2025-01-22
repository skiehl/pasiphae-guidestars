# pasiphae-field-grid

## About
The [Pasiphae project](http://pasiphae.science/) aims to map, with
unprecedented accuracy, the polarization of millions of stars at areas of the
sky away from the Galactic plane, in both the Northern and the Southern
hemispheres. New instruments, calibration and analysis methods, and dedicated
software are currently under development. The survey will be performed with two
dedicated instruments WALOP-N and WALOP-S, each mounted at a telescope in the
Northern or Southern hemisphere.

The pasiphae-guidestars package provides classes to select guide stars for
observations. The WALOP-N and WALOP-S instrument both use movable guide
cameras. The range of motion differs between both instruments. This package
implements classes for both instruments. The guide star selection for a
specific sky position covers three steps:

1. Identify potential guide stars in the region accessible to the guide camera.
   Select the N brightest ones as guide stars, but avoid stars where the guide
   camera would include very bright objects.
2. Determine the position of the guide camera for each selected guide star.
3. Determine the guide camera exposure time for each guide star.

This package integrates with the
[pasiphae-field-grid](https://github.com/skiehl/pasiphae-field-grid) package.
Selection of guide stars can be done for a current pointing position, for
one-the-fly selection, or can be applied to the full Pasiphae field grid as
defined in
[pasiphae-field-grid](https://github.com/skiehl/pasiphae-field-grid)
to pre-select guide stars for all Pasiphae science fields.

## Modules

* `guidestars.py`: Classes to select guide stars.
    * `GuideStarSelector` class: Parent class that provides methods shared by
      the following two classes and that defines some abstract methods.
    * `GuideStarWalopS` class: Guide star selection for the Southern instrument
      WALOP-S.
    * `GuideStarWalopN` class: Guide star selection for the Northern instrument
      WALOP-N.
* `utilities.py`: Provides various utility function shared by the
  `guidestars.py` and the `fieldgrid.py` modules. This module is being
  developed in
  [pasiphae-field-grid](https://github.com/skiehl/pasiphae-field-grid).
* `fieldgrid.py`: Module providing the Pasiphae field grid implementations.
  This module is being developed in
  [pasiphae-field-grid](https://github.com/skiehl/pasiphae-field-grid).
  It is integrated here only for demonstrating the use of the `guidestars.py`
  module.

## Notebooks

* `Develop_Guidestar_S.ipynb`: Development of the WALOP-S guide star selection
  code.
* `Develop_Guidestar_N.ipynb`: Development of the WALOP-N guide star selection
  code.
* `Test_Guidestars.ipynb`: Application of the guide star selection to the
  Northern and Southern Pasiphae field grids. Tests which magnitude limit is
  required to find at least one guide star for each Pasiphae field.
* `Test_runtime.ipynb`: Runtime tests. Tests if it is feasible to run the guide
  star selection on-the-fly.

## Auxiliary files
* `*.json`: Contain parameters for setting up the guide star selection and the
  field grids.
* `info/`: Contains figures that illustrate the region accesible to the guide
  camera .
* `gaia/`: Contains the info how to obtain the VOTables that contain the Gaia
  star coordinates and magnitude, from which we select the guide stars.
