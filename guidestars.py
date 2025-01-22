#!/usr/bin/env python3
"""Sky fields for the Pasiphae survey.
"""

from abc import  ABCMeta, abstractmethod
from astropy.coordinates import Angle, SkyCoord
from astropy.io.votable import parse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from utilities import \
    cart_to_sphere, close_to_edge, inside_polygon, rotate_frame, rot_tilt, \
    sphere_to_cart

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD3"
__version__ = "0.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# FUNCTIONS
#==============================================================================

def load_gaia(votable_files, dir_in='', dec_lolim=None, dec_uplim=None):
    """Read Gaia stars' coordinates and magnitudes from VOTables.

    Parameters
    ----------
    votable_files : str or list of str
        The VOTable file name(s).
    dir_in : str, optional
        Directory where the VOTable files are located. The default is ''.
    dec_lolim : float, optional
        Lower declination limit in degrees. The default is None.
    dec_uplim : float, optional
        Upper declination limit in degrees. The default is None.

    Returns
    -------
    gaia_ra : numpy.ndarray
        Gaia star right ascensions in radians.
    gaia_dec : numpy.ndarray
        Gaia star declinations in radians.
    gaia_mag : numpy.ndarray
        Gaia star right ascensions in radians.
    """

    if isinstance(votable_files, str):
        votable_files = [votable_files]

    gaia_ra = []
    gaia_dec = []
    gaia_mag = []

    for i, filename in enumerate(votable_files, start=1):
        print(f'Read VOTable {i}/{len(votable_files)}..')
        votable = parse(os.path.join(dir_in, filename))
        table = votable.get_first_table()

        if dec_lolim:
            sel0 = table.array['dec'] >= dec_lolim
        else:
            sel0 = np.ones(table.array['ra'].shape[0], dtype=bool)

        if dec_uplim:
            sel1 = table.array['dec'] <= dec_uplim
        else:
            sel1 = np.ones(table.array['ra'].shape[0], dtype=bool)

        sel = np.logical_and(sel0, sel1)

        gaia_ra.append(table.array['ra'].data[sel])
        gaia_dec.append(table.array['dec'].data[sel])
        gaia_mag.append(table.array['phot_g_mean_mag'].data[sel])

    gaia_ra = np.concatenate(gaia_ra)
    gaia_dec = np.concatenate(gaia_dec)
    gaia_mag = np.concatenate(gaia_mag)

    return gaia_ra, gaia_dec, gaia_mag

#==============================================================================
# CLASSES
#==============================================================================

class GuideStarSelector(metaclass=ABCMeta):
    """A class to select guide stars."""

    #--------------------------------------------------------------------------
    def __init__(self):
        """Create GuideStarSelector instance.
        """

        self.params = None
        self.avoid_coord = None
        self.stars_coord = None
        self.stars_mag = None
        self.fieldgrid = None
        self.guidestars = None
        self.n_guidestars = None
        self.n_fields = None

    #--------------------------------------------------------------------------
    @abstractmethod
    def _locate(self, field_ra, field_dec, return_coord=False):
        """Select guide stars for one field.

        Parameters
        ----------
        field_ra : float
            Field center right ascension in radians.
        field_dec : float
            Field center declination in radians.
        return_coord : bool, optional
            If True, coordinates of the selected guide stars, of stars too
            close to the edge of the guider area, and stars within the
            instrument area are returned in addition to the indices of the
            selected stars. Otherwise, only the indices of the selected stars
            are returned.

        Returns
        -------
        i_guide : numpy.ndarray
            Indices of the stars in the guide area.
        coord_rot_guidearea : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area. Only returned if
            `return_coord=True`.
        coord_rot_edge : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area, but too close to the
            edge. Only returned if `return_coord=True`.
        coord_rot_circle : astropy.coordinates.SkyCoord
            Coordinates of the stars in the instrument area. Only returned if
            `return_coord=True`.

        Notes
        -----
        This method is called by `_select()`.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def _guider_position(self, coord_rot_guidearea):
        """Calculate the guide camera position for a selection of guide stars.

        Parameters
        ----------
        coord_rot_guidearea : astropy.coordinates.SkyCoord
            Coordinates of selected guide stars rotated into a reference frame
            that has its origin at the corresponding field center.

        Returns
        -------
        pos_x : numpy.array
            x-position of the guide camera in the units defined by the input
            parameter `scale_xy`.
        pos_y : numpy.array
            y-position of the guide camera in the units defined by the input
            parameter `scale_xy`.

        Notes
        -----
        This method is called by `_select()`.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def _select(self, field_ra, field_dec, return_coord=False, n_max=None):
        """Select guide stars from the candidate star list for a field.

        Parameters
        ----------
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`. Alternatively, provide `fieldgrid`. The default is
            None.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`. Alternatively, provide `fieldgrid`. The default is
            None.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        n_max : int, optional
            Maximum number of guide stars to select for each field. If None,
            all stars in the guider area are saved. The default is None.

        Returns
        -------
        list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. The guide stars are sorted by increasing magnitude.

        Notes
        -----
        This method is called by `select()` or by `_iter_grid()`.
        This method calls `_locate()` and `_guider_position()`.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def _clear_from_stars_to_avoid(self):
        """Checks if the guide camera contains too bright stars.

        Parameters
        ----------
        Depend on the implementation.

        Returns
        -------
        clear : bool
            True, if the guide camera region does not contain too bright stars.
            False, otherwise.
        """

        pass

    #--------------------------------------------------------------------------
    def _count_guidestars(self, warn=True):
        """Counts the number of fields and number of guide stars selected for
        each field.

        Parameters
        ----------
        warn : bool, optional
            If True, print out a warning when at least one field does not have
            the required minimum number of guide stars. The default is True.

        Returns
        -------
        None

        Notes
        -----
        This method is called by `select()` and by `_update_guidestars()`.
        """

        self.n_fields = len(self.guidestars)
        self.n_guidestars = np.array([
                len(gs['guidestars']) for gs in self.guidestars])

        if warn and np.any(self.n_guidestars < self.n_min):
            print('WARNING: Not all fields have the required minimum number '
                  f'of guide stars ({self.n_min}) available.\n')

    #--------------------------------------------------------------------------
    def _update_guidestars(self):
        """Update the selection of guide stars for fields that do not have the
        required number of guide stars.

        Returns
        -------
        None

        Notes
        -----
        This method is called by `set_stars()`.
        """

        n_missing = self.n_min - self.n_guidestars
        i_sel = np.nonzero(n_missing > 0)[0]
        n_missing = n_missing[i_sel]
        n = i_sel.shape[0]
        print(f'\n{n} field do not have enough guide stars. Search for more..')

        # iterate though fields with insufficient number of guide stars:
        for i, (j, n_max) in enumerate(zip(i_sel, n_missing)):
            print(f'\rField {i} of {n} ({i/n*100:.1f}%)..', end='')

            # select guide stars:
            field_ra = self.guidestars[j]['field_center_ra']
            field_dec = self.guidestars[j]['field_center_dec']
            guidestars = self._select(field_ra, field_dec, n_max=n_max)

            # append guide stars:
            self.guidestars[j]['guidestars'] += guidestars

        print('\r  done.                             \n')
        self._count_guidestars()
        self.check_results()

    #--------------------------------------------------------------------------
    def _iter_grid(self, fieldgrid):
        """

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid.

        Returns
        -------
        ist of dict
            One list entry for each field. Each dict contains the field center
            coordinates and a list of selected guide stars. This list contains
            a dict for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y.

        Notes
        -----
        This method is called by `select()`.
        This method calls `_select()`.
        """

        field_ras, field_decs = fieldgrid.get_center_coords()
        n = len(fieldgrid)
        print('Iterate through field grid..')
        guidestars = []

        for i, (field_ra, field_dec) in enumerate(zip(field_ras, field_decs)):
            print(f'\rField {i} of {n} ({i/n*100:.1f}%)..', end='')

            guidestars_for_field = self._select(field_ra, field_dec)
            guidestars.append({
                    'field_center_ra': field_ra, 'field_center_dec': field_dec,
                    'guidestars': guidestars_for_field})

        print('\r  done.                             \n')

        return guidestars

    #--------------------------------------------------------------------------
    def _set_stars(self, ra, dec, mag):
        """Store the stars' coordinates and magnitudes, from which guide stars
        will be selected.

        Parameters
        ----------
        ra : array-like
            Right ascensions of the stars in radians.
        dec : array-like
            Declinations of the stars in radians.
        mag : array-like
            Magnitudes of the stars.

        Returns
        -------
        None

        Notes
        -----
        This method is called by `set_stars()`.
        """

        self.stars_coord = SkyCoord(ra, dec, unit='rad')
        self.stars_mag = np.asarray(mag)
        self.stars_mag_min = self.stars_mag.min()
        self.stars_mag_max = self.stars_mag.max()

    #--------------------------------------------------------------------------
    def set_stars(self, ra, dec, mag):
        """Add the stars' coordinates and magnitudes, from which guide stars
        will be selected.

        Parameters
        ----------
        ra : array-like
            Right ascensions of the stars in radians.
        dec : array-like
            Declinations of the stars in radians.
        mag : array-like
            Magnitudes of the stars.

        Returns
        -------
        None

        Notes
        -----
        This method calls `_set_stars()` and, if the list of stars is updated,
        `_update_guidestars()`.
        """

        # set stars for the first time:
        if self.stars_mag is None:
            self._set_stars(ra, dec, mag)
            print(f'{len(ra)} candidate stars added.')
            print(f'Magnitude range: {self.stars_mag_min:.1f} - '
                  f'{self.stars_mag_max:.1f}')

        # update stars:
        else:
            mag = np.asarray(mag)

            # check that new stars have higher magnitudes than previous stars:
            if mag.min() < self.stars_mag_max:
                raise ValueError(
                        "New stars must have higher magnitudes than "
                        f"previously set stars: > {self.stars_mag_max}")

            # update variables:
            print('Overwriting previous stars..')
            self._set_stars(ra, dec, mag)
            print(f'{len(ra)} candidate stars added.')
            print(f'Magnitude range: {self.stars_mag_min:.1f} - '
                  f'{self.stars_mag_max:.1f}')
            self._update_guidestars()

    #--------------------------------------------------------------------------
    def set_stars_to_avoid(self, ra, dec):
        """Store the coordinates of bright stars that should be avoided in the
        guide area.

        Parameters
        ----------
        ra : array-like
            Right ascensions of the stars in radians.
        dec : array-like
            Declinations of the stars in radians.

        Returns
        -------
        None
        """

        self.avoid_coord = SkyCoord(ra, dec, unit='rad')

        print(f'{len(ra)} coordinates of bright stars set that will be '
              'avoided in the guide camera.')

    #--------------------------------------------------------------------------
    @abstractmethod
    def set_params(self):
        """Set science field and guide area parameters.

        Parameters
        ----------
        Depend on the specific instrument.

        Raises
        ------
        ValueError
            Raised if the grid parameters are without their allowed bounds.

        Returns
        -------
        None
        """

        # check inputs:
        # custom code goes here

        # store parameters:
        self.params = {}
        # custom code goes here, all parameters need to be stored in this dict

    #--------------------------------------------------------------------------
    def save_params(self, filename):
        """Save science field and guide area in JSON file.

        Parameters
        ----------
        filename : str
            Filename for saving the parameters.

        Returns
        -------
        None
        """

        with open(filename, mode='w') as f:
            json.dump(self.params, f, indent=4)

        print('Guide parameters saved in:', filename)

    #--------------------------------------------------------------------------
    def load_params(self, filename):
        """Load science field and guide area parameters from JSON file.

        Parameters
        ----------
        filename : str
            Filename that stores the parameters.

        Returns
        -------
        None
        """

        with open(filename, mode='r') as f:
            params = json.load(f)
            self.set_params(**params)

        print(f'Parameters loaded from {filename}.')

    #--------------------------------------------------------------------------
    @abstractmethod
    def select(
            self, fieldgrid=None, field_ra=None, field_dec=None,
            mag_to_exp=None, return_coord=False, verbose=1):
        """Select guide stars from the candidate star list for either each
        field in a field grid or for a specific field's coordinates.

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid. If not given, then provide `field_ra` and `field_dec`. The
            default is None.
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`. Alternatively, provide `fieldgrid`. The default is
            None.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`. Alternatively, provide `fieldgrid`. The default is
            None.
        mag_to_exp : callable, optional
            A function that takes magnitudes as argument and converts them to
            exposure times (in seconds). If None is given, no exposure times
            are calculated. The default is None.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        verbose : int, optional
            Controls the level of detail of information printed. The default is
            1.

        Raises
        ------
        ValueError
            Raised, if `mag_to_exp` is neither None nor a function.
        ValueError
            Raised, if neither `field_ra` and `field_dec` or `fieldgrid` are
            given.

        Returns
        -------
        If guide stars are selected for a specific field given through
        `field_ra` and `field_dec` the following data   structure is returned:

        guidestars : list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. Exposure times (exp) in seconds are included, if a magnitude
            to exposure time conversion function is given to the called
            `select()` method. The guide stars are sorted by increasing
            magnitude.

        If guide stars are selected for all fields in a grid given through
        `fieldgrid` the following data structure is returned:

        guidestars : list of dict
            Each list entry is a dictionary corresponding to one field in the
            grid. The dictionary contains the field's right ascension and
            declination and a list of associated guide stars. This list has the
            same data structure as explained above.

        If return_coord is True, the following items are returned as well:

        Note: These returned items depend on the instrument. The docstring
        needs to be changed accordingly. The rest of the docstring holds for
        all child class implementations.

        Notes
        -----
        This method calls `_select()` for a single field or `_iter_grid()`,
        `_count_guidestars()`, and `check_results()` for a field grid.
        """

        # check input:
        if mag_to_exp is not None and not callable(mag_to_exp):
            raise ValueError("`mag_to_exp` must be None or a function.")

        self.mag_to_exp = mag_to_exp

        # select guide stars for single field:
        if field_ra is not None and field_dec is not None:
            return self._select(field_ra, field_dec, return_coord=return_coord)

        # select guide stars for entire field grid:
        elif fieldgrid is not None:
            guidestars = self._iter_grid(fieldgrid)
            self.guidestars = guidestars
            self._count_guidestars()
            self.check_results(verbose=verbose)

        else:
            raise ValueError(
                    "Either `field_ra` and `field_dec` must be given or "
                    "`fieldgrid`.")

        return guidestars

    #--------------------------------------------------------------------------
    def check_results(self, verbose=1):
        """Print out various statistics about the selected guide stars.

        Parameters
        ----------
        verbose : int, optional
            Controls the level of detail of information printed. The default is
            1.

        Returns
        -------
        None

        Notes
        -----
        This method is called by `select()` and by `_update_guidestars()`.
        """

        if verbose > 0:
            n_fields = self.n_fields
            n_total = self.n_guidestars.sum()
            n_zero = (self.n_guidestars == 0).sum()
            n_median = np.median(self.n_guidestars)
            n_mean = np.mean(self.n_guidestars)
            n_max = np.max(self.n_guidestars)

            print('Results:')
            print('--------------------------------------------')
            print(f'Guide stars selected:         {n_total:6.0f}')
            print(f'Fields without guide stars:   {n_zero:6d} '
                  f'({n_zero/n_fields*100:.1f}%)')
            print('--------------------------------------------')
            print(f'Median number of field stars: {n_median:6.0f}')
            print(f'Mean number of field stars:   {n_mean:6.0f}')
            print(f'Max number of field stars:    {n_max:6.0f}')

        if verbose > 1:
            print('--------------------------------------------')
            print('No. of guide stars: No. of fields')

            for value, count in zip(
                    *np.unique(self.n_guidestars, return_counts=True)):
                print(f'{value:2.0f}: {count:3d}')

            print('--------------------------------------------')

#==============================================================================

class GuideStarWalopS(GuideStarSelector):
    """A class to select guide stars for WALOP-South targets."""

    #--------------------------------------------------------------------------
    def _get_instrument_center(self, field_ra, field_dec):
        """Calculate the instrument center coordinates from a field's
        coordinates.

        Parameters
        ----------
        field_ra : float
            Field center right ascension in radians.
        field_dec : float
            Field center declination in radians.

        Returns
        -------
        instrument_center : astropy.coordinates.SkyCoord
            Instrument center coordinates corresponding to the input field
            center coordinates.

        Notes
        -----
        This method is called by `_select()`.
        """

        instrument_center = SkyCoord(
                field_ra + self.circle_offset.rad,
                field_dec + self.circle_offset.rad,
                unit='rad')

        return instrument_center

    #--------------------------------------------------------------------------
    def _select_in_circle(self, instrument_center, stars_coord):
        """

        Parameters
        ----------
        instrument_center : astropy.coordinates.SkyCoord
            Instrument center coordinates corresponding to the input field
            center coordinates.
        stars_coord : astropy.coordinates.SkyCoord
            Coordinates of stars from which so select those in the instrument
            circle area.

        Returns
        -------
        i_circle : numpy.ndarray
            Indices of the selected stars.
        selected_coord : astropy.coordinates.SkyCoord
            Coordinates of the selected stars.

        Notes
        -----
        This method is called by `_locate()`.
        """

        radius = self.circle_radius - self.limit
        sel_circle = stars_coord.separation(instrument_center) < radius
        i_circle = np.nonzero(sel_circle)[0]
        selected_coord = stars_coord[sel_circle]

        return i_circle, selected_coord

    #--------------------------------------------------------------------------
    def _locate(self, instrument_center, return_coord=False):
        """Select guide stars for one field.

        Parameters
        ----------
        instrument_center : astropy.coordinates.SkyCoord
            Instrument center coordinates corresponding to the input field
            center coordinates.
        return_coord : bool, optional
            If True, coordinates of the selected guide stars, of stars too
            close to the edge of the guider area, and stars within the
            instrument area are returned in addition to the indices of the
            selected stars. Otherwise, only the indices of the selected stars
            are returned.

        Returns
        -------
        i_guide : numpy.ndarray
            Indices of the stars in the guide area.
        coord_rot_guidearea : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area. Only returned if
            `return_coord=True`.
        coord_rot_edge : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area, but too close to the
            edge. Only returned if `return_coord=True`.
        coord_rot_circle : astropy.coordinates.SkyCoord
            Coordinates of the stars in the instrument area. Only returned if
            `return_coord=True`.

        Notes
        -----
        This method calls `_select_in_circle()`.
        This method is called by `_select()`.
        """

        # select closest stars:
        i_circle, candidates_coord = self._select_in_circle(
            instrument_center, self.stars_coord)

        # rotate coordinate frame:
        ra_rot, dec_rot = rotate_frame(
                candidates_coord.ra.rad, candidates_coord.dec.rad,
                instrument_center, tilt=self.instr_rot.rad)
        n = ra_rot.shape[0]

        # select candidates within guide area:
        sel_guide = np.zeros(n, dtype=bool)

        for i, point in enumerate(zip(ra_rot, dec_rot)):
            sel_guide[i] = inside_polygon(point, self.guide_area)

        i_guide = i_circle[sel_guide]
        n = i_guide.shape[0]

        # select candidates far enough from the guide area edges:
        sel_edge = np.zeros(n, dtype=bool)

        for i, point in enumerate(zip(ra_rot[sel_guide], dec_rot[sel_guide])):
            sel_edge[i] = close_to_edge(
                    point, self.guide_area, self.limit.rad)

        i_guide = i_guide[~sel_edge]

        if return_coord:
            coord_rot_circle = SkyCoord(ra_rot, dec_rot, unit='rad')
            coord_rot_edge = coord_rot_circle[sel_guide][sel_edge]
            coord_rot_guidearea = coord_rot_circle[sel_guide][~sel_edge]

            return (i_guide, coord_rot_guidearea, coord_rot_edge,
                    coord_rot_circle)

        else:
            return i_guide

    #--------------------------------------------------------------------------
    def _clear_from_stars_to_avoid(self, instrument_center, cam_pos):
        """Checks if the guide camera contains too bright stars.

        Parameters
        ----------
        instrument_center : astropy.coordinates.SkyCoord
            Coordinates where the telescope is pointed at.
        cam_pos : list of floats
            x- and y-position of the guide camera.

        Returns
        -------
        clear : bool
            True, if the guide camera region does not contain too bright stars.
            False, otherwise.
        coord_rot_avoid : astropy.coordinates.SkyCoord
            Coordinates of the bright stars in the guide camera area, rotated
            into a reference frame that the instrument center as origin.

        Notes
        -----
        This method is called by `_select()`.
        """

        # select closest stars:
        i_circle, avoid_coord = self._select_in_circle(
            instrument_center, self.avoid_coord)

        # rotate coordinate frame:
        ra_rot, dec_rot = rotate_frame(
                avoid_coord.ra.rad, avoid_coord.dec.rad, instrument_center,
                tilt=self.instr_rot.rad)
        coord_rot_avoid = SkyCoord(ra_rot, dec_rot, unit='rad')

        # define camera area:
        cam_area = [[cam_pos.ra.rad + point[0].rad,
                     cam_pos.dec.rad + point[1].rad]
                    for point in self.cam_area]

        # check for stars to avoid within camera area:
        clear = True

        for i, point in enumerate(zip(ra_rot, dec_rot)):
            if inside_polygon(point, cam_area):
                clear = False
                break

        return clear, coord_rot_avoid

    #--------------------------------------------------------------------------
    def _guider_position(self, coord_rot_guidearea):
        """Calculate the guide camera position for a selection of guide stars.

        Parameters
        ----------
        coord_rot_guidearea : astropy.coordinates.SkyCoord
            Coordinates of selected guide stars rotated into a reference frame
            that has its origin at the corresponding field center.

        Returns
        -------
        pos_x : numpy.array
            x-position of the guide camera in the units defined by the input
            parameter `scale_xy`.
        pos_y : numpy.array
            y-position of the guide camera in the units defined by the input
            parameter `scale_xy`.

        Notes
        -----
        This method is called by `_select()`.
        """

        pos_x = (coord_rot_guidearea.ra - self.home_pos[0]).rad \
                * -self.scale_xy
        pos_y = (coord_rot_guidearea.dec - self.home_pos[1]).rad \
                * -self.scale_xy

        return pos_x, pos_y

    #--------------------------------------------------------------------------
    def _select(self, field_ra, field_dec, return_coord=False, n_max=None):
        """Select guide stars from the candidate star list for a field.

        Parameters
        ----------
        field_ra : float,
            Right ascension of a field center position in radians. Also provide
            `field_dec`.
        field_dec : float
            Declination of a field center position in radians. Also provide
            `field_ra`.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        n_max : int, optional
            Maximum number of guide stars to select for each field. If None,
            all stars in the guider area are saved. The default is None.

        Returns
        -------
        guidestars : list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. The guide stars are sorted by increasing magnitude.

        If return_coord is True, the following items are returned as well:

        coord_rot_guidestars : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars, in a reference frame that
            has its origin at the instrument center.
        coord_rot_cand : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera, in a reference frame that has its origin at the instrument
            center.
        coord_rot_avoid : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera that should be avoided, in the rotated reference frame.

        Notes
        -----
        This method is called by `select()` or by `_iter_grid()`.
        This method calls `_select()`, `_locate()`, `_guider_position()` and
        `_clear_from_stars_to_avoid()`.
        """

        # locate guide stars in the guider area:
        instrument_center = self._get_instrument_center(field_ra, field_dec)
        i_guide, coord_rot_guidearea, coord_rot_edge, coord_rot_circle \
                = self._locate(instrument_center, return_coord=True)

        # get coordinates and magnitudes:
        candidates_coord = self.stars_coord[i_guide]
        candidates_mag = self.stars_mag[i_guide]

        # sort by brightness:
        i_sort = np.argsort(candidates_mag)
        candidates_coord = candidates_coord[i_sort]
        candidates_mag = candidates_mag[i_sort]
        coord_rot_guidearea = coord_rot_guidearea[i_sort]
        del i_sort

        if n_max is None:
            n_max = self.n_max

        if n_max == 0:
            n_max = np.inf

        # prepare list of guide stars:
        guidestars = []
        i_sel = []

        # iterate though selected guide star candidates:
        for i, (coord, coord_rot, mag) in enumerate(zip(
                candidates_coord, coord_rot_guidearea, candidates_mag)):

            # determine guider camera position:
            pos_x, pos_y = self._guider_position(coord_rot)

            # check if clear from stars that should be avoided:
            if self.avoid_coord is not None:
                clear, coord_rot_avoid = self._clear_from_stars_to_avoid(
                        instrument_center, coord_rot)
            else:
                clear = True
                coord_rot_avoid = None

            # store if clear:
            if clear:
                guidestars.append({
                        'guidestar_ra': coord.ra.rad,
                        'guidestar_dec': coord.dec.rad,
                        'guidestar_mag': mag,
                        'cam_pos_x': pos_x,
                        'cam_pos_y': pos_y})
                i_sel.append(i)

            # stop iteration if maximum number of guide stars is reached:
            if len(guidestars) >= n_max:
                break

        # calculate exposure times:
        if self.mag_to_exp is not None:
            for guidestar in guidestars:
                guidestar['exp'] = self.mag_to_exp(guidestar['guidestar_mag'])

        if return_coord:
            coord_rot_guidestars = coord_rot_guidearea[i_sel]

            return (guidestars, coord_rot_guidestars, coord_rot_guidearea,
                    coord_rot_edge, coord_rot_circle, coord_rot_avoid)

        else:
            return guidestars

    #--------------------------------------------------------------------------
    def select(
            self, fieldgrid=None, field_ra=None, field_dec=None,
            mag_to_exp=None, return_coord=False, verbose=1):
        """Select guide stars from the candidate star list for either each
        field in a field grid or for a specific field's coordinates.

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid. If not given, then provide `field_ra` and `field_dec`. The
            default is None.
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`. Alternatively, provide `fieldgrid`. The default is
            None.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`. Alternatively, provide `fieldgrid`. The default is
            None.
        mag_to_exp : callable, optional
            A function that takes magnitudes as argument and converts them to
            exposure times (in seconds). If None is given, no exposure times
            are calculated. The default is None.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        verbose : int, optional
            Controls the level of detail of information printed. The default is
            1.

        Raises
        ------
        ValueError
            Raised, if `mag_to_exp` is neither None nor a function.
        ValueError
            Raised, if neither `field_ra` and `field_dec` or `fieldgrid` are
            given.

        Returns
        -------
        If guide stars are selected for a specific field given through
        `field_ra` and `field_dec` the following data   structure is returned:

        guidestars : list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. Exposure times (exp) in seconds are included, if a magnitude
            to exposure time conversion function is given to the called
            `select()` method. The guide stars are sorted by increasing
            magnitude.

        If guide stars are selected for all fields in a grid given through
        `fieldgrid` the following data structure is returned:

        guidestars : list of dict
            Each list entry is a dictionary corresponding to one field in the
            grid. The dictionary contains the field's right ascension and
            declination and a list of associated guide stars. This list has the
            same data structure as explained above.

        If return_coord is True, the following items are returned as well:

        coord_rot_guidestars : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars, in a reference frame that
            has its origin at the instrument center.
        coord_rot_cand : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera, in a reference frame that has its origin at the instrument
            center.
        coord_rot_avoid : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera that should be avoided, in the rotated reference frame.

        Notes
        -----
        This method calls `_select()` for a single field or `_iter_grid()`,
        `_count_guidestars()`, and `check_results()` for a field grid.
        """

        return super().select(
                fieldgrid=fieldgrid, field_ra=field_ra, field_dec=field_dec,
                mag_to_exp=mag_to_exp, return_coord=return_coord,
                verbose=verbose)

    #--------------------------------------------------------------------------
    def set_params(
            self, circle_radius, circle_offset, field_size, cam_size,
            guide_area, home_pos, instr_rot=0, limit=0, scale=1, scale_xy=1,
            n_min=1, n_max=0):
        """Set science field and guide area parameters.

        Parameters
        ----------
        circle_radius : float
            Radius of the full instrument area. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
        circle_offset : float
            Offset of the circle center from the science field center. Can be
            given in any unit. Multiplied with the `scale` factor that converts
            unit to radians.
        field_size : float
            Size of the science field. Can be given in any unit. Multiplied
            with the `scale` factor that converts unit to radians.
        cam_size : tuple/list of floats
            Size of the camera image. The first value gives the horizontal (x)
            size, the second the vertical (y) size. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
        guide_area : array-like
            Two-dimensional array (or list of lists), where each element
            defines a corner point of the guide area. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
        home_pos : array-like
            The home position of the guide camera. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
        instr_rot : float, optional
            Instrument rotation in radians, counted clockwise. Use if the
            instrument is mounted such that the guide area is not facing North-
            East. The default is 0.
        limit : float, optional
            Only stars located off the edge of the guide area by at least this
            limit are selected as guide stars. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
            The default is 0.
        scale : float, optional
            Scale factor that converts the above values to radians. The default
            is 1.
        scale_xy : float, optional
            Scale factor that converts the guide camera position from radians
            to the designated unit. The default is 1.
        n_min : int, optional
            The minimum number of guide stars intended for each field. A
            warning is printed if some fields end up having fewer guide stars.
            If additional stars are added through `set_stars()`, fields that
            do not reach the minimum number are updated. The default is 1.
        n_max : int, optional
            The maximum number of guide stars selected for a field. If more are
            available, the brightest ones are selected. If 0, all available
            stars are selected. The default is 0.

        Raises
        ------
        ValueError
            Raised, if `scale` is negative or zero.
            Raised, if `scale_xy` is negative or zero.
            Raised, if `limit` is negative.
            Raised, if `cam_size` does not consist of two numbers.
            Raised, if `guide_area` is not 2-dimensional.
            Raised, if `home_pos` does not consist of two numbers.
            Raised, if `instr_rot` is not between 0 and 2*pi.
            Raised, if `n_min` is not int or is smaller than 1.
            Raised, if `n_max` is not int or is smaller than 0.

        Returns
        -------
        None
        """

        guide_area = np.asarray(guide_area)
        home_pos = np.asarray(home_pos)
        cam_size = np.asarray(cam_size)

        # check inputs:
        if scale <= 0:
            raise ValueError("`scale` must be > 0.")
        if scale_xy <= 0:
            raise ValueError("`scale_xy` must be > 0.")
        if limit < 0:
            raise ValueError("`limit` must be >= 0.")
        if cam_size.shape[0] != 2 or cam_size.ndim != 1:
            raise ValueError("`cam_size` must consist of two numbers.")
        if guide_area.ndim != 2:
            raise ValueError("`guide_area` must be 2-dimensional.")
        if home_pos.shape[0] != 2 or home_pos.ndim != 1:
            raise ValueError("`home_pos` must consist of two numbers.")
        if instr_rot < 0 or instr_rot >= 2 * np.pi:
            raise ValueError("`instr_rot` must be >=0 and <2*pi.")
        if not isinstance(n_min, int) or n_min < 1:
            raise ValueError("`n_min` must be int >= 1.")
        if not isinstance(n_min, int) or n_max < 0:
            raise ValueError("`n_max` must be int >= 0.")

        # store parameters:
        self.params = {
                'circle_radius': circle_radius,
                'circle_offset': circle_offset,
                'field_size': field_size,
                'cam_size': cam_size.tolist(),
                'guide_area': guide_area.tolist(),
                'home_pos': home_pos.tolist(),
                'instr_rot': instr_rot,
                'limit': limit,
                'scale': scale,
                'scale_xy': scale_xy,
                'n_min': n_min,
                'n_max': n_max}
        self.circle_radius = Angle(circle_radius*scale, unit='rad')
        self.circle_offset = Angle(circle_offset*scale, unit='rad')
        self.field_size = Angle(field_size*scale, unit='rad')
        self.cam_size = Angle(cam_size*scale, unit='rad')
        cam_area = [[-self.cam_size[0].rad/2, -self.cam_size[1].rad/2],
                    [self.cam_size[0].rad/2, -self.cam_size[1].rad/2],
                    [self.cam_size[0].rad/2, self.cam_size[1].rad/2],
                    [-self.cam_size[0].rad/2, self.cam_size[1].rad/2]]
        self.cam_area = Angle(cam_area, unit='rad')
        self.guide_area = \
                Angle(guide_area*scale, unit='rad') - self.circle_offset
        self.home_pos = Angle(home_pos*scale, unit='rad') - self.circle_offset
        self.instr_rot = Angle(instr_rot, unit='rad')
        self.limit = Angle(limit*scale, unit='rad')
        self.scale_xy = scale_xy
        self.n_min = n_min
        self.n_max = n_max

    #--------------------------------------------------------------------------
    def visualize_selection(
            self, coord_rot_guidestars, coord_rot_guidearea, coord_rot_edge,
            coord_rot_circle, coord_rot_avoid):
        """Visualize the guide star selection.

        Parameters
        ----------
        coord_rot_guidestars : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars.
        coord_rot_guidearea : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area.
        coord_rot_edge : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area, but too close to the
            edge.
        coord_rot_circle : astropy.coordinates.SkyCoord
            Coordinates of the stars in the instrument area.
        coord_rot_circle : astropy.coordinates.SkyCoord
            Coordinates of stars in the instrument area that should be avoided
            in the camera.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

        # plot instrument field:
        offset = self.circle_offset.arcmin
        radius = self.circle_radius.arcmin
        circle = plt.Circle(
                [0, 0], radius, fill=False, color='k', linestyle='-')
        plt.gca().add_artist(circle)
        plt.plot(0, 0, marker='+', ms=10, color='k')

        # plot science field:
        field_size = self.field_size.arcmin
        rectangle = plt.Rectangle(
                (-field_size/2-offset, -field_size/2-offset), field_size,
                field_size, fill=False, color='0.5', linestyle='-')
        plt.gca().add_artist(rectangle)
        plt.plot(-offset, -offset, marker='+', ms=10, color='0.5')

        # plot guide area:
        guide_area = self.guide_area.arcmin
        for ((x0, y0), (x1, y1)) in zip(
                    guide_area, np.r_[guide_area[1:], [guide_area[0]]]):
            plt.plot([x0, x1], [y0, y1], color='tab:orange', linestyle='-')

        # plot stars to avoid:
        if coord_rot_avoid is not None:
            ra = coord_rot_avoid.ra.arcmin
            ra = np.where(ra>180*60, ra-360*60, ra)
            plt.plot(
                    ra, coord_rot_avoid.dec.arcmin,
                    marker='o', linestyle='None', color='tab:red')

        # plot stars in instrument area:
        ra = coord_rot_circle.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_circle.dec.arcmin,
                marker='o', linestyle='None', color='0.7')

        # plot stars in guide area:
        ra = coord_rot_guidearea.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_guidearea.dec.arcmin,
                marker='o', linestyle='None', color='tab:blue')

        # plot stars close to edge:
        ra = coord_rot_edge.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_edge.dec.arcmin,
                marker='o', linestyle='None', color='tab:blue', mfc='w')

        # plot guide stars:
        ra = coord_rot_guidestars.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_guidestars.dec.arcmin,
                marker='o', linestyle='None', color='tab:orange')

        # edit figure:
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_aspect(1)
        xylim = radius * 1.1
        plt.xlim(-xylim, xylim)
        plt.ylim(-xylim, xylim)
        plt.ylabel('Dec offset from field center (arcmin)')
        plt.xlabel('RA offset from field center (arcmin)')

        return fig, ax

#==============================================================================

class GuideStarWalopN(GuideStarSelector):
    """A class to select guide stars for WALOP-North targets."""

    #--------------------------------------------------------------------------
    def _clear_from_stars_to_avoid(
            self, avoid_ra_rot, avoid_dec_rot, rotary_angle):
        """Checks if the guide camera contains too bright stars.

        Parameters
        ----------
        avoid_ra_rot : numpy.ndarray
            Right ascensions of the bright stars that need to be avoided, in a
            coordinate frame that has the instrument center as origin.
        avoid_dec_rot : numpy.ndarray
            Declinations of the bright stars that need to be avoided, in a
            coordinate frame that has the instrument center as origin.
        rotary_angle : float
            Angle at which the guide camera is positioned relative to the home
            position.

        Returns
        -------
        clear : bool
            True, if the guide camera region does not contain too bright stars.
            False, otherwise.

        Notes
        -----
        This method is called by `_select()`.
        """

        # rotate coordinates system such that guide cam is North:
        x, y, z = sphere_to_cart(avoid_ra_rot, avoid_dec_rot)
        x, y, z = rot_tilt(x, y, z, -rotary_angle)
        ra_rot, __ = cart_to_sphere(x, y, z)

        # check if all stars are outside of camera RA range:
        clear = np.all(np.logical_or(
            ra_rot < -self.cam_size[0].rad / 2,
            ra_rot > self.cam_size[0].rad / 2))

        return clear

    #--------------------------------------------------------------------------
    def _guider_position(self, ra_rot, dec_rot):
        """Calculate the guide camera position for a selection of guide stars.

        Parameters
        ----------
        ra_rot : astropy.coordinates.SkyCoord
            Right ascension (in radians) of selected guide stars rotated into a
            reference frame that has its origin at the corresponding field
            center.
        dec_rot : astropy.coordinates.SkyCoord
            Declination (in radians) of selected guide stars rotated into a
            reference frame that has its origin at the corresponding field
            center.

        Returns
        -------
        rotary_angle : numpy.array
            Rotary angle relative to the home position in radians.

        Notes
        -----
        This method is called by `_select()`.
        """

        rotary_angle = np.arctan2(dec_rot, ra_rot) - np.pi / 2
        rotary_angle -= self.home_pos.rad
        rotary_angle = np.mod(rotary_angle, 2 * np.pi)

        return rotary_angle

    #--------------------------------------------------------------------------
    def _locate(self, field_center, stars_coord):
        """Select guide stars for one field.

        Parameters
        ----------
        field_center : astropy.coordinates.SkyCoord
            Field center coordinates.
        stars_coord : astropy.coordinates.SkyCoord
            Coordinates of the stars from which to identify those that are
            located in the guide camera area.

        Returns
        -------
        i_guide : numpy.ndarray
            Indices of the stars in the guide area.

        Notes
        -----
        This method is called by `_select()`.
        """

        sel_outer = stars_coord.separation(field_center) \
            < self.guide_radius_outer
        sel_inner = stars_coord[sel_outer].separation(field_center) \
            < self.guide_radius_inner
        i_guide = np.nonzero(sel_outer)[0][~sel_inner]

        return i_guide

    #--------------------------------------------------------------------------
    def _select(self, field_ra, field_dec, return_coord=False, n_max=None):
        """Select guide stars from the candidate star list for a field.

        Parameters
        ----------
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        n_max : int, optional
            Maximum number of guide stars to select for each field. If None,
            all stars in the guider area are saved. The default is None.

        Returns
        -------
        guidestars : list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. The guide stars are sorted by increasing magnitude.

        If return_coord is True, the following items are returned as well:

        coord_rot_guidestars : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars, in a rotated reference
            frame that has its origin at the instrument center.
        coord_rot_cand : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera, in the rotated reference frame.
        coord_rot_avoid : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera that should be avoided, in the rotated reference frame.

        Notes
        -----
        This method is called by `select()` or by `_iter_grid()`.
        This method calls `_select()`, `_locate()`, `_guider_position()`, and
        `_clear_from_stars_to_avoid()`.
        """

        # locate guide stars in the guider area:
        field_center = SkyCoord (field_ra, field_dec, unit='rad')
        i_guide = self._locate(field_center, self.stars_coord)

        # get coordinates and magnitudes:
        candidates_coord = self.stars_coord[i_guide]
        candidates_mag = self.stars_mag[i_guide]

        # sort by brightness:
        i_sort = np.argsort(candidates_mag)
        candidates_coord = candidates_coord[i_sort]
        candidates_mag = candidates_mag[i_sort]
        del i_sort

        # rotate coordinates to field center (0, 0)-position:
        candidates_ra_rot, candidates_dec_rot = rotate_frame(
            candidates_coord.ra.rad, candidates_coord.dec.rad, field_center)

        # locate stars to avoid:
        if self.avoid_coord is not None:
            i_avoid = self._locate(field_center, self.avoid_coord)
            avoid_coord = self.avoid_coord[i_avoid]
            avoid_ra_rot, avoid_dec_rot = rotate_frame(
                avoid_coord.ra.rad, avoid_coord.dec.rad, field_center)

        if n_max is None:
            n_max = self.n_max

        if n_max == 0:
            n_max = np.inf

        # prepare list of guide stars:
        guidestars = []
        guidestars_ra_rot = []
        guidestars_dec_rot = []
        i_sel = []

        # iterate though selected guide star candidates:
        for i, (coord, ra_rot, dec_rot, mag) in enumerate(zip(
                candidates_coord, candidates_ra_rot, candidates_dec_rot,
                candidates_mag)):

            # determine guider camera position:
            rotary_angle = self._guider_position(ra_rot, dec_rot)

            # check if clear from stars that should be avoided:
            if self.avoid_coord is not None:
                clear = self._clear_from_stars_to_avoid(
                        avoid_ra_rot, avoid_dec_rot, rotary_angle)
            else:
                clear = True

            # store if clear:
            if clear:
                guidestars.append({
                        'guidestar_ra': coord.ra.rad,
                        'guidestar_dec': coord.dec.rad,
                        'guidestar_mag': mag,
                        'cam_rot': rotary_angle * self.scale_rot})
                guidestars_ra_rot.append(ra_rot)
                guidestars_dec_rot.append(dec_rot)
                i_sel.append(i)

            # stop iteration if maximum number of guide stars is reached:
            if len(guidestars) >= n_max:
                break

        # calculate exposure times:
        if self.mag_to_exp is not None:
            for guidestar in guidestars:
                guidestar['exp'] = self.mag_to_exp(guidestar['guidestar_mag'])

        if return_coord:
            coord_rot_cand = SkyCoord(
                candidates_ra_rot, candidates_dec_rot, unit='rad')
            coord_rot_guidestars = SkyCoord(
                guidestars_ra_rot, guidestars_dec_rot, unit='rad')
            coord_rot_avoid = SkyCoord(
                avoid_ra_rot, avoid_dec_rot, unit='rad')

            return (guidestars, coord_rot_guidestars, coord_rot_cand,
                    coord_rot_avoid)

        else:
            return guidestars

    #--------------------------------------------------------------------------
    def select(
            self, fieldgrid=None, field_ra=None, field_dec=None,
            mag_to_exp=None, return_coord=False, verbose=1):
        """Select guide stars from the candidate star list for either each
        field in a field grid or for a specific field's coordinates.

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid. If not given, then provide `field_ra` and `field_dec`. The
            default is None.
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`. Alternatively, provide `fieldgrid`. The default is
            None.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`. Alternatively, provide `fieldgrid`. The default is
            None.
        mag_to_exp : callable, optional
            A function that takes magnitudes as argument and converts them to
            exposure times (in seconds). If None is given, no exposure times
            are calculated. The default is None.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        verbose : int, optional
            Controls the level of detail of information printed. The default is
            1.

        Raises
        ------
        ValueError
            Raised, if `mag_to_exp` is neither None nor a function.
        ValueError
            Raised, if neither `field_ra` and `field_dec` or `fieldgrid` are
            given.

        Returns
        -------
        If guide stars are selected for a specific field given through
        `field_ra` and `field_dec` the following data   structure is returned:

        guidestars : list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. Exposure times (exp) in seconds are included, if a magnitude
            to exposure time conversion function is given to the called
            `select()` method. The guide stars are sorted by increasing
            magnitude.

        If guide stars are selected for all fields in a grid given through
        `fieldgrid` the following data structure is returned:

        guidestars : list of dict
            Each list entry is a dictionary corresponding to one field in the
            grid. The dictionary contains the field's right ascension and
            declination and a list of associated guide stars. This list has the
            same data structure as explained above.

        If return_coord is True, the following items are returned as well:

        coord_rot_guidestars : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars, in a rotated reference
            frame that has its origin at the instrument center.
        coord_rot_cand : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera, in the rotated reference frame.
        coord_rot_avoid : astropy.coordinates.SkyCoord
            Coordinates of all stars in the region accessible to the guide
            camera that should be avoided, in the rotated reference frame.

        Notes
        -----
        This method calls `_select()` for a single field or `_iter_grid()`,
        `_count_guidestars()`, and `check_results()` for a field grid.
        """

        return super().select(
                fieldgrid=fieldgrid, field_ra=field_ra, field_dec=field_dec,
                mag_to_exp=mag_to_exp, return_coord=return_coord,
                verbose=verbose)

    #--------------------------------------------------------------------------
    def set_params(
            self, guide_radius, field_size, cam_size, home_pos, limit=0,
            scale=1, scale_rot=1, n_min=1, n_max=0):
        """Set science field and guide area parameters.

        Parameters
        ----------
        guide_radius : float
            Distance between science field center and guide camera center. Can
            be given in any unit. Multiplied with the `scale` factor that
            converts unit to radians.
        field_size : float
            Size of the science field. Can be given in any unit. Multiplied
            with the `scale` factor that converts unit to radians.
        cam_size : tuple/list of floats
            Size of the camera image. The first value gives the horizontal (x)
            size, the second the vertical (y) size, when the camera would be
            positioned in the North. Can be given in any unit. Multiplied with
            the `scale` factor that converts unit to radians.
        home_pos : array-like
            The home position of the guide camera, counterclockwise relative to
            the North position. Must be given in radians.
        limit : float, optional
            Only stars located off the edge of the guide area by at least this
            limit are selected as guide stars. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
            The default is 0.
        scale : float, optional
            Scale factor that converts the above values to radians. The default
            is 1.
        scale_rot : float, optional
            Scale factor that converts the guide camera position from radians
            to the designated unit. The default is 1.
        n_min : int, optional
            The minimum number of guide stars intended for each field. A
            warning is printed if some fields end up having fewer guide stars.
            If additional stars are added through `set_stars()`, fields that
            do not reach the minimum number are updated. The default is 1.
        n_max : int, optional
            The maximum number of guide stars selected for a field. If more are
            available, the brightest ones are selected. If 0, all available
            stars are selected. The default is 0.

        Raises
        ------
        ValueError
            Raised, if `scale` is negative or zero.
            Raised, if `scale_xy` is negative or zero.
            Raised, if `limit` is negative.
            Raised, if `cam_size` does not consist of two numbers.
            Raised, if `guide_area` is not 2-dimensional.
            Raised, if `home_pos` does not consist of two numbers.
            Raised, if `instr_rot` is not between 0 and 2*pi.
            Raised, if `n_min` is not int or is smaller than 1.
            Raised, if `n_max` is not int or is smaller than 0.

        Returns
        -------
        None
        """

        home_pos = np.asarray(home_pos)
        cam_size = np.asarray(cam_size)

        # check inputs:
        if guide_radius <= 0:
            raise ValueError("`guide_radius` must be > 0.")
        if scale <= 0:
            raise ValueError("`scale` must be > 0.")
        if scale_rot <= 0:
            raise ValueError("`scale_rot` must be > 0.")
        if limit < 0:
            raise ValueError("`limit` must be >= 0.")
        if cam_size.shape[0] != 2 or cam_size.ndim != 1:
            raise ValueError("`cam_size` must consist of two numbers.")
        if not isinstance(n_min, int) or n_min < 1:
            raise ValueError("`n_min` must be int >= 1.")
        if not isinstance(n_min, int) or n_max < 0:
            raise ValueError("`n_max` must be int >= 0.")

        # store parameters:
        home_pos = np.mod(home_pos, 2 * np.pi)
        self.params = {
                'guide_radius': guide_radius,
                'field_size': field_size,
                'cam_size': cam_size.tolist(),
                'home_pos': home_pos,
                'limit': limit,
                'scale': scale,
                'scale_rot': scale_rot,
                'n_min': n_min,
                'n_max': n_max}
        self.guide_radius = Angle(guide_radius*scale, unit='rad')
        self.guide_radius_inner = Angle(
            (guide_radius - cam_size[0] / 2. + limit) * scale, unit='rad')
        self.guide_radius_outer = Angle(
            (guide_radius + cam_size[0] / 2. - limit) * scale, unit='rad')
        self.field_size = Angle(field_size*scale, unit='rad')
        self.cam_size = Angle(cam_size*scale, unit='rad')
        self.home_pos = Angle(home_pos, unit='rad')
        self.limit = Angle(limit*scale, unit='rad')
        self.scale_rot = scale_rot
        self.n_min = n_min
        self.n_max = n_max

    #--------------------------------------------------------------------------
    def visualize_selection(
            self, coord_rot_guidestars, coord_rot_cand, coord_rot_avoid):
        """Visualize the guide star selection.

        Parameters
        ----------
        coord_rot_guidestars : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars, in the reference fram that
            has the instrument center als origin.
        coord_rot_cand : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area, in the same rotated
            reference frame.
        coord_rot_avoid : astropy.coordinates.SkyCoord
            Coordinates of stars in the instrument area that should be avoided
            in the camera, in the same rotated reference frame.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

        # plot science field:
        field_size = self.field_size.arcmin
        rectangle = plt.Rectangle(
                (-field_size/2, -field_size/2), field_size, field_size,
                fill=False, color='0.5', linestyle='-')
        plt.gca().add_artist(rectangle)
        plt.plot(0, 0, marker='+', ms=10, color='0.5')

        # plot guide area:
        radius = self.guide_radius_inner.arcmin - self.limit.arcmin
        circle = plt.Circle(
            [0, 0], radius, fill=False, color='tab:orange', linestyle='-')
        plt.gca().add_artist(circle)
        radius = self.guide_radius_outer.arcmin - self.limit.arcmin
        circle = plt.Circle(
            [0, 0], radius, fill=False, color='tab:orange', linestyle='-')
        plt.gca().add_artist(circle)

        # plot stars to avoid in guide area:
        if coord_rot_avoid is not None:
            ra = coord_rot_avoid.ra.arcmin
            ra = np.where(ra>180*60, ra-360*60, ra)
            plt.plot(
                    ra, coord_rot_avoid.dec.arcmin,
                    marker='o', linestyle='None', color='tab:red', zorder=0)

        # plot candidates in guide area:
        ra = coord_rot_cand.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_cand.dec.arcmin,
                marker='o', linestyle='None', color='0.7', zorder=1)

        # plot guide stars:
        ra = coord_rot_guidestars.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_guidestars.dec.arcmin,
                marker='o', linestyle='None', color='tab:orange', zorder=2)

        # edit figure:
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_aspect(1)
        xylim = radius * 1.1
        plt.xlim(-xylim, xylim)
        plt.ylim(-xylim, xylim)
        plt.ylabel('Dec offset from field center (arcmin)')
        plt.xlabel('RA offset from field center (arcmin)')

        return fig, ax

#==============================================================================
