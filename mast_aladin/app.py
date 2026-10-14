from pathlib import Path
import os
import warnings
from traitlets import observe

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import warnings

from astropy.io import fits, votable
from astropy.wcs import WCS
from astroquery.mast import MastMissions

from regions import (
    PolygonSkyRegion
)

from ipyaladin import Aladin

from mast_table.base import mission_mast_ra_dec_colnames

from IPython.display import display
from sidecar import Sidecar
import ipywidgets as widgets
from ipypopout import PopoutButton

from mast_aladin.aida import AIDA
from mast_aladin.mixins import DelayUntilRendered
import mast_aladin.utils.parquet as parquet

from astroquery.mast import MastMissionsClass

from mast_aladin.utils.validators import is_valid_s3_uri, check_table_for_columns
from mast_aladin.utils.validators import (
    is_valid_s3_uri, check_table_for_columns, guess_coord_cols
)
from mast_aladin.catalogs import (
    PerformanceCatalog,
    ConvexHullRegion,
    is_likely_an_observation_table,
)
from mast_aladin.catalogs.throttle import throttle
import roman_datamodels.datamodels as rdd
import os
import numpy as np


__all__ = [
    'MastAladin',
    'gca',
]

# store reference to the latest instantiation:
_latest_instantiated_app = None

# STScI's AWS S3 mirror of the DSS Color survey:
stpubdata_DSS_url = "https://stpubdata.s3.us-east-1.amazonaws.com/mast/skybackgrounds/DSSColor"

mast_footprint_color = '#ed702d'


class MastAladin(Aladin, DelayUntilRendered, AIDA):
    """
    `MastAladin` extends `ipyaladin.widget.Aladin` with
    enhanced support for data products from MAST from `MAST <https://mast.stsci.edu/>`_.
    """

    # list of `mast_aladin.catalogs.PerformanceCatalog` objects
    # initialized within this instance of MastAladin
    performance_catalogs = []

    def __init__(self, *args, **kwargs):
        """
        Some of the arguments to the `ipyaladin.widget.Aladin` constructor
        are listed below, see the `ipyaladin documentation for Aladin
        <https://cds-astro.github.io/ipyaladin/user_documentation/user_documentation.html>`_
        for the full set of available arguments.

        Parameters
        ----------
        target : `str`, `~astropy.coordinates.SkyCoord`
            The target of the Aladin Lite widget. The target can be provided as coordinates (either
            `~astropy.coordinates.SkyCoord` or tuple of `~astropy.coordinates.Longitude` and
            `~astropy.coordinates.Latitude`) or as a name (as a string).

            The conversion from a name to coordinates does call different online services
            depending on the base layer of the widget. For the sky, it uses CDS's
            `Sesame <https://vizier.cds.unistra.fr/viz-bin/Sesame>`_  name resolver. This
            looks first in SIMBAD, then in VizieR, and then in NED for the object's
            coordinates.

        fov : `int`, `float`, `~astropy.units.Quantity`
            Sets the field of view of the viewer. If int or float, units are degrees.

        survey : `str`
            URL pointing to a HiPS survey sky background. Default is the
            `Digitized Sky Survey <https://stdatu.stsci.edu/dss/getting_started.html>`,
            served via AWS S3. The full URL is:
            https://stpubdata.s3.us-east-1.amazonaws.com/mast/skybackgrounds/DSSColor

        projection : `str`
            Projection of the sky atlas in the display. Accepted values are:
            {'TAN', 'STG', 'SIN', 'ZEA', 'MER', 'AIT', 'MOL'}

        coo_frame : str
            Astronomical coordinate frame for grid lines and the coordinate search box.
            Accepted values are: {'ICRS', 'ICRSd', 'GAL'}

        rotation : `int`, `float`, `~astropy.units.Quantity`
            Rotate the field of view counter clockwise about its center. If int or float,
            units are degrees.

        height : `int`
            The height of the widget in pixels.
            Setting the height to -1 will expand the widget at 100% height of its
            container. This is generally a bad idea in a notebook but can be usefull
            for dashbord applications. The default height is 400 pixels.
        """

        # default survey is DSS served via AWS S3:
        kwargs.setdefault('survey', stpubdata_DSS_url)

        # set ICRSd as the default visible coordinate system
        # in aladin-lite:
        kwargs.setdefault('coo_frame', 'ICRSd')

        super().__init__(*args, **kwargs)

        global _latest_instantiated_app
        _latest_instantiated_app = self

        self.sidecar = kwargs.get("sidecar", None)
        self.popout_button = PopoutButton(self, window_features="popup,width=800,height=600")

    def add_table(
        self,
        table,
        include_column_names=None,
        shape="cross",
        name=None,
        ra_field=None,
        dec_field=None,
        n_sources_max=5_000,
        performance_cls=ConvexHullRegion,
        **table_options
    ):
        """
        Add ``table`` to mast-aladin. ``table`` may be a source catalog, an observation
        query result table, or a data product query result table.

        If ``table`` is a source catalog, display up to ``n_sources_max`` as scatter points,
        otherwise draw a region representing the convex hull of all source coordinates.

        See `~ipyaladin.widget.Aladin.add_table` for more details on the
        underlying implementation.

        Parameters
        ----------
        table : `~astropy.table.table.QTable` or `~astropy.table.table.Table` or `str`
            The table to add. Valid types are astropy table and S3 URIs of parquet files.

        include_column_names : list[str], optional
            List of column names from the remote parquet table to stream into mast-aladin.

        shape : str | `~ipyaladin.CircleError` | `~ipyaladin.EllipseError`, optional
            The shape to draw for each source. It accepts the strings "square",
            "circle", "plus", "cross", "rhomb", and "triangle" as well as the two
            specific classes `ipyaladin.CircleError` and `ipyaladin.EllipseError`
            that adapt the size of the drawn shapes (circles or ellipses) to error
            columns.
            See ipyaladin example notebook `04_Importing_Tables`.

        name : str, optional
            Name given to the overlay in Aladin.

        ra_field : str, optional
            For source catalog tables, ``ra_field`` is the column name in ``table``
            corresponding to the right ascension of a source. If not specified,
            it is automatically inferred from standard RA column naming conventions.

        dec_field : str, optional
            For source catalog tables, ``dec_field`` is the column name in ``table``
            corresponding to the declination of a source. If not specified, it is
            automatically inferred from standard Dec column naming conventions.

        performance_cls : Subclass of `~mast_aladin.catalogs.performance.PerformanceCatalog`
                          or None, optional
            Options include, for example,
            `~mast_aladin.catalogs.convex_hull_region.ConvexHullRegion` or
            `~mast_aladin.catalogs.priority_column_subset.PriorityColumnSubset`. Use
            `None`, to guarantee that no performance catalog will be constructed for
            this table.

        color : str, function, optional
            The color associated with the catalog. A function can be given similar
            to `shape`.

        source_size : int, float, function, optional
            The size of the sources in the catalog. A function can be given similar
            to `shape`.

        shape : str, function, Image, HTMLCanvasElement, HTMLImageElement, optional
            The shape of the sources (can be, "square", "circle", "plus", "cross", "rhomb",
            "triangle"). If a function is given, user can return Image, HTMLImageCanvas,
            HTMLImageElement or a string being in ["square", "circle", "plus", "cross",
            "rhomb", "triangle"]. This allows to define different shape for a specific
            catalog source.

        limit : int, float, optional
            The maximum number of sources to display.

        on_click : str, function, optional
            Whether the source data appears as a table row or a in popup. Can be
            'showTable' string, 'showPopup' string or a custom user defined
            function that handles the click.

        read_only : bool, optional
            Whether the catalog is read-only.

        filter : function, optional
            The filtering function for sources.

        selection_color : str, optional
            The color to apply to selected sources in the catalog.

        selection_line_width : str, optional
            The line width to apply to selected source footprints in the catalog
            (i.e. the shapes returned by a custom shape function).

        hover_color : str, optional
            The color to apply to sources in the catalog when they are hovered.

        display_label : bool, optional
            Whether to display labels for sources.

        label_column : str, optional
            The name of the column to be used for the label.

        label_color : str, optional
            The color of the source labels.

        label_font : str, optional
            The font for the source labels.

        only_footprints : bool, optional
            When shapes/footprints are associated to a source (through a shape function
            given), decide wheter to show the point source as well. Point source is
            hidden by default

        """
        # handle prefixes to parquet files on S3
        if isinstance(table, (os.PathLike, str)):
            if is_valid_s3_uri(table) and str(table).endswith('.parquet'):
                table = parquet.table_from_s3(table, include_column_names)
            else:
                raise ValueError(
                    f"table='{table}' is not a valid S3 URI for a parquet table."
                )

        if ra_field is not None and dec_field is not None:
            check_table_for_columns(table, ra_field, dec_field)
        else:
            # search for ra/dec columns
            ra_guess, dec_guess = guess_coord_cols(table)
            ra_field = ra_field or ra_guess
            dec_field = dec_field or dec_guess

        if isinstance(ra_field, SkyCoord):
            ra_field = ra_field.ra
            dec_field = ra_field.dec

        if '---' in [ra_field, dec_field]:
            raise ValueError(
                "Could not automatically determine RA/Dec columns in table. "
                "Please specify `ra_field` and `dec_field` explicitly."
            )

        # add an already-constructed performance catalog
        if isinstance(table, PerformanceCatalog):
            table._attach_to_mast_aladin(self)
            self.performance_catalogs.append(table)
            return table.overlay_info

        # call `Aladin.add_table` if (1) `performance_cls = None`, (2) the
        # table appears to be an observation query result table (contains an
        # s_region column), (3) n_sources < n_sources_max:
        elif performance_cls is None or (
                (len(table) < n_sources_max or
                 is_likely_an_observation_table(table))):

            # for small source catalogs, or observation tables (not source catalogs),
            # show all sources without performance optimization. Within the PerformanceLayer
            # framework, `performance_cls` is set to `None` in some calls to `add_table`
            # to explicitly avoid the performance behaviors.
            if not name:
                name = "catalog"

            with warnings.catch_warnings():
                # Ignore votable unit warning raised by ipyaladin.Aladin.add_table
                # for units that cannot be represented with vounits. The warning is
                # raised when add_table attempts to write the table object to a
                # votable bytes buffer:
                warnings.simplefilter('ignore', votable.exceptions.W50)

                return super().add_table(
                    table,
                    shape=shape,
                    name=name,
                    ra_field=ra_field,
                    dec_field=dec_field,
                    **table_options
                )

        # construct a performance catalog
        else:
            # for large catalogs, show sources within the viewport up to `n_sources_max`;
            # otherwise use `performance_cls` to show an "optimized" vizualization.
            # construct a performance catalog layer:

            if name is None:
                name = f"Catalog {len(self.performance_catalogs) + 1}"

            catalog_layer = performance_cls(
                table,
                name=name,
                ra_field=ra_field,
                dec_field=dec_field,
                n_sources_max=n_sources_max,
                shape=shape,
                **table_options
            )
            catalog_layer._attach_to_mast_aladin(self)
            self.performance_catalogs.append(catalog_layer)
            return catalog_layer.overlay_info

    @observe('_target', '_rotation', '_fov_xy')
    @throttle(1)
    def _on_viewport_update(self, change):
        # on updates to the viewport center/pan (_target), rotation (_rotation),
        # or the viewport zoom/browser resize/widget aspect ratio (_fov_xy):
        for catalog in self.performance_catalogs:
            catalog._on_viewport_update(change)

    def add_asdf(
        self, asdf, **image_options
    ):
        """Load an ASDF image into the widget.

        Parameters
        ----------
        asdf : Union[str or Path-like, rdd]
            The ASDF image to load in the widget. It can be given as a path (either a
            string or as a `roman_datamodels.datamodels._datamodels.ImageModel`).
        image_options : any
            The options for the image. See the `Aladin Lite image options
            <https://cds-astro.github.io/aladin-lite/global.html#ImageOptions>`_

        """
        if isinstance(asdf, rdd._datamodels.ImageModel):
            asdf_file = asdf
        else:
            asdf_file = rdd.open(asdf)

        wcs_header = fits.Header(asdf_file.meta.wcs.to_fits()[0])

        hdu_list = fits.HDUList(
            [
                fits.PrimaryHDU(header=wcs_header),
                fits.ImageHDU(
                    header=wcs_header,
                    data=asdf_file.data
                )
            ]
        )

        self.add_fits(hdu_list, **image_options)

    def add_fits(
        self, f, extension=1, **image_options
    ):
        """Load a FITS image into the widget.

        Parameters
        ----------
        f : Union[str, Path, HDUList]
            The FITS image to load in the widget. It can be given as a path (either a
            string or a `pathlib.Path` object), or as an `astropy.io.fits.HDUList`.
        extension: int, optional
            FITS extension containing the image data to load. Default is 1.
        image_options : any
            The options for the image. See the `Aladin Lite image options
            <https://cds-astro.github.io/aladin-lite/global.html#ImageOptions>`_

        """

        # Wraps add_fits in ipyaladin to temporarily handle SIP.
        # See ipyaladin for definitions of parameters.

        is_path = isinstance(f, (Path, str))
        if is_path:
            fits_file = fits.open(f)
        else:
            fits_file = f

        if len(fits_file) == 1:
            extension = 0

        data = fits_file[extension].data
        wcs = WCS(fits_file[extension].header)

        if data is None:
            raise ValueError(
                f"No data in extension {extension}."
            )

        wcs.sip = None

        wcs_header = wcs.to_header()

        hdu_list = fits.HDUList(
            [
                fits.PrimaryHDU(header=wcs_header),
                fits.ImageHDU(
                    header=wcs_header,
                    data=data
                )
            ]
        )

        super().add_fits(hdu_list, **image_options)

    def get_viewport_region(self, center=False):
        """Return a `regions.PolygonSkyRegion` representing the perimeter of the
        MastAladin viewport.

        Parameters
        ----------
        center : bool, optional
            If `False` (default), return a region where the vertices are the
            the outer corners of the corner pixels; otherwise the vertices will
            be the corner pixel centers.

        Returns
        -------
        `regions.PolygonSkyRegion`
            Region with vertices representing the corners of the current field
            of view in the viewport.
        """

        sky_corners = SkyCoord(
            self.wcs.calc_footprint(undistort=False, center=center),
            unit='deg'
        )
        return PolygonSkyRegion(sky_corners)

    def _apply_column_filters(table, filters):
        """
        Filter a table by exact-match (or membership, for list/tuple/set values)
        criteria on one or more columns.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table to filter.
        filters : dict or None
            Mapping of column name to an acceptable value, or a list/tuple/set of
            acceptable values (OR logic within a column, AND logic across columns).
            e.g. ``{"optical_element": ["F158", "F184"]}``.

        Returns
        -------
        `~astropy.table.Table`
            The filtered table. Returned unchanged if ``filters`` is `None` or empty.
        """
        if not filters:
            return table

        mask = np.ones(len(table), dtype=bool)
        for column, value in filters.items():
            if column not in table.colnames:
                raise KeyError(
                    f"Column '{column}' not found in results table. "
                    f"Available columns: {table.colnames}"
                )
            if isinstance(value, (list, tuple, set)):
                mask &= np.isin(table[column], list(value))
            else:
                mask &= (table[column] == value)

        return table[mask]

    def find_level3_products(self, l4_filename,
                             mission_app,
                             coordinates=None,
                             radius=1 * u.arcsec,
                             filters=None,
                             download=False):
        """
        Given a Roman Level 4 product, find the corresponding Roman Level 3 product(s)
        in the MAST archive.

        As a first pass, the ``fileSetName`` derived from ``l4_filename`` is used to
        query the archive directly via
        `MastMissionsClass.query_criteria`. If no Level 3
        match is found this way, a fallback search is performed using
        `MastMissionsClass.query_region` centered on ``ra``/``dec`` within ``radius``;
        if None then the center of the current mast_aladin viewport.

        Parameters
        ----------
        l4_filename : str or Path
            The filename (or path) of a Roman Level 4 product.
        coordinates : `~astropy.coordinates.SkyCoord`, optional
            SkyCoord object containing an optional position to center coordinate search.
            Only used as a fallback if the filename-based search returns no Level 3 matches
            and user wants to specify the ra/dec.
            If None, use current viewport center from aida methods.
        radius : ~astropy.units.Quantity, optional
            Search radius around ``ra``/``dec`` for the fallback coordinate search.
            Default is 1 arcsecond.
        mission_app : `~astroquery.mast.missions.MastMissionsClass`, optional
            A MastMissions client for the Roman mission. If not provided, a new
            client instance is created.
        filters : dict or None, optional
            Mapping of column name to an acceptable value, or a list/tuple/set of
            acceptable values (OR logic within a column, AND logic across columns).
            e.g. ``{"optical_element": ["F158", "F184"]}``.
        download : bool, optional
            Whether to download the retrieved L3 products to a local path. Default is True.

        Returns
        -------
        `~astropy.table.Table`
            Table of Level 3 products matching either search. Empty if none are found.
        """

        # given the need for an API token to use MastMissions on test,
        # for now we need to require the user to pass in a MastMissionsClass instance
        # for the Roman mission.
        # TODO: consider adding a way to pass in an API token to create a MastMissionsClass instance
        if not isinstance(mission_app, MastMissionsClass):
            raise TypeError(
                f"mission_app must be an instance of MastMissionsClass, "
                f"not {type(mission_app)}."
            )

        fileset_name = fileset_name_from_filename(l4_filename)
        print(rf"Searching for Level 3 products with fileSetName: {fileset_name}")

        results = mission_app.get_unique_product_list(fileSetName=fileset_name)
        mask = np.array(['3' in level for level in results["productLevel"]])
        level3 = results[mask]

        if len(level3) == 0:
            if 'segm' in l4_filename:
                # the segmentation map retains the original filename in its metadata,
                # even if the user renamed the input file
                segmap = rdd.open(l4_filename)
                orig_filename = segmap.meta.filename
                fileset_name = fileset_name_from_filename(orig_filename)
                results = mission_app.get_unique_product_list(fileSetName=fileset_name)
                mask = np.array(['3' in level for level in results["productLevel"]])
                level3 = results[mask]

            elif 'cat' in l4_filename:
                # photometric and spectroscopic catalogs have less metadata
                # so perform an astroquery search by coordinates instead of by fileSetName
                warnings.warn(f"No level 3 products found for fileSetName '{fileset_name}'. "
                              "Falling back to a coordinate-based search.")
                if coordinates is None:
                    coordinates = self.get_viewport("sky")["center"]
                results = mission_app.query_region(coordinates, radius=radius)
                mask = np.array(['3' in level for level in results["productLevel"]])
                level3 = self._apply_column_filters(results[mask], filters)

            if download:
                mission_app.download_products(level3)['Local Path']

        return level3


def fileset_name_from_filename(filename):
    """
    Derive the MAST Roman ``fileSetName`` from a Level 3 or Level 4 product filename.

    Roman Level 3 (mosaic) and Level 4 (catalog/segmentation map) products for the
    same field share an identical ``fileSetName`` (e.g.
    ``r00004_r0_full_270p65x69y50_f158``). This string is a prefix shared by every
    product filename for that field, followed by a single underscore-delimited
    suffix identifying the product type - e.g. ``_coadd.asdf`` for the Level 3
    mosaic, or ``_segm.asdf``/``_cat.parquet`` for Level 4 products.

    Parameters
    ----------
    filename : str or path
        The filename (or full path) of a Roman Level 4 product.

    Returns
    -------
    str
        The ``fileSetName`` prefix shared by all products for this field,
        including the corresponding Level 3 product.
    """
    stem = Path(filename).stem  # drop directory and extension
    fileset_name, middle, _suffix = stem.rpartition('_')
    return fileset_name

    def show(self, loc="sidecar:split-right", title="mast-aladin", ref=None):
        """
        Display MastAladin.

        Parameters
        ----------
        loc : str, optional. Default "sidecar:split-right"
            Location to display the MastAladin. Supported locations include:

            - "inline": Displays the MastAladin inline in a notebook.
            - "sidecar": Displays the MastAladin in a separate JupyterLab window from the
              notebook, with location decided by 'anchor'. Anchor options available from
              ``jupyterlab-sidecar`` are: {'split-right', 'split-left', 'split-top',
              'split-bottom', 'tab-before', 'tab-after', 'right'}.
              Example ``loc`` for a sidecar at the bottom is ``loc='sidecar:split-bottom'``.

            - "popout": Display the MastAladin in a detached display. By default, a new
              window will open (browser popup permissions required). Anchor options are
              ``popout:window`` (default, opens MastAladin in a new, detached popout) or
              ``popout:tab`` (opens MastAladin in a new, detached tab in your browser).

        title : str, optional. Default "mast-aladin"
            The title of the sidecar tab, only applicable to a "sidecar" display.

        ref : sidecar.Sidecar [1]_ or app, optional. Default None.
            An existing reference sidecar or app with `.sidecar` attribute. Opens the new sidecar
            display with its anchor defined relative to the ref sidecar. Only applicable to a
            "sidecar" display.

        References
        ----------
        .. [1] `JupyterLab Sidecar
               <https://github.com/jupyter-widgets/jupyterlab-sidecar#sidecar>`_.

        """
        valid_anchors = [
            "split-right", "split-left", "split-top", "split-bottom",
            "tab-before", "tab-after", "right", None
        ]

        if not isinstance(loc, str):
            raise TypeError("Invalid loc type provided. Must be a string.")

        if loc == "inline":
            display(self)
        elif loc.startswith("sidecar"):
            anchor = None if loc == "sidecar" else loc.split(":")[-1]

            if anchor not in valid_anchors:
                raise ValueError(
                    "Invalid anchor provided. Must be one of the available from "
                    "``jupyterlab-sidecar``. Valid anchors are: "
                    f"{valid_anchors[:-1]}")

            if ref and not isinstance(ref, Sidecar):
                try:
                    ref = ref.sidecar
                except AttributeError:
                    raise ValueError(
                        "Invalid ref provided. Must be a `Sidecar` instance or an "
                        "app with `.sidecar` attribute."
                    )

                if ref is None or not isinstance(ref, Sidecar):
                    raise ValueError(
                        "Invalid ref provided. App must have an initialized `.sidecar` "
                        "attribute."
                    )

            # When displaying mast-aladin in a sidecar, set the app height
            # to `-1`, which is meant to make ipyaladin fill its container.
            # Note: `height=-1` does not work as intended when the widget
            # is shown inline or in a popout.
            self.height = -1
            sc = Sidecar(title=title, anchor=anchor, ref=ref)
            with sc:
                display(self)
            self.sidecar = sc

        elif loc.startswith("popout"):
            anchor = None if loc == "popout" else loc.split(":")[-1]

            if not hasattr(self, "_popout_output"):
                self._popout_output = widgets.Output()

            with self._popout_output:
                self.height = 600
                display(self.popout_button)

            if anchor in (None, "window"):
                self.popout_button.open_window()
            elif anchor == "tab":
                self.popout_button.open_tab()
            else:
                raise ValueError(
                    "Invalid anchor provided. Must be 'window' or 'tab'."
                )
        else:
            raise ValueError("Invalid loc provided. Must be 'inline', 'sidecar', or 'popout'.")

    @property
    def viewport_to_cone_search_args(self):
        """
        Get positional arguments for a cone search that circumscribes
        the current viewport boundaries.

        Results may be passed to `~astroquery.mast.missions.MastMissionsClass`,
        for example.

        Returns
        -------
        dict
            Dictionary of the viewport center (``coordinates``) and the
            angular separation from the center to a corner of the viewport
            (``radius``).
        """
        fov_x, fov_y = self.fov_xy
        coordinates = self.target

        # measured from the center to the corner of the viewport:
        radius = ((fov_x / 2) ** 2 + (fov_y / 2) ** 2) ** 0.5

        return dict(
            coordinates=coordinates,
            radius=radius
        )

    def search_this_area(
            self,
            mission=None,
            mission_search=None,
            add_footprints=True,
            name="MAST Search",
            replace=False,
            color=mast_footprint_color,
            **query_region_kwargs):
        """
        Query MAST `Mission Search <https://mast.stsci.edu/search/ui/#/>`_ for observations
        from ``mission`` within the current viewport, and optionally add the
        footprints to the viewer.

        Runs a cone search via ``MastMissions.query_region``. The cone center
        coordinate is the center of the viewport, and the cone radius
        is the separation from the viewport's center to its corner, calculated
        by ``~mast_aladin.app.MastAladin.viewport_to_cone_search_args``.

        Missions MAST cone searches are limited to radii of <=30 arcmin. If the current
        viewport limits produce a radius >30 arcmin, a warning is raised and the
        query is completed with a radius of 30 arcmin.

        Extra keyword arguments are passed to ``MastMissions.query_region``.

        Users can also provide a pre-configured `~astroquery.mast.missions.MastMissionsClass`
        instance via ``mission_mast``. If both ``mission`` and ``mission_mast``
        are specified, ``mission`` takes precedent.

        Note: the maximum cone search radius supported by
        `~astroquery.mast.missions.MastMissionsClass` is 30 arcmin.

        Parameters
        ----------
        mission : str {'hst', 'jwst', 'roman'}, optional
            Mission to query via Missions Mast. If None, ``mission_mast`` must
            be given.
        mission_mast : `~astroquery.mast.missions.MastMissionsClass`, optional
            User-defined instance of `~astroquery.mast.missions.MastMissionsClass`. If None,
            ``mission`` must be given.
        add_footprints : bool, default True
            Add the query results' observation footprints to Aladin
        name : str, optional
            Name given to footprint overlays in Aladin.
        replace : bool, default False
            If True and footprint overlays already exist with the name ``name``, replace
            the existing footprint overlay with the new query result.
        color : str, default "#ed702d"
            Hex string specifying the color of footprint overlays created by
            this method.
        limit : int
            Default is 5000. The maximum number of dataset IDs in the results.
        offset : int
            Default is 0. The number of records you wish to skip before selecting
            records.
        select_cols: iterable or str or None, optional
            Default is None. Names of columns that will be included in the result
            table. If None, a default set of columns will be returned.
            Can either be an iterable of column names, a comma-separated string
            of column names, or 'all'/'*' to return all available columns.
        count_only : bool, optional
            Default is False. If True, only the count of matching datasets will
            be returned.
        **criteria
            Other mission-specific criteria arguments.
            All valid filters can be found using
            `~astroquery.mast.missions.MastMissionsClass.get_column_list` function.
            For example, one can specify the output columns(select_cols) or use other
            filters(conditions). To filter by multiple values for a single column,
            pass in a list of values or a comma-separated string of values. For the
            Roman mission, you can also use the special "pass_id"

        Returns
        -------
        query_result : `~astropy.table.Table`
            MAST Mission Search query results via astroquery.

        footprint_overlays : `~ipyaladin.overlays.overlay.Overlay` or None
            Overlay object for the footprint layer created by this method, if there
            is one.
        """
        if mission is not None and mission_search is None:
            mission_search = MastMissions(mission=mission)

        elif mission is None and mission_search is not None:
            mission = mission_search.mission.lower()

        elif mission and mission_search:
            warnings.warn(
                "`search_this_area` was called with values for both `mission` and "
                "`mission_mast`. `mission_mast` takes precedent. `mission` will be "
                "ignored."
            )
            mission = mission_search.mission.lower()

        else:
            raise ValueError("One of the arguments `mission` or `mission_mast` must be given.")

        if 'coordinates' in query_region_kwargs or 'radius' in query_region_kwargs:
            raise ValueError(
                "`search_this_area` does not accept the MastMissions arguments "
                "`coordinates` or `radius` as keyword arguments. The values for these "
                "arguments are set by `search_this_area` based on the current viewport. "
            )

        search_coordinates = self.viewport_to_cone_search_args

        if search_coordinates['radius'] > 30 * u.arcmin:
            warnings.warn(
                "Mission MAST cone searches are limited to 30 arcminutes in "
                "radius, but the current viewport limits are larger. The MAST cone "
                "search will be done with a radius of 30 arcmin."
            )
            search_coordinates['radius'] = 30 * u.arcmin

        query_result = mission_search.query_region(
            **search_coordinates,
            **query_region_kwargs
        )

        # will return `footprint_overlays`, set to None in case no overlay gets created:
        footprint_overlays = None

        if add_footprints and len(query_result):
            ra_field, dec_field = mission_mast_ra_dec_colnames[mission]

            if replace and name in self.overlays:
                self.remove_overlay(name)

            footprint_overlays = self.add_table(
                query_result,
                name=name,
                ra_field=ra_field,
                dec_field=dec_field,
                only_footprints=True,
                color=color
            )

            if 's_region' not in query_result.colnames:
                warnings.warn(
                    "Footprints can't be added because the query result "
                    "contains no 's_region' column.", UserWarning
                )

        if len(query_result) == 0:
            # this warning will appear after astroquery.mast's warning that says:
            #  "NoResultsWarning: Query returned no results."
            warnings.warn(
                "`search_this_area` returning None. "
            )

        return query_result, footprint_overlays


def gca():
    """
    Get the current mast-aladin application instance.
    If none exist, create a new one.

    Returns
    -------
    `~mast_aladin.app.MastAladin`
    """
    if _latest_instantiated_app is None:
        return MastAladin()

    return _latest_instantiated_app
