from pathlib import Path
import os
import warnings
from traitlets import observe

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from astropy.wcs import WCS
from astroquery.mast import MastMissions

from regions import (
    PolygonSkyRegion
)

from ipyaladin import Aladin

from mast_table import MastTable
from mast_table.base import mission_mast_ra_dec_colnames

from IPython.display import display
from sidecar import Sidecar
import ipywidgets as widgets
from ipypopout import PopoutButton

from mast_aladin.aida import AIDA
from mast_aladin.mixins import DelayUntilRendered
import mast_aladin.utils.parquet as parquet

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
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'MastAladin',
    'gca',
]

# store reference to the latest instantiation:
_latest_instantiated_app = None

# STScI's AWS S3 mirror of the DSS Color survey:
stpubdata_DSS_url = "https://stpubdata.s3.us-east-1.amazonaws.com/mast/skybackgrounds/DSSColor"


class MastAladin(Aladin, DelayUntilRendered, AIDA):
    """
    `MastAladin` extends `ipyaladin.widget.Aladin` with
    enhanced support for data products from MAST from `MAST <https://mast.stsci.edu/>`_.
    """
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

    def load_table(
        self,
        table,
        load_footprints=True,
        update_viewport=True,
        unique_column=None
    ):
        table_widget = MastTable(
            table,
            app=self,
            unique_column=unique_column,
            update_viewport=update_viewport
        )

        if load_footprints:
            if 's_region' in table.colnames:
                self.add_graphic_overlay_from_stcs(table['s_region'])
            else:
                raise ValueError(
                    "The table does not contain an `s_region` column, so no "
                    "footprints can be loaded."
                )

        return table_widget

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

    def show(self, loc="sidecar:split-right", title="mast-aladin", ref=None):
        """
        Display MastAladin.

        Parameters
        ----------
        loc : str, optional (default "sidecar:split-right")
            Location to display the MastAladin. Supported locations:
                "inline": Displays the MastAladin inline in a notebook.
                "sidecar": Displays the MastAladin in a separate JupyterLab window from the
                    notebook, with location decided by 'anchor'. Anchor options available from
                    ``jupyterlab-sidecar`` are:
                        {'split-right', 'split-left', 'split-top', 'split-bottom', 'tab-before',
                         'tab-after', 'right'}
                    Example loc for a sidecar at the bottom is ``loc='sidecar:split-bottom'``.
                "popout": Display the MastAladin in a detached display. By default, a new
                    window will open (browser popup permissions required). Anchor options are:
                        * ``popout:window`` (default, opens MastAladin in a new, detached popout)
                        * ``popout:tab`` (opens MastAladin in a new, detached tab in your browser)

        title : str, optional (default "mast-aladin")
            The title of the sidecar tab, only applicable to a "sidecar" display.

        ref: Sidecar | app, optional (default None)
            An existing reference sidecar or app with `.sidecar` attribute. Opens the new sidecar
            display with its anchor defined relative to the ref sidecar. Only applicable to a
            "sidecar" display.

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
            mission_mast=None,
            add_footprints=True,
            name="MAST Search",
            replace=False,
            **query_region_kwargs):
        """
        Search Missions MAST for observations from ``mission`` within the
        current viewport, and optionally add the footprints to the viewer.

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
            Mission MAST query results via astroquery.

        footprint_overlays : `~ipyaladin.overlays.overlay.Overlay` or None
            Overlay object for the footprint layer created by this method, if there
            is one.
        """
        if mission is not None and mission_mast is None:
            mission_mast = MastMissions(mission=mission)

        elif mission is None and mission_mast is not None:
            mission = mission_mast.mission.lower()

        elif mission and mission_mast:
            warnings.warn(
                "`search_this_area` was called with values for both `mission` and "
                "`mission_mast`. `mission_mast` takes precedent. `mission` will be "
                "ignored."
            )
            mission = mission_mast.mission.lower()

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

        query_result = mission_mast.query_region(
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
