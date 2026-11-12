from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import MastMissionsClass
from astroquery.exceptions import InvalidQueryError
import json
import roman_datamodels.datamodels as rdd
import pyarrow.parquet as pq
import logging
logger = logging.getLogger(__name__)


def get_parent(filename=None, mission_app=None, to_level=None):
    """
    Given any Roman product, find its constituent lower-level products in the MAST archive.
    For example, given a Roman Level 4 product, find the corresponding Roman Level 3 product(s).
    *NOTE* this currently only works for Level 4 --> Level 3 but will be extended in the future.

    As a first pass, the ``fileSetName`` derived from ``filename`` is used to
    query the archive directly via `MastMissionsClass.query_criteria`. If no Level 3
    match is found this way, a fallback search is performed by querying the archive
    using `MastMissionsClass.query_criteria` based on the metadata extracted from the file.

    Parameters
    ----------
    filename : str or Path
        The filename (or path) of a Roman Level 4 product.
    mission_app : `~astroquery.mast.missions.MastMissionsClass`
        A MastMissions client for the Roman mission.
    to_level: int, optional
        The product level to search for. Currently only supports 3 (Level 3).
    Returns
    -------
    `~astropy.table.Table`
        Table of Level 3 products matching either search. Empty if none are found.
    """

    # TODO: deprecate once the new `get_parent` function is fully implemented and tested.
    if to_level not in [3]:
        raise NotImplementedError("Only to_level=3 is currently supported.")

    # given the need for an API token to use MastMissions on test,
    # for now we need to require the user to pass in a MastMissionsClass instance
    # for the Roman mission.
    # TODO: consider adding a way to pass in an API token to create a MastMissionsClass instance
    if not isinstance(mission_app, MastMissionsClass):
        raise TypeError(
            f"mission_app must be an instance of MastMissionsClass, "
            f"not {type(mission_app)}."
        )

    stem = Path(filename).stem  # drop directory and extension
    fileset_name, _, _ = stem.rpartition('_')  # if it's an L4, it will end in "_segm" or "_cat"
    fileset_name = fileset_name or None

    logger.info("Searching for Level 3 products with fileSetName: %s", fileset_name)
    try:
        return _extract_product_from_metadata(mission_app=mission_app, level=3,
                                              **{'fileSetName': fileset_name, 'productLevel': 3})
    except (ValueError, InvalidQueryError):
        logger.debug("No Level 3 products found with fileSetName %s. "
                     "Attempting to extract original filename from metadata...", fileset_name)

    # treat the file as ASDF
    try:
        segmap = rdd.open(filename)
        items = {k.split('roman.meta.')[1]: v for k, v in segmap.to_flat_dict().items() if
                 'roman.meta' in k}

        query_items = {'coordinates': SkyCoord(ra=items['wcsinfo.ra'], dec=items['wcsinfo.dec'],
                                               unit=u.degree),
                       'instrument_name': items['instrument.name'],
                       'optical_element': items['instrument.optical_element'],
                       'execution_plan': items['observation.execution_plan'],
                       'program': items['observation.program'],
                       'exposure_time': items['coadd_info.exposure_time'],
                       'productLevel': 3
                       }

        return _extract_product_from_metadata(mission_app=mission_app, level=3, **query_items)
    except TypeError:
        logger.debug("Could not extract original filename assuming %s is ASDF. "
                     "Attempting to treat it as Parquet...", filename)

    # treat the file as parquet
    try:
        catalog = pq.read_schema(filename).metadata
        items = {k.decode().split('roman.meta.')[1]: v.decode("utf-8")
                 for k, v in catalog.items() if 'roman.meta' in k.decode()}
        query_items = {'coordinates': SkyCoord(ra=items['wcsinfo.ra'], dec=items['wcsinfo.dec'],
                                               unit=u.degree),
                       'instrument_name': items['instrument.name'],
                       'optical_element': items['instrument.optical_element'],
                       'execution_plan': items['observation.execution_plan'],
                       'program': items['observation.program'],
                       'exposure_time': items['coadd_info.exposure_time'],
                       'productLevel': 3
                       }
        return _extract_product_from_metadata(mission_app=mission_app, level=3, **query_items)
    except ValueError:
        logger.debug("Could not extract original filename assuming %s is Parquet. ", filename)

    # if the user has not provided a conventional Roman file,
    # has changed the metadata, or converted to FITS, raise an error
    raise ValueError(
        f"Could not extract original filename from {filename}. "
        f"Please provide a valid Roman Level 4 product file."
    )


def get_child(filename=None, mission_app=None, to_level=4):
    """
    Given any Roman product, find its constituent higher-level products in the MAST archive.
    For example, given a Roman Level 3 product, find the corresponding Roman Level 4 product(s).
    *NOTE* this currently only works for Level 3 --> Level 4 but will be extended in the future.

    As a first pass, the ``fileSetName`` derived from ``filename`` is used to
    query the archive directly via `MastMissionsClass.query_criteria`. If no Level 3
    match is found this way, a fallback search is performed by querying the archive
    using `MastMissionsClass.query_criteria` based on the metadata extracted from the file.

    Parameters
    ----------
    filename : str or Path
        The filename (or path) of a Roman Level 3 product.
    mission_app : `~astroquery.mast.missions.MastMissionsClass`
        A MastMissions client for the Roman mission.
    to_level: int, optional
        The product level to search for. Currently only supports 4 (Level 4).
    Returns
    -------
    `~astropy.table.Table`
        Table of Level 4 products matching either search. Empty if none are found.
    """

    # TODO: deprecate once the new `get_child` function is fully implemented and tested.
    if to_level not in [4]:
        raise NotImplementedError("Only to_level=4 is currently supported.")

    # given the need for an API token to use MastMissions on test,
    # for now we need to require the user to pass in a MastMissionsClass instance
    # for the Roman mission.
    # TODO: consider adding a way to pass in an API token to create a MastMissionsClass instance
    if not isinstance(mission_app, MastMissionsClass):
        raise TypeError(
            f"mission_app must be an instance of MastMissionsClass, "
            f"not {type(mission_app)}."
        )

    stem = Path(filename).stem  # drop directory and extension
    fileset_name, _, _ = stem.rpartition('_')  # if it's an L3, it will end in "_coadd" or "_asn"
    fileset_name = fileset_name or None

    logger.info("Searching for Level 4 products with fileSetName: %s", fileset_name)
    try:
        return _extract_product_from_metadata(mission_app=mission_app, level=4,
                                              **{'fileSetName': fileset_name})
    except (ValueError, InvalidQueryError):
        logger.debug("No Level 4 products found with fileSetName %s. "
                     "Attempting to extract original filename from metadata...", fileset_name)

    # treat the file as ASDF
    try:
        coadd = rdd.open(filename)
        items = {k.split('roman.meta.')[1]: v for k, v in coadd.to_flat_dict().items() if
                 'roman.meta' in k}

        query_items = {'coordinates': SkyCoord(ra=items['wcsinfo.ra'], dec=items['wcsinfo.dec'],
                                               unit=u.degree),
                       'instrument_name': items['instrument.name'],
                       'optical_element': items['instrument.optical_element'],
                       'execution_plan': items['observation.execution_plan'],
                       'program': items['observation.program'],
                       'exposure_time': items['coadd_info.exposure_time']
                       }

        return _extract_product_from_metadata(mission_app=mission_app, level=4, **query_items)
    except (TypeError, ImportError):
        logger.debug("Could not extract original filename assuming %s is ASDF. "
                     "Attempting to treat it as json...", filename)

    # treat the file as json
    try:
        with open(filename, 'r') as f:
            asn = json.load(f)
        fileset_name = asn['products'][0]['name']
        return _extract_product_from_metadata(mission_app=mission_app, level=4,
                                              **{'fileSetName': fileset_name})
    except (ValueError, InvalidQueryError):
        logger.debug("Could not extract original filename assuming %s is json. ", filename)

    # if the user has not provided a conventional Roman file,
    # has changed the metadata, or converted to FITS, raise an error
    raise ValueError(
        f"Could not extract original filename from {filename}. "
        f"Please provide a valid Roman Level 3 product file."
    )


def _extract_product_from_metadata(mission_app, level, **items):
    query = mission_app.query_criteria(**items)
    if len(query) == 0:
        raise ValueError(
            f"Could not find Level {level} products matching {items}. "
            f"Please provide a valid Roman Level {level} product file."
        )
    results = mission_app.get_unique_product_list(query)
    level_products = results[results['category'] == str(level)]
    return level_products
