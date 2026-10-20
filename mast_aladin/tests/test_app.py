from unittest.mock import Mock, patch
from astropy.table import Table
from astroquery.exceptions import InvalidQueryError
from astroquery.mast import MastMissionsClass
from astropy.table import Table, QTable
import pytest

from mast_aladin.app import MastAladin, gca
from mast_aladin.utils.validators import guess_coord_cols


def test_current_app(MastAladin_app):
    # MastAladin_app should be the current instance of the app
    assert gca() == MastAladin_app

    # create new app instance
    instance2 = MastAladin()

    # gca should refer to the newly instantiated app:
    assert gca() == instance2


@patch("mast_aladin.utils.parquet.table_from_s3")
def test_add_astropy_table(mock_table_from_s3, MastAladin_app):
    """
    Test that the add_table method functions as defined by ipyaladin when
    given an astropy table, and that parquet logic is not invoked.
    """
    table = Table(names=['ra', 'dec'], data=[[10.0], [-5.0]])

    result = MastAladin_app.add_table(table)

    mock_table_from_s3.assert_not_called()
    assert result["type"] == "table"


@patch("mast_aladin.utils.parquet.table_from_s3")
def test_add_parquet_table(mock_table_from_s3, MastAladin_app):
    """
    Test that the add_table method correctly handles parquet URIs and invokes
    the super method from ipyaladin.
    """
    mock_table = Table(names=['ra', 'dec'], data=[[10.0], [-5.0]])
    mock_table_from_s3.return_value = mock_table
    parquet_uri = "s3://some-bucket/test.parquet"

    result = MastAladin_app.add_table(
        parquet_uri,
        shape="circle",
        include_column_names=["ra", "dec"]
    )

    mock_table_from_s3.assert_called_once_with(
        parquet_uri,
        ["ra", "dec"]
    )
    assert result["type"] == "table"
    assert result['options']['shape'] == "circle"


@patch("mast_aladin.utils.parquet.table_from_s3")
def test_add_invalid_table(mock_table_from_s3, MastAladin_app):
    """
    Test that an invalid parquet table not from S3 raises a ValueError
    """
    mock_table = Mock()
    mock_table_from_s3.return_value = mock_table
    parquet_uri = "https://some-bucket/test.parquet"

    with pytest.raises(ValueError):
        MastAladin_app.add_table(parquet_uri)

    mock_table_from_s3.assert_not_called()


def test_guess_ra_dec_columns(MastAladin_app):
    """
    Test that the guess_coord_cols function correctly identifies RA and Dec columns
    from a given astropy table.
    """
    # these are the unique variations in VO TAP query outputs
    ra_variations = ['ALPHA_J2000', 'RA', 'RA_deg', 'matchra', 'ra', 'ra_img', 'radeg',
                     'ramean', 's_ra', 'sci_ra', 'targ_ra', 'trgposra']
    dec_variations = ['DEC', 'DEC_deg', 'DELTA_J2000', 'dec', 'decdeg', 'decl',
                      'decl_img', 'decmean', 'matchdec', 's_dec', 'sci_dec', 'targ_dec',
                      'trgposdec']

    variations_to_pass = list(zip(ra_variations, dec_variations))

    for v in variations_to_pass:
        ra, dec = v
        tab = QTable({ra: [10.0], dec: [-5.0]})

        guess_ra, guess_dec = guess_coord_cols(tab)
        assert guess_ra == ra
        assert guess_dec == dec
        MastAladin_app.add_table(tab)

    # check that certain strings that contain 'ra' and 'dec' substrings are not
    # misidentified as coordinate columns
    tab = QTable({'radial_velocity': [10.0], 'fluxradius': [5.0], 'decrement': [1.0]})
    guess_ra, guess_dec = guess_coord_cols(tab)
    # none of the column names in the input table should have been identified as RA or Dec columns,
    # so they should be set as a placeholder value of '---'
    assert guess_ra == '---'
    assert guess_dec == '---'


def test_illegal_column_names(MastAladin_app):
    tab = QTable({'ra': [10.0], 'dec': [5.0]})

    with pytest.raises(ValueError):
        MastAladin_app.add_table(tab, ra_field='nonexistent_ra', dec_field='nonexistent_dec')


def test_show_invalid_locs():
    """
    Test that the show method correctly raises errors for invalid locs.
    """
    ma = MastAladin()

    # check invalid loc type
    with pytest.raises(TypeError, match="Invalid loc type provided. Must be a string."):
        ma.show(loc=1234)

    # check invalid sidecar anchor
    with pytest.raises(ValueError, match="Invalid anchor provided. Must be one of"):
        ma.show(loc="sidecar:invalid")

    # check invalid sidecar ref
    with pytest.raises(ValueError, match="Invalid ref provided. Must be a `Sidecar`"):
        ma.show(ref="invalid_ref")
    with pytest.raises(ValueError, match="Invalid ref provided. App must have an"):
        ma.show(ref=ma)
    ma.sidecar = "fake"
    with pytest.raises(ValueError, match="Invalid ref provided. App must have an"):
        ma.show(ref=ma)

    # check invalid popout anchor
    with pytest.raises(
        ValueError,
        match="Invalid anchor provided. Must be 'window' or 'tab'"
    ):
        ma.show(loc="popout:invalid")

    # check invalid loc str
    with pytest.raises(
        ValueError,
        match="Invalid loc provided. Must be 'inline', 'sidecar', or 'popout'."
    ):
        ma.show(loc="invalid")


ORIG_FILENAME = "r0034201001001001001_0001_wfi01_f087_coadd.asdf"
L3_FILESET_NAME = ORIG_FILENAME.rpartition('_')[0]


def _mock_mission_app():
    mission_app = Mock(spec=MastMissionsClass)
    l3_table = Table({"category": ["3"], "fileSetName": [L3_FILESET_NAME]})

    def get_unique_product_list(fileset_name):
        if fileset_name == L3_FILESET_NAME:
            return l3_table
        raise InvalidQueryError("no match")

    mission_app.get_unique_product_list.side_effect = get_unique_product_list
    return mission_app


def test_mock_mission_app_get_unique_product_list():
    """
    Unit test for the `get_unique_product_list` side-effect function used
    by the `_mock_mission_app` test helper: it should return the Level 3
    product table when queried with the matching fileSetName, and raise
    `InvalidQueryError` for any other fileSetName.
    """
    mission_app = _mock_mission_app()

    result = mission_app.get_unique_product_list(L3_FILESET_NAME)
    assert len(result) == 1
    assert result["fileSetName"][0] == L3_FILESET_NAME
    assert result["category"][0] == "3"

    with pytest.raises(InvalidQueryError, match="no match"):
        mission_app.get_unique_product_list("some_other_fileset_name")


@pytest.mark.parametrize("l4_filename", [
    # standardized asdf filename for Roman Level 4 segmentation map product
    "r0034201001001001001_0001_wfi01_f087_segm.asdf",
    # file renamed by the user
    "dummy.asdf",
])
def test_find_roman_level3_products_from_asdf(MastAladin_app, write_asdf_file, l4_filename):
    l4_path = write_asdf_file(l4_filename, orig_filename=ORIG_FILENAME)
    mission_app = _mock_mission_app()

    result = MastAladin_app.find_roman_level3_products(str(l4_path), mission_app)

    assert len(result) > 0
    assert result["fileSetName"][0] == L3_FILESET_NAME


@pytest.mark.parametrize("l4_filename", [
    # standardized parquet filename for Roman Level 4 catalog product
    "r0034201001001001001_0001_wfi01_f087_cat.parquet",
    # file renamed by the user
    "dummy.parquet",
])
def test_find_roman_level3_products_from_parquet(MastAladin_app, write_parquet_file, l4_filename):
    l4_path = write_parquet_file(l4_filename, orig_filename=ORIG_FILENAME)
    mission_app = _mock_mission_app()

    result = MastAladin_app.find_roman_level3_products(str(l4_path), mission_app)

    assert len(result) > 0
    assert result["fileSetName"][0] == L3_FILESET_NAME


def test_find_roman_level3_products_invalid_mission_app(MastAladin_app, write_asdf_file):
    l4_path = write_asdf_file("dummy.asdf", orig_filename=ORIG_FILENAME)

    with pytest.raises(TypeError,
                       match="mission_app must be an instance of MastMissionsClass, not <class 'NoneType'>."):  # noqa: E501
        MastAladin_app.find_roman_level3_products(str(l4_path), None)
