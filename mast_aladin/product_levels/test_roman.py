from astroquery.exceptions import InvalidQueryError
from astroquery.mast import MastMissionsClass
from astropy.table import Table
from mast_aladin.data_levels.roman import get_parent, get_child
import pytest
from unittest.mock import Mock, patch

ORIG_FILENAME_L3 = "r0034201001001001001_0001_wfi01_f087_coadd.asdf"
L3_FILESET_NAME = ORIG_FILENAME_L3.rpartition('_')[0]
ORIG_FILENAME_L4 = "r0034201001001001001_0001_wfi01_f087_segm.asdf"
L4_FILESET_NAME = ORIG_FILENAME_L4.rpartition('_')[0]


def _mock_mission_app():
    """
    Create a mocked `MastMissionsClass` instance that simulates the two-step
    `query_criteria` -> `get_unique_product_list` lookup used by
    `_extract_product_from_metadata`.
    """
    mission_app = Mock(spec=MastMissionsClass)
    both_table = Table(
        {"category": ["3", "4"],
         "fileSetName": [L3_FILESET_NAME, L4_FILESET_NAME],
         "productLevel": ["3", "4"]}
    )

    def query_criteria(**kwargs):
        # stand-in for the query object astroquery would normally build
        return kwargs

    def get_unique_product_list(query):
        # a real query (by coordinates or fileSetName) returns matching products
        # of any level sharing that fileSetName; let `_extract_product_from_metadata`
        # filter by the requested category. Note L3_FILESET_NAME == L4_FILESET_NAME
        # in this test suite, mirroring how MAST groups products by fileSetName.
        if "coordinates" in query or query.get("fileSetName") in (L3_FILESET_NAME, L4_FILESET_NAME):
            return both_table
        raise InvalidQueryError("no match")

    mission_app.query_criteria.side_effect = query_criteria
    mission_app.get_unique_product_list.side_effect = get_unique_product_list

    return mission_app


def test_mock_mission_app_query_chain():
    """
    Unit test for the `_mock_mission_app` test helper: it should return the
    Level 3 product table for a matching fileSetName or coordinates-based
    query, and raise `InvalidQueryError` for any other query.
    """
    mission_app = _mock_mission_app()

    query = mission_app.query_criteria(fileSetName=L3_FILESET_NAME, productLevel=3)
    result = mission_app.get_unique_product_list(query)
    assert len(result) == 2

    query = mission_app.query_criteria(coordinates="10d 20d", productLevel=3)
    result = mission_app.get_unique_product_list(query)
    assert len(result) == 2

    query = mission_app.query_criteria(fileSetName="nonexistent", productLevel=3)
    with pytest.raises(InvalidQueryError, match="no match"):
        mission_app.get_unique_product_list(query)


@pytest.fixture
def mock_rdd_open():
    """
    Patch the `mast_aladin.app.rdd.open` function to return a mock object
    with a `to_flat_dict` method that returns a dictionary containing
    the necessary metadata for roman module to recover
    the original Level 3/4 filename, since `roman.get_parent` and `roman.get_child`
    use `rdd.open` to read the file and extract metadata.
    """
    flat = {
           "roman.meta.filename": ORIG_FILENAME_L3,
           "roman.meta.wcsinfo.ra": 10.0,
           "roman.meta.wcsinfo.dec": 20.0,
           "roman.meta.instrument.name": "WFI",
           "roman.meta.instrument.optical_element": "F087",
           "roman.meta.observation.execution_plan": "EP1",
           "roman.meta.observation.program": "P1",
           "roman.meta.coadd_info.exposure_time": 100.0}
    with patch("mast_aladin.data_levels.roman.rdd.open") as m:
        m.return_value.to_flat_dict.return_value = flat
        yield m


def test_find_roman_level3_products_from_asdf(write_asdf_file):
    """
    Test that `find_roman_level3_products` can extract the original Level 3 filename
    from a Level 4 ASDF file and return the corresponding Level 3 product table.
    It won't open the file, so rdd.open doesn't need to be called.
    """
    l4_path = write_asdf_file("r0034201001001001001_0001_wfi01_f087_segm.asdf",
                              orig_filename=ORIG_FILENAME_L4)
    mission_app = _mock_mission_app()

    result = get_parent(str(l4_path), mission_app, to_level=3)

    assert len(result) > 0
    assert result["fileSetName"][0] == L3_FILESET_NAME


def test_find_roman_level3_products_from_renamed_asdf(write_asdf_file, mock_rdd_open):
    """
    In practice, a renamed ASDF file will be opened by rdd.open, which will work for
    real Roman files because they are valid inputs for roman_datamodels.
    This test requires a mocked rdd.open to simulate the behavior of opening a renamed ASDF file
    since the pytest fixture `write_asdf_file` creates a minimal ASDF file that is not a
    valid Roman file and cannot be opened by rdd.open.
    """
    l4_path = write_asdf_file("dummy.asdf",
                              orig_filename=ORIG_FILENAME_L4)
    mission_app = _mock_mission_app()

    result = get_parent(str(l4_path), mission_app, to_level=3)

    mock_rdd_open.assert_called_once()
    assert len(result) > 0
    assert result["fileSetName"][0] == L3_FILESET_NAME


@pytest.mark.parametrize("l4_filename", [
    # standardized parquet filename for Roman Level 4 catalog product
    "r0034201001001001001_0001_wfi01_f087_cat.parquet",
    # file renamed by the user
    "dummy.parquet",
])
def test_find_roman_level3_products_from_parquet(write_parquet_file, l4_filename):
    """
    Test that `find_roman_level3_products` can find L3 products given any L4 parquet file.
    """
    l4_path = write_parquet_file(l4_filename, orig_filename=ORIG_FILENAME_L4)
    mission_app = _mock_mission_app()

    result = get_parent(str(l4_path), mission_app, to_level=3)

    assert len(result) > 0
    assert result["fileSetName"][0] == L3_FILESET_NAME


def test_find_roman_level3_products_invalid_file(tmp_path):
    """
    A file that isn't ASDF or parquet, and whose name doesn't follow the L4
    naming convention, should raise a ValueError once all fallbacks fail.
    """
    l4_path = tmp_path / "some_random_file.fits"
    l4_path.touch()
    mission_app = _mock_mission_app()

    with pytest.raises(ValueError,
                       match="Could not extract original filename"):
        get_parent(str(l4_path), mission_app, to_level=3)


def test_find_roman_level3_products_invalid_mission_app(write_asdf_file):
    l4_path = write_asdf_file("dummy.asdf", orig_filename=ORIG_FILENAME_L4)

    with pytest.raises(TypeError,
                       match="mission_app must be an instance of MastMissionsClass, not <class 'NoneType'>."):  # noqa: E501
        get_parent(str(l4_path), None, to_level=3)


def test_find_roman_level4_products_from_asdf(write_asdf_file):
    """
    Test that `find_roman_level4_products` can extract the original Level 4 filename
    from a Level 4 ASDF file and return the corresponding Level 4 product table.
    It won't open the file, so rdd.open doesn't need to be called.
    """
    l3_path = write_asdf_file("r0034201001001001001_0001_wfi01_f087_coadd.asdf",
                              orig_filename=ORIG_FILENAME_L3)
    mission_app = _mock_mission_app()

    result = get_child(str(l3_path), mission_app, to_level=4)

    assert len(result) > 0
    assert result["fileSetName"][0] == L4_FILESET_NAME


def test_find_roman_level4_products_from_renamed_asdf(write_asdf_file, mock_rdd_open):
    """
    In practice, a renamed ASDF file will be opened by rdd.open, which will work for
    real Roman files because they are valid inputs for roman_datamodels.
    This test requires a mocked rdd.open to simulate the behavior of opening a renamed ASDF file
    since the pytest fixture `write_asdf_file` creates a minimal ASDF file that is not a
    valid Roman file and cannot be opened by rdd.open.
    """
    l3_path = write_asdf_file("dummy.asdf",
                              orig_filename=ORIG_FILENAME_L3)
    mission_app = _mock_mission_app()

    result = get_child(str(l3_path), mission_app, to_level=4)

    mock_rdd_open.assert_called_once()
    assert len(result) > 0
    assert result["fileSetName"][0] == L4_FILESET_NAME


@pytest.mark.parametrize("l3_filename", [
    # standardized json filename for Roman Level 3 association product
    "r0034201001001001001_0001_wfi01_f087_asn.json",
    # file renamed by the user
    "dummy.json",
])
def test_find_roman_level4_products_from_json(write_json_file, l3_filename):
    """
    Test that ``roman.get_child`` can find L4 products given any L3 json file.
    """
    l3_path = write_json_file(l3_filename,
                              orig_filename=L3_FILESET_NAME + '_asn.json')
    mission_app = _mock_mission_app()

    result = get_child(str(l3_path), mission_app, to_level=4)

    assert len(result) > 0
    assert result["fileSetName"][0] == L4_FILESET_NAME


def test_find_roman_level4_products_invalid_file(tmp_path):
    """
    A file that isn't ASDF or parquet, and whose name doesn't follow the L4
    naming convention, should raise a ValueError once all fallbacks fail.
    """
    l3_path = tmp_path / "some_random_file.fits"
    l3_path.touch()
    mission_app = _mock_mission_app()

    with pytest.raises(ValueError,
                       match="Could not extract original filename"):
        get_child(str(l3_path), mission_app, to_level=4)


def test_find_roman_level4_products_invalid_mission_app(write_asdf_file):
    l3_path = write_asdf_file("dummy.asdf", orig_filename=ORIG_FILENAME_L3)

    with pytest.raises(TypeError,
                       match="mission_app must be an instance of MastMissionsClass, not <class 'NoneType'>."):  # noqa: E501
        get_child(str(l3_path), None, to_level=4)
