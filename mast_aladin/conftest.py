import os
import pytest
from astropy.table import Table
from mast_aladin import MastAladin
import jdaviz
import asdf
import pyarrow as pa
import pyarrow.parquet as pq


@pytest.fixture
def MastAladin_app():
    return MastAladin()


@pytest.fixture
def imviz_helper():
    return jdaviz.gca()


@pytest.fixture
def mast_observation_table():
    """
    To reproduce the table file, run:

        from astroquery.mast.missions import MastMissions

        mast = MastMissions(mission='jwst')
        result = mast.query_object("M4", limit=5)
        result.write("mm_jwst_M4.ecsv")
    """
    path = os.path.join(
        os.path.dirname(__file__), "tests", "data", "mm_jwst_M4.ecsv"
    )
    return Table.read(path)


@pytest.fixture
def write_asdf_file(tmp_path):
    """
    Create a minimal ASDF file mimicking a Roman Level 4 segmentation map
    product, containing just enough metadata for ``find_roman_level3_products``
    to recover the original Level 3 filename.
    """
    def _write(filename, orig_filename=None):
        path = tmp_path / filename
        tree = {"roman": {"meta": {"filename": orig_filename,
                                   "wcsinfo": {"ra": 10.0, "dec": 20.0},
                                   "instrument": {"name": "WFI", "optical_element": "F087"},
                                   "observation": {"execution_plan": "EP1", "program": "P1"},
                                   "coadd_info": {"exposure_time": 100.0}}}
                }
        asdf.AsdfFile(tree).write_to(path)
        return path
    return _write


@pytest.fixture
def write_parquet_file(tmp_path):
    """
    Create a minimal parquet file mimicking a Roman Level 4 catalog
    product, containing just enough metadata for ``find_roman_level3_products``
    to recover the original Level 3 filename.
    """
    def _write(filename, orig_filename=None):
        path = tmp_path / filename
        schema = pa.schema([("ra", pa.float64()), ("dec", pa.float64())])
        schema = schema.with_metadata(
            {b"roman.meta.filename": orig_filename.encode("utf-8"),
             b"roman.meta.wcsinfo.ra": b"10.0",
             b"roman.meta.wcsinfo.dec": b"20.0",
             b"roman.meta.instrument.name": b"WFI",
             b"roman.meta.instrument.optical_element": b"F087",
             b"roman.meta.observation.execution_plan": b"EP1",
             b"roman.meta.observation.program": b"P1",
             b"roman.meta.coadd_info.exposure_time": b"100.0"}
        )
        table = pa.table({"ra": [10.0], "dec": [20.0]}, schema=schema)
        pq.write_table(table, path)
        return path
    return _write
