import pandas as pd
import pytest

from evasao import data, predict


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return data.load_raw()


@pytest.fixture(scope="session")
def model():
    return predict.load_model()


@pytest.fixture(scope="session")
def metadata() -> dict:
    return predict.load_metadata()
