from importlib.resources import open_binary

from ..exchange import read_csv


def test_read_csv():
    with open_binary("cchdo.hydro.tests.data", "btl_csv.csv") as f:
        test_data = f.read()

    read_csv(test_data)
