from plotastrodata.analysis_utils import AstroData


def test_astrodata():
    d = AstroData()
    assert type(d.todict()) is dict
