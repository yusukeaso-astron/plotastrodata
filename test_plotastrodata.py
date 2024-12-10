from plotastrodata.analysis_utils import AstroData


def test_astrodata():
    assert type(AstroData.todict()) is dict
