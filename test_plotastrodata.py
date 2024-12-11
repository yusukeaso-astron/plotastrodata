from plotastrodata.analysis_utils import *
from plotastrodata.fft_utils import *
from plotastrodata.fits_utils import *
from plotastrodata.fitting_utils import *
from plotastrodata.los_utils import *
from plotastrodata.other_utils import *
from plotastrodata.plot_utils import *


def test_astrodata():
    d = AstroData()
    assert type(d.todict()) is dict
