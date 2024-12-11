from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata.fft_utils import fftcentering2, ifftcentering2
from plotastrodata.fits_utils import FitsData, fits2data, data2fits
from plotastrodata.fitting_utils import PTEmceeCorner
from plotastrodata.los_utils import obs2sys, sys2obs, polarvel2losvel
from plotastrodata.other_utils import (coord2xy, xy2coord, rel2abs, abs2rel,
                                       estimate_rms, trim, BnuT, JnuT, gaussian2d)
from plotastrodata.plot_utils import (set_rcparams, PlotAstroData,
                                      plotprofile, plotslice, plot3d)


def test_astrodata():
    d = AstroData()
    assert type(d.todict()) is dict
