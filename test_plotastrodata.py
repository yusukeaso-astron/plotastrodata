from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata import const_utils
from plotastrodata.coord_utils import coord2xy, xy2coord, rel2abs, abs2rel
from plotastrodata.ext_utils import BnuT, JnuT
from plotastrodata.fft_utils import fftcentering2, ifftcentering2
from plotastrodata.fits_utils import FitsData, fits2data, data2fits
from plotastrodata.fitting_utils import EmceeCorner
from plotastrodata.los_utils import obs2sys, sys2obs, polarvel2losvel
from plotastrodata.matrix_utils import Mfac, Mrot, Mrot3d, dot2d
from plotastrodata.other_utils import estimate_rms, trim, gaussian2d
from plotastrodata.plot_utils import (set_rcparams, PlotAstroData,
                                      plotprofile, plotslice, plot3d)


def test_astrodata():
    d = AstroData()
    assert type(d.todict()) is dict
