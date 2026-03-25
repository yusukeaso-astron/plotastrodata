from plotastrodata import const_utils as cu
from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata.coord_utils import (abs2rel, coord2xy,
                                       rel2abs, xy2coord)
from plotastrodata.ext_utils import BnuT, JnuT
from plotastrodata.fft_utils import fftcentering2, ifftcentering2
from plotastrodata.fits_utils import data2fits, fits2data, FitsData
from plotastrodata.fitting_utils import EmceeCorner
from plotastrodata.los_utils import obs2sys, polarvel2losvel, sys2obs
from plotastrodata.matrix_utils import dot2d, Mfac, Mrot, Mrot3d
from plotastrodata.noise_utils import estimate_rms
from plotastrodata.other_utils import trim, gaussian2d
from plotastrodata.plot_utils import (plot3d, PlotAstroData,
                                      plotprofile, plotslice,
                                      set_rcparams)


def test_import():
    a = [AstroData, AstroFrame,
         abs2rel, coord2xy, rel2abs, xy2coord,
         BnuT, JnuT,
         fftcentering2, ifftcentering2,
         data2fits, fits2data, FitsData,
         EmceeCorner,
         obs2sys, polarvel2losvel, sys2obs,
         dot2d, Mfac, Mrot, Mrot3d,
         estimate_rms,
         trim, gaussian2d,
         plot3d, PlotAstroData, plotprofile, plotslice, set_rcparams
         ]
    d = AstroData()
    c1 = None not in a
    c2 = type(d.todict()) is dict
    c3 = 299000 < cu.c_kms < 300000
    assert c1 and c2 and c3
