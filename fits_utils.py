import numpy as np
from astropy.io import fits
from astropy import constants, units, wcs

from other_utils import coord2xy, estimate_rms, trim



def Jy2K(header = None, bmaj: float = None, bmin: float = None,
         restfrq: float = None) -> float:
    """Calculate a conversion factor in the unit of K/Jy.

    Args:
        header (optional): astropy.io.fits.open('a.fits')[0].header
                           Defaults to None.
        bmaj (float, optional): beam major axis in degree. Defaults to None.
        bmin (float, optional): beam minor axis in degree. Defaults to None.
        freq (float, optional): rest frequency in Hz. Defaults to None.

    Returns:
        float: the conversion factor in the unit of K/Jy.
    """
    freq = None
    if not header is None:
        bmaj, bmin = header['BMAJ'], header['BMIN']
        if 'RESTFREQ' in header.keys(): freq = header['RESTFREQ']
        if 'RESTFRQ' in header.keys(): freq = header['RESTFRQ']
    if not (restfrq is None): freq = restfrq
    if freq is None:
        print('Please input restfrq.')
        return -1
    omega = bmaj * bmin * np.radians(1)**2 * np.pi / 4. * np.log(2.)
    lam = constants.c.to('m/s').value / freq
    a = units.Jy.to('J*s**(-1)*m**(-2)*Hz**(-1)') \
        * lam**2 / 2. / constants.k_B.to('J/K').value / omega
    return a


class FitsData:
    """For practical treatment of data in a FITS file."""
    def __init__(self, fitsimage: str):
        self.fitsimage = fitsimage

    def gen_hdu(self):
        self.hdu = fits.open(self.fitsimage)[0]
        
    def gen_header(self) -> None:
        if not hasattr(self, 'hdu'):
            self.gen_hdu()
        self.header = self.hdu.header

    def get_header(self, key: str = None) -> int or float:
        if not hasattr(self, 'header'):
            self.gen_header()
        if key is None:
            return self.header
        if key in self.header:
            return self.header[key]
        print(f'{key} is not in the header.')
        return None

    def gen_beam(self, dist: float = 1.) -> None:
        bmaj = self.get_header('BMAJ')
        bmin = self.get_header('BMIN')
        bpa = self.get_header('BPA')
        bmaj = 0 if bmaj is None else bmaj * 3600.
        bmin = 0 if bmin is None else bmin * 3600.
        bpa = 0 if bpa is None else bpa
        self.bmaj, self.bmin, self.bpa = bmaj * dist, bmin * dist, bpa

    def get_beam(self, dist: float = 1.) -> list:
        if not hasattr(self, 'bmaj'):
            self.gen_beam(dist)
        return [self.bmaj, self.bmin, self.bpa]

    def gen_data(self, Tb: bool = False, log: bool = False,
                 drop: bool = True, restfrq: float = None) -> None:
        self.data = None
        if not hasattr(self, 'hdu'):
            self.gen_hdu()
        h, d = self.hdu.header, self.hdu.data
        if drop == True: d = np.squeeze(d)
        if Tb == True: d *= Jy2K(header=h, restfrq=restfrq)
        if log == True: d = np.log10(d.clip(np.min(d[d > 0]), None))
        self.data = d
        
    def get_data(self, Tb: bool = False, log: bool = False,
                 drop: bool = True, restfrq: float = None) -> list:
        if not hasattr(self, 'data'):
            self.gen_data(Tb, log, drop, restfrq)
        return self.data

    def gen_grid(self, center: str = None, rmax: float = 1e10,
                 xoff: float = 0., yoff: float = 0., dist: float = 1.,
                 restfrq: float = None, vsys: float = 0.,
                 vmin: float = -1e10, vmax: float = 1e10) -> None:
        if not hasattr(self, 'header'):
            self.gen_header()
        h = self.header
        if not center is None:
            cx, cy = coord2xy(center)
        else:
            cx, cy = h['CRVAL1'], h['CRVAL2']
        self.x, self.y, self.v = None, None, None
        self.dx, self.dy, self.dv = None, None, None
        if h['NAXIS'] > 0:
            if h['NAXIS1'] > 1:
                s = np.arange(h['NAXIS1'])
                s = (s-h['CRPIX1']+1) * h['CDELT1'] + h['CRVAL1'] - cx
                s *= 3600. * dist
                self.x, self.dx = s, s[1] - s[0]
        if h['NAXIS'] > 1:
            if h['NAXIS2'] > 1:
                s = np.arange(h['NAXIS2'])
                s = (s-h['CRPIX2']+1) * h['CDELT2'] + h['CRVAL2'] - cy
                s *= 3600. * dist
                self.y, self.dy = s, s[1] - s[0]
        if h['NAXIS'] > 2:
            if h['NAXIS3'] > 1:
                s = np.arange(h['NAXIS3'])
                s = (s-h['CRPIX3']+1) * h['CDELT3'] + h['CRVAL3']
                freq = None
                if 'RESTFREQ' in h.keys(): freq = h['RESTFREQ']
                if 'RESTFRQ' in h.keys(): freq = h['RESTFRQ']
                if not restfrq is None: freq = restfrq
                if not (freq is None):
                    s = (freq-s) / freq
                    s = s * constants.c.to('km*s**(-1)').value - vsys
                    self.v, self.dv = s, s[1] - s[0]
                else:
                    print('Please input restfrq.')
        if not hasattr(self, 'data'): self.data = None
        self.x, self.y, self.v, self.data \
            = trim(self.x, self.y,
                   [xoff - rmax, xoff + rmax],
                   [yoff - rmax, yoff + rmax],
                   self.v, [vmin, vmax], self.data)
                    
    def get_grid(self, center: str = None, rmax: float = 1e10,
                 xoff: float = 0., yoff: float = 0., dist: float = 1.,
                 restfrq: float = None, vsys: float = 0.,
                 vmin: float = -1e10, vmax: float = 1e10) -> None:
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            self.gen_grid(center, rmax, xoff, yoff, dist,
                          restfrq, vsys, vmin, vmax)
        if hasattr(self, 'v'):
            return [self.x, self.y, self.v]
        else:
            return [self.x, self.y, None]


def fits2data(fitsimage: str, Tb: bool = False, log: bool = False,
              dist: float = 1., method: str = None,
              restfrq: float = None, **kwargs) -> list:
    fd = FitsData(fitsimage)
    fd.gen_data(Tb=Tb, log=log, drop=True, restfrq=restfrq)
    rms = None if method is None else estimate_rms(fd.data, method)
    grid = fd.get_grid(**kwargs)
    beam = fd.get_beam(dist=dist)
    bunit = fd.get_header('BUNIT')
    return [fd.data, grid, beam, bunit, rms]

    
def data2fits(d: list = None, h: dict = {}, crpix: int = None,
              crval: int = None, cdelt: float = None, ctype: str = None,
              fitsname: str = 'test', foraplpy: bool = False):
    ctype0 = ["RA---AIR", "DEC--AIR", "VELOCITY"]
    if foraplpy:
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [h['CRPIX1'], h['CRPIX2']]
        w.wcs.cdelt = [h['CDELT1'], h['CDELT2']]
        w.wcs.crval = [h['CRVAL1'], h['CRVAL2']]
        w.wcs.ctype = ctype0[:2]
        header = w.to_header()
        hdu = fits.PrimaryHDU(d, header=header)
        if 'BUNIT' in h.keys(): hdu.header['BUNIT'] = h['BUNIT']
        hdu.header['BMAJ'] = h['BMAJ']
        hdu.header['BMIN'] = h['BMIN']
        hdu.header['BPA'] = h['BPA']
        if 'RESTFREQ' in h.keys(): hdu.header['RESTFRQ'] = h['RESTFREQ']
        if 'RESTFRQ' in h.keys(): hdu.header['RESTFRQ'] = h['RESTFRQ']
    else:
        naxis = len(np.shape(d))
        w = wcs.WCS(naxis=naxis)
        w.wcs.crpix = [0] * naxis if crpix is None else crpix
        w.wcs.crval = [0] * naxis if crval is None else crval
        w.wcs.cdelt = [1] * naxis if cdelt is None else cdelt
        w.wcs.ctype = ctype0[:naxis] if ctype is None else ctype
        header = w.to_header()
        hdu = fits.PrimaryHDU(d, header=header)
        for k in h.keys():
            if not ('COMMENT' in k or 'HISTORY' in k):
                hdu.header[k]=h[k]
    hdu = fits.HDUList([hdu])
    hdu.writeto(fitsname.replace('.fits', '') + '.fits', overwrite=True)

'''
import subprocess, shlex

def hdu4aplpy(fitslist: list, data: list) -> list:
    if type(fitslist) is str: fitslist = [fitslist] 
    c = [None] * len(fitslist)
    for i, (n, d) in enumerate(zip(fitslist, data)):
        h = fits.open(n)[0].header
        data2fits(d=d, h=h, foraplpy=True, fitsname='hdu4aplpy')
        c[i] = fits.open('hdu4aplpy.fits')[0]
        subprocess.run(shlex.split('rm hdu4aplpy.fits'))
    return c
'''
