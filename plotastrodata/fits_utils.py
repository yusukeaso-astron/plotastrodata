import numpy as np
from astropy.io import fits
from astropy import constants, units, wcs

from plotastrodata.other_utils import coord2xy, xy2coord, estimate_rms, trim



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
    if header is not None:
        bmaj, bmin = header['BMAJ'], header['BMIN']
        if 'RESTFREQ' in header.keys(): freq = header['RESTFREQ']
        if 'RESTFRQ' in header.keys(): freq = header['RESTFRQ']
    if restfrq is not None: freq = restfrq
    if freq is None:
        print('Please input restfrq.')
        return 1
    omega = bmaj * bmin * units.deg**2 * np.pi / 4. / np.log(2.)
    eqv = units.brightness_temperature(freq * units.Hz, beam_area=omega)
    T = (1 * units.Jy / units.beam).to(units.K, equivalencies=eqv)
    return T.value


class FitsData:
    """For practical treatment of data in a FITS file."""
    def __init__(self, fitsimage: str):
        self.fitsimage = fitsimage

    def gen_hdu(self):
        hdu = fits.open(self.fitsimage)
        self.hdu = hdu[0]
        if 'BEAMS' in hdu:
            print('Beam table found in HDU list. Use median beam.')
            b = hdu['BEAMS'].data
            area = b['BMAJ'] * b['BMIN']  # arcsec^2?
            imed = np.nanargmin(np.abs(area - np.nanmedian(area)))
            self.hdubeam = b['BMAJ'][imed], b['BMIN'][imed], b['BPA'][imed]
        
    def gen_header(self) -> None:
        if not hasattr(self, 'hdu'):
            self.gen_hdu()
        self.header = self.hdu.header

    def get_header(self, key: str = None) -> dict or float:
        if not hasattr(self, 'header'):
            self.gen_header()
        if key is None:
            return self.header
        if key in self.header:
            return self.header[key]
        print(f'{key} is not in the header.')
        return None

    def gen_beam(self, dist: float = 1.) -> None:
        if hasattr(self, 'hdubeam'):
            bmaj, bmin, bpa = self.hdubeam
        else:
            bmaj = self.get_header('BMAJ')
            bmin = self.get_header('BMIN')
            bpa = self.get_header('BPA')
            bmaj = 0 if bmaj is None else bmaj * 3600.
            bmin = 0 if bmin is None else bmin * 3600.
            bpa = 0 if bpa is None else bpa
        self.bmaj, self.bmin, self.bpa = bmaj * dist, bmin * dist, bpa

    def get_beam(self, dist: float = 1.) -> tuple:
        if not hasattr(self, 'bmaj'):
            self.gen_beam(dist)
        return np.array([self.bmaj, self.bmin, self.bpa])

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
        
    def get_data(self, **kwargs) -> list:
        if not hasattr(self, 'data'): self.gen_data(**kwargs)
        return self.data

    def gen_grid(self, center: str = None, dist: float = 1.,
                 restfrq: float = None, vsys: float = 0.,
                 pv: bool = False) -> None:
        if not hasattr(self, 'header'): self.gen_header()
        h = self.header
        # spatial center
        if center is not None:
            c0 = xy2coord([h['CRVAL1'], h['CRVAL2']])
            cx, cy = coord2xy(coords=center, coordorg=c0)
        else:
            cx, cy = 0, 0
        # rest frequency
        if 'RESTFRQ' in h.keys():
            restfrq = h['RESTFRQ']
        elif 'RESTFREQ' in h.keys():
            restfrq = h['RESTFREQ']
        self.x, self.y, self.v = None, None, None
        self.dx, self.dy, self.dv = None, None, None
        def get_list(i: int, crval=False) -> list:
            s = np.arange(h[f'NAXIS{i:d}'])
            s = (s - h[f'CRPIX{i:d}'] + 1) * h[f'CDELT{i:d}']
            if crval: s = s + h[f'CRVAL{i:d}']
            return s
        def gen_x(s: list) -> None:
            s = (s - cx) * dist
            if h['CUNIT1'].strip() in ['deg', 'DEG', 'degree']:
                s *= 3600.
            self.x, self.dx = s, s[1] - s[0]
        def gen_y(s: list) -> None:
            s = (s - cy) * dist
            if h['CUNIT2'].strip() in ['deg', 'DEG', 'degree']:
                s *= 3600. 
            self.y, self.dy = s, s[1] - s[0]
        def gen_v(s: list) -> None:
            if restfrq is None:
                freq = np.mean(s)
                print('restfrq is assumed to be the center.')            
            else:
                freq = restfrq
            s = (freq-s) / freq
            s = s * constants.c.to('km*s**(-1)').value - vsys
            self.v, self.dv = s, s[1] - s[0]
        if h['NAXIS'] > 0:
            if h['NAXIS1'] > 1:
                gen_x(get_list(1))
        if h['NAXIS'] > 1:
            if h['NAXIS2'] > 1:
                gen_v(get_list(2, True)) if pv else gen_y(get_list(2))
        if h['NAXIS'] > 2:
            if h['NAXIS3'] > 1:
                gen_v(get_list(3, True))
                    
    def get_grid(self, **kwargs) -> tuple:
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            self.gen_grid(**kwargs)
        return [self.x, self.y, self.v]

    def trim(self, rmax: float = 1e10, xoff: float = 0., yoff: float = 0.,
             vmin: float = -1e10, vmax: float = 1e10,
             pv: bool = False) -> None:
        data = self.data if hasattr(self, 'data') else None
        x = self.x if hasattr(self, 'x') else None
        y = self.y if hasattr(self, 'y') else None
        v = self.v if hasattr(self, 'v') else None
        self.data, grid = trim(data=data, x=x, y=y, v=v,
                               xlim=[xoff - rmax, xoff + rmax],
                               ylim=[yoff - rmax, yoff + rmax],
                               vlim=[vmin, vmax], pv=pv)
        self.x, self.y, self.v = grid


def fits2data(fitsimage: str, Tb: bool = False, log: bool = False,
              dist: float = 1., sigma: str = None,
              restfrq: float = None, center: str = None,
              vsys: float = 0., pv: bool = False, **kwargs) -> tuple:
    """Extract data from a fits file.
       kwargs are arguments of FitsData.trim().

    Args:
        fitsimage (str): Input fits name.
        Tb (bool, optional):
            True means ouput data are brightness temperature.
            Defaults to False.
        log (bool, optional):
            True means output data are logarhismic. Defaults to False.
        dist (float, optional):
            Change x and y in arcsec to au. Defaults to 1..
        sigma (str, optional):
            Noise level or method for measuring it. Defaults to None.
        restfrq (float, optional):
            Used for velocity and brightness temperature. Defaults to None.
        center (str, optional): Text coordinates. Defaults to None.
        vsys (float, optional): In the unit of km/s. Defaults to 0.
        pv (bool, optional): True means PV fits file. Defaults to False.

    Returns:
        tuple: (data, (x, y, v), (bmaj, bmin, bpa), bunit, rms)
    """
    fd = FitsData(fitsimage)
    fd.gen_data(Tb=Tb, log=log, drop=True, restfrq=restfrq)
    rms = estimate_rms(fd.data, sigma)
    fd.gen_grid(center=center, dist=dist, restfrq=restfrq, vsys=vsys, pv=pv)
    fd.trim(pv=pv, **kwargs)
    beam = fd.get_beam(dist=dist)
    bunit = fd.get_header('BUNIT')
    return fd.data, (fd.x, fd.y, fd.v), beam, bunit, rms

    
def data2fits(d: list = None, h: dict = {}, crpix: list = None,
              crval: list = None, cdelt: list = None, ctype: str = None,
              fitsimage: str = 'test') -> None:
    """Make a fits file from a N-D array.

    Args:
        d (list, optional): N-D array. Defaults to None.
        h (dict, optional): Fits header. Defaults to {}.
        crpix (list, optional): Defaults to None.
        crval (list, optional): Defaults to None.
        cdelt (list, optional): Defaults to None.
        ctype (str, optional): Defaults to None.
        fitsimage (str, optional): Output name. Defaults to 'test'.
    """
    ctype0 = ["RA---SIN", "DEC--SIN", "VELOCITY"]
    naxis = np.ndim(d)
    w = wcs.WCS(naxis=naxis)
    if h == {}:
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
    hdu.writeto(fitsimage.replace('.fits', '') + '.fits', overwrite=True)
