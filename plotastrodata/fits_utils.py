import numpy as np
from astropy.io import fits
from astropy import units, wcs

from plotastrodata.other_utils import coord2xy, xy2coord, estimate_rms, trim
from plotastrodata import const_utils as cu


def Jy2K(header=None, bmaj: float | None = None, bmin: float | None = None,
         restfreq: float | None = None) -> float:
    """Calculate a conversion factor in the unit of K/Jy.

    Args:
        header (optional): astropy.io.fits.open('a.fits')[0].header. Defaults to None.
        bmaj (float, optional): beam major axis in degree. Defaults to None.
        bmin (float, optional): beam minor axis in degree. Defaults to None.
        freq (float, optional): rest frequency in Hz. Defaults to None.

    Returns:
        float: the conversion factor in the unit of K/Jy.
    """
    freq = None
    if header is not None:
        if 'BMAJ' in header and 'BMIN' in header:
            bmaj, bmin = header['BMAJ'] * 3600, header['BMIN'] * 3600
        else:
            print('Use CDELT1^2 for Tb conversion.')
            bmaj = bmin = header['CDELT1'] * np.sqrt(4*np.log(2)/np.pi) * 3600
            if header['CUNIT1'] == 'arcsec':
                bmaj, bmin = bmaj / 3600, bmin / 3600
        if 'RESTFREQ' in header:
            freq = header['RESTFREQ']
        if 'RESTFRQ' in header:
            freq = header['RESTFRQ']
    if restfreq is not None:
        freq = restfreq
    if freq is None:
        print('Please input restfreq.')
        return 1
    omega = bmaj * bmin * units.arcsec**2 * np.pi / 4. / np.log(2.)
    equiv = units.brightness_temperature(freq * units.Hz, beam_area=omega)
    T = (1 * units.Jy / units.beam).to(units.K, equivalencies=equiv)
    return T.value


class FitsData:
    """For practical treatment of data in a FITS file.

    Args:
        fitsimage (str): Input FITS file name.
    """
    def __init__(self, fitsimage: str):
        self.fitsimage = fitsimage

    def gen_hdu(self) -> None:
        """Generate self.hdu, which is astropy,io.fits.open()[0].
        """
        hdu = fits.open(self.fitsimage)
        self.hdu = hdu[0]
        if 'BEAMS' in hdu:
            print('Beam table found in HDU list. Use median beam.')
            b = hdu['BEAMS'].data
            area = b['BMAJ'] * b['BMIN']  # arcsec^2?
            imed = np.nanargmin(np.abs(area - np.nanmedian(area)))
            self.hdubeam = b['BMAJ'][imed], b['BMIN'][imed], b['BPA'][imed]

    def gen_header(self) -> None:
        """Generate self.header., which is astropy.io.fits.open()[0].header.
        """
        if not hasattr(self, 'hdu'):
            self.gen_hdu()
        self.header = self.hdu.header

    def get_header(self, key: str | None = None) -> dict | float:
        """Output the entire header or a value when a key is given.

        Args:
            key (str, optional): Key name of the FITS header. Defaults to None.

        Returns:
            dict or float: The entire header or a value.
        """
        if not hasattr(self, 'header'):
            self.gen_header()
        if key is None:
            return self.header
        if key in self.header:
            return self.header[key]
        print(f'{key} is not in the header.')
        return None

    def gen_beam(self, dist: float = 1.) -> None:
        """Generate sef.bmaj, self.bmin, self.bpa from header['BMAJ'], etc.

        Args:
            dist (float, optional): bmaj and bmin are multiplied by dist. Defaults to 1..
        """
        if hasattr(self, 'hdubeam'):
            bmaj, bmin, bpa = self.hdubeam
            bmaj = bmaj * dist
            bmin = bmin * dist
        else:
            bmaj = self.get_header('BMAJ')
            bmin = self.get_header('BMIN')
            bpa = self.get_header('BPA')
            if bmaj is not None:
                bmaj = bmaj * 3600 * dist
            if bmin is not None:
                bmin = bmin * 3600 * dist
        self.bmaj, self.bmin, self.bpa = bmaj, bmin, bpa

    def get_beam(self, dist: float = 1.) -> np.ndarray:
        """Output the beam array of [bmaj, bmin, bpa].

        Args:
            dist (float, optional): bmaj and bmin are multiplied by dist. Defaults to 1..

        Returns:
            np.ndarray: [bmaj, bmin, bpa].
        """
        if not hasattr(self, 'bmaj'):
            self.gen_beam(dist)
        return np.array([self.bmaj, self.bmin, self.bpa])

    def get_center(self) -> str:
        """Output the central coordinates as text.

        Returns:
            str: The central coordinates.
        """
        ra_deg = self.get_header('CRVAL1')
        dec_deg = self.get_header('CRVAL2')
        radesys = self.get_header('RADESYS')
        a = xy2coord([ra_deg, dec_deg])
        if radesys is not None:
            a = f'{radesys} {a}'
        return a

    def gen_data(self, Tb: bool = False, log: bool = False,
                 drop: bool = True, restfreq: float = None) -> None:
        """Generate data, which may be brightness temperature.

        Args:
            Tb (bool, optional): True means the data are brightness temperatures. Defaults to False.
            log (bool, optional): True means the data are after taking the logarithm to the base 10. Defaults to False.
            drop (bool, optional): True means the data are after using np.squeeze. Defaults to True.
            restfreq (float, optional): Rest frequency for calculating the brightness temperature. Defaults to None.
        """
        self.data = None
        if not hasattr(self, 'hdu'):
            self.gen_hdu()
        h, d = self.hdu.header, self.hdu.data
        if drop:
            d = np.squeeze(d)
        if Tb:
            d *= Jy2K(header=h, restfreq=restfreq)
        if log:
            d = np.log10(d.clip(np.min(d[d > 0]), None))
        self.data = d

    def get_data(self, **kwargs) -> np.ndarray:
        """Output data. This method can take the arguments of gen_data().

        Returns:
            np.ndarray: data in the format of np.ndarray.
        """
        if not hasattr(self, 'data'):
            self.gen_data(**kwargs)
        return self.data

    def gen_grid(self, center: str | None = None, dist: float = 1.,
                 restfreq: float | None = None, vsys: float = 0.,
                 pv: bool = False) -> None:
        """Generate grids relative to the center and vsys.

        Args:
            center (str, optional): Center for the spatial grids. Defaults to None.
            dist (float, optional): The spatial grids are multiplied by dist. Defaults to 1..
            restfreq (float, optional): Rest frequency for converting the frequencies to velocities. Defaults to None.
            vsys (float, optional): The velocity is relative to vsys. Defaults to 0..
            pv (bool, optional): Mode for position-velocity diagram. Defaults to False.
        """
        h = self.get_header()
        # spatial center
        if center is not None:
            c0 = xy2coord([h['CRVAL1'], h['CRVAL2']])
            if 'RADESYS' in h:
                radesys = h['RADESYS']
                c0 = f'{radesys}  {c0}'
            cx, cy = coord2xy(center, c0)
        else:
            cx, cy = 0, 0
        # rest frequency
        if restfreq is None:
            if 'RESTFRQ' in h:
                restfreq = h['RESTFRQ']
            if 'RESTFREQ' in h:
                restfreq = h['RESTFREQ']
        self.x, self.y, self.v = None, None, None
        self.dx, self.dy, self.dv = None, None, None

        def get_list(i: int, crval=False) -> np.ndarray:
            s = np.arange(h[f'NAXIS{i:d}'])
            s = (s - h[f'CRPIX{i:d}'] + 1) * h[f'CDELT{i:d}']
            if crval:
                s = s + h[f'CRVAL{i:d}']
            return s

        def gen_x(s: np.ndarray) -> None:
            s = (s - cx) * dist
            if h['CUNIT1'].strip() in ['deg', 'DEG', 'degree', 'DEGREE']:
                s *= 3600.
            self.x, self.dx = s, s[1] - s[0]

        def gen_y(s: np.ndarray) -> None:
            s = (s - cy) * dist
            if h['CUNIT2'].strip() in ['deg', 'DEG', 'degree', 'DEGREE']:
                s *= 3600.
            self.y, self.dy = s, s[1] - s[0]

        def gen_v(s: np.ndarray) -> None:
            if restfreq is None:
                freq = np.mean(s)
                print('restfreq is assumed to be the center.')
            else:
                freq = restfreq

            vaxis = '2' if pv else '3'
            key = f'CUNIT{vaxis}'
            cunitv = h[key]
            match cunitv:
                case 'Hz':
                    if freq == 0:
                        print('v is frequency because restfreq=0.')
                    else:
                        s = (1 - s / freq) * cu.c_kms - vsys
                case 'HZ':
                    if freq == 0:
                        print('v is frequency because restfreq=0.')
                    else:
                        s = (1 - s / freq) * cu.c_kms - vsys
                case 'm/s':
                    print(f'{key}=\'m/s\' found.')
                    s = s * 1e-3 - vsys
                case 'M/S':
                    print(f'{key}=\'M/S\' found.')
                    s = s * 1e-3 - vsys
                case 'km/s':
                    print(f'{key}=\'km/s\' found.')
                    s = s - vsys
                case 'KM/S':
                    print(f'{key}=\'KM/S\' found.')
                    s = s - vsys
                case _:
                    print(f'Unknown CUNIT3 {cunitv} found.'
                          + ' v is read as is.')
                    s = s - vsys
            
            self.v, self.dv = s, s[1] - s[0]

        if h['NAXIS'] > 0 and h['NAXIS1'] > 1:
            gen_x(get_list(1))
        if h['NAXIS'] > 1 and h['NAXIS2'] > 1:
            gen_v(get_list(2, True)) if pv else gen_y(get_list(2))
        if h['NAXIS'] > 2 and h['NAXIS3'] > 1:
            gen_v(get_list(3, True))

    def get_grid(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Output the grids, [x, y, v]. This method can take the arguments of gen_grid().

        Returns:
            tuple: (x, y, v).
        """
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            self.gen_grid(**kwargs)
        return self.x, self.y, self.v

    def trim(self, rmax: float = 1e10, xoff: float = 0., yoff: float = 0.,
             vmin: float = -1e10, vmax: float = 1e10,
             pv: bool = False) -> None:
        """Trim the data and grids. The data range will be from xoff - rmax, yoff - rmax, vmin to xoff + rmax, yoff + rmax, vmax.

        Args:
            rmax (float, optional): Defaults to 1e10.
            xoff (float, optional): Defaults to 0..
            yoff (float, optional): Defaults to 0..
            vmin (float, optional): Defaults to -1e10.
            vmax (float, optional): Defaults to 1e10.
            pv (bool, optional): Mode for position-velocity diagram. Defaults to False.
        """
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
              dist: float = 1., sigma: str | None = None,
              restfreq: float | None = None, center: str | None = None,
              vsys: float = 0., pv: bool = False, **kwargs
              ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray],
                         tuple[float, float, float], float, float]:
    """Extract data from a fits file. kwargs are arguments of FitsData.trim().

    Args:
        fitsimage (str): Input fits name.
        Tb (bool, optional): True means ouput data are brightness temperature. Defaults to False.
        log (bool, optional): True means output data are logarhismic. Defaults to False.
        dist (float, optional): Change x and y in arcsec to au. Defaults to 1..
        sigma (str, optional): Noise level or method for measuring it. Defaults to None.
        restfreq (float, optional): Used for velocity and brightness temperature. Defaults to None.
        center (str, optional): Text coordinates. Defaults to None.
        vsys (float, optional): In the unit of km/s. Defaults to 0.
        pv (bool, optional): True means PV fits file. Defaults to False.

    Returns:
        tuple: (data, (x, y, v), (bmaj, bmin, bpa), bunit, rms)
    """
    fd = FitsData(fitsimage)
    fd.gen_data(Tb=Tb, log=log, drop=True, restfreq=restfreq)
    rms = estimate_rms(fd.data, sigma)
    fd.gen_grid(center=center, dist=dist, restfreq=restfreq, vsys=vsys, pv=pv)
    fd.trim(pv=pv, **kwargs)
    beam = fd.get_beam(dist=dist)
    bunit = fd.get_header('BUNIT')
    return fd.data, (fd.x, fd.y, fd.v), beam, bunit, rms


def data2fits(d: np.ndarray | None = None, h: dict = {},
              templatefits: str | None = None,
              fitsimage: str = 'test') -> None:
    """Make a fits file from a N-D array.

    Args:
        d (np.ndarray, optional): N-D array. Defaults to None.
        h (dict, optional): Fits header. Defaults to {}.
        templatefits (str, optional): Fits file to copy header. Defaults to None.
        fitsimage (str, optional): Output name. Defaults to 'test'.
    """
    ctype0 = ["RA---SIN", "DEC--SIN", "VELOCITY"]
    naxis = np.ndim(d)
    w = wcs.WCS(naxis=naxis)
    _h = {} if templatefits is None else FitsData(templatefits).get_header()
    _h.update(h)
    if _h == {}:
        w.wcs.crpix = [0] * naxis
        w.wcs.crval = [0] * naxis
        w.wcs.cdelt = [1] * naxis
        w.wcs.ctype = ctype0[:naxis]
    header = w.to_header()
    hdu = fits.PrimaryHDU(d, header=header)
    for k in _h:
        if not ('COMMENT' in k or 'HISTORY' in k) and _h[k] is not None:
            hdu.header[k] = _h[k]
    hdu = fits.HDUList([hdu])
    hdu.writeto(fitsimage.replace('.fits', '') + '.fits', overwrite=True)
