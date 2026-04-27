import numpy as np
from astropy import units, wcs
from astropy.io import fits

from plotastrodata import const_utils as cu
from plotastrodata.coord_utils import coord2xy, xy2coord
from plotastrodata.matrix_utils import dot2d
from plotastrodata.noise_utils import estimate_rms
from plotastrodata.other_utils import isdeg, RGIxy, trim


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
            todiameter = np.sqrt(4 * np.log(2) / np.pi) * 3600
            bmaj = bmin = np.abs(header['CDELT1']) * todiameter
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
        return
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
        cunit1 = self.get_header('CUNIT1')
        cunit2 = self.get_header('CUNIT2')
        if not (isdeg(cunit1) and isdeg(cunit2)):
            print(f'CUNIT1=\'{cunit1}\' and CUNIT2=\'{cunit2}\'. \'center\' is ignored.')
            return None
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

    def _read_cd(self):
        h = self.header
        cdij = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
        if not np.all([k in list(h.keys()) for k in cdij]):
            self.wcsrot = False
            self.Mcd = None
            return

        self.wcsrot = True
        cd11, cd12, cd21, cd22 = [h[k] for k in cdij]
        self.Mcd = [[cd11, cd12], [cd21, cd22]]
        if cd21 == 0:
            rho_a = 0
        else:
            rho_a = np.arctan2(np.abs(cd21), np.sign(cd21) * cd11)
        if cd12 == 0:
            rho_b = 0
        else:
            rho_b = np.arctan2(np.abs(cd12), -np.sign(cd12) * cd22)
        if (drho := np.abs(np.degrees(rho_a - rho_b))) > 1.0:
            print('Angles from (CD21, CD11) and (CD12, CD22)'
                  + f' are different by {drho:.2} degrees.')
        crota2 = (rho_a + rho_b) / 2.
        sin_rho = np.sin(crota2)
        cos_rho = np.cos(crota2)
        cdelt1 = cd11 * cos_rho + cd21 * sin_rho
        cdelt2 = -cd12 * sin_rho + cd22 * cos_rho
        crota2 = np.degrees(crota2)
        h['CDELT1'] = cdelt1
        h['CDELT2'] = cdelt2
        for k in cdij:
            del h[k]
        print(f'WCS rotation was found (CROTA2 = {crota2:f} deg).')

    def _rotate_cd(self):
        h = self.header
        data = self.get_data()
        ic = len(self.x) // 2
        jc = len(self.y) // 2
        h['CRPIX1'] = ic + 1
        h['CRPIX2'] = jc + 1
        xc = self.x[ic] / self.dx
        yc = self.y[jc] / self.dy
        xc, yc = dot2d(self.Mcd, [xc, yc])
        newcenter = xy2coord([xc, yc], coordorg=self.get_center())
        xc, yc = coord2xy(coords=newcenter)
        h['CRVAL1'] = xc
        h['CRVAL2'] = yc
        self.x = self.x - self.x[ic]
        self.y = self.y - self.y[jc]
        x = self.x / (3600 if isdeg(h['CUNIT1']) else 1)
        y = self.y / (3600 if isdeg(h['CUNIT2']) else 1)
        X, Y = np.meshgrid(x, y)
        Mcdinv = np.linalg.inv(self.Mcd)
        xnew, ynew = dot2d(Mcdinv, [X, Y])
        datanew = RGIxy(self.y / self.dy, self.x / self.dx,
                        data, (ynew, xnew))
        self.data = datanew
        print('Data values were interpolated for WCS rotation.')

    def _get_genx_geny(self, center: str, dist: float):
        h = self.header
        cxy = (0, 0)
        if center is not None and not self.wcsrot:
            coordorg = xy2coord([h['CRVAL1'], h['CRVAL2']])
            if (radesys := h.get('RADESYS')) is not None:
                coordorg = f'{radesys}  {coordorg}'
            cxy = coord2xy(center, coordorg)
        slabel = ['x', 'y']

        def wrapper(i: int):
            def gen_s(s_in: np.ndarray | None) -> None:
                if h.get(f'NAXIS{i+1}') is None or s_in is None:
                    s, ds = None, None
                else:
                    s = (s_in - cxy[i]) * dist
                    if isdeg(h[f'CUNIT{i+1}']):
                        s *= 3600.
                    ds = None if len(s) == 0 else s[1] - s[0]
                setattr(self, f'{slabel[i]}', s)
                setattr(self, f'd{slabel[i]}', ds)
            return gen_s

        return wrapper(0), wrapper(1)

    def _get_genv(self, restfreq: float | None, vsys: float, pv: bool):
        h = self.header

        def gen_v(v_in: np.ndarray) -> None:
            vaxis = '2' if pv else '3'
            if h.get(f'NAXIS{vaxis}') is None or v_in is None:
                self.v, self.dv = None, None
                return

            if restfreq is None:
                freq = np.mean(v_in)
                print('restfreq is assumed to be the center.')
            else:
                freq = restfreq
            v = v_in + h[f'CRVAL{vaxis}']
            key = f'CUNIT{vaxis}'
            cunitv = h[key].strip()
            match cunitv:
                case 'Hz' | 'HZ':
                    if freq == 0:
                        print('v is read as is, because restfreq=0.')
                    else:
                        v = (1 - v / freq) * cu.c_kms - vsys
                case 'm/s' | 'M/S':
                    print(f'{key}={cunitv} found.')
                    v = v * 1e-3 - vsys
                case 'km/s' | 'KM/S':
                    print(f'{key}={cunitv} found.')
                    v = v - vsys
                case _:
                    print(f'Unknown CUNIT3 {cunitv} found.'
                          + ' v is read as is.')
            dv = None if len(v) == 0 else v[1] - v[0]
            self.v, self.dv = v, dv

        return gen_v

    def _get_array(self, i: int) -> np.ndarray:
        h = self.header
        n = h.get(f'NAXIS{i:d}')
        if n is None:
            return None

        s = (np.arange(n) - h[f'CRPIX{i:d}'] + 1) * h[f'CDELT{i:d}']
        return s

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
        # WCS rotation (Calabretta & Greisen 2002, Astronomy & Astrophysics, 395, 1077)
        self._read_cd()
        gen_x, gen_y = self._get_genx_geny(center, dist)
        restfreq = restfreq or h.get('RESTFRQ') or h.get('RESTFREQ')
        gen_v = self._get_genv(restfreq, vsys, pv)
        if pv:
            gen_x(self._get_array(1))
            gen_v(self._get_array(2))
            self.y, self.dy = None, None
        else:
            gen_x(self._get_array(1))
            gen_y(self._get_array(2))
            gen_v(self._get_array(3))
        if self.wcsrot:
            self._rotate_cd()

    def get_grid(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Output the grids, [x, y, v]. This method can take the arguments of gen_grid().

        Returns:
            tuple: (x, y, v).
        """
        if not np.all([hasattr(self, s) for s in ['x', 'y', 'v']]):
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


def data2fits(d: np.ndarray, h: dict = {},
              templatefits: str | None = None,
              fitsimage: str = 'test') -> None:
    """Make a fits file from a N-D array.

    Args:
        d (np.ndarray): N-D array.
        h (dict, optional): Additional FITS header. Defaults to {}.
        templatefits (str, optional): FITS file whose header is used as a temperate. Defaults to None.
        fitsimage (str, optional): Output filename, with or without '.fits'. Defaults to 'test'.
    """
    _h = {} if templatefits is None else FitsData(templatefits).get_header()
    _h.update(h)
    naxis = np.ndim(d)
    w = wcs.WCS(naxis=naxis)
    if _h == {}:
        w.wcs.crpix = [0] * naxis
        w.wcs.crval = [0] * naxis
        w.wcs.cdelt = [1] * naxis
    defaults = {'CTYPE': ['RA---SIN', 'DEC--SIN', 'FREQ'],
                'CUNIT': ['deg', 'deg', 'Hz']}
    for k, v in defaults.items():
        for i in range(naxis):
            _h.setdefault(f'{k}{i+1:d}', v[i])
    othernames = {'deg': ['DEG', 'Deg'], 'Hz': ['HZ', 'hz']}
    for i in range(naxis):
        key = f'CUNIT{i+1}'
        old = _h[key].strip()
        for standard, NGlist in othernames.items():
            if old in NGlist:
                _h[key] = standard
                print(f'{key}={old} has been changed to {key}={standard}.')
    w.wcs.ctype = [_h[f'CTYPE{i+1}'] for i in range(naxis)]
    w.wcs.cunit = [_h[f'CUNIT{i+1}'] for i in range(naxis)]
    _h.setdefault('BUNIT', 'Jy/beam')
    if naxis >= 3:
        _h.setdefault('SPECSYS', 'LSRK')
    header = w.to_header()
    hdu = fits.PrimaryHDU(d, header=header)
    for k, v in _h.items():
        if v is not None and 'COMMENT' not in k and 'HISTORY' not in k:
            hdu.header[k] = v
    hdu = fits.HDUList([hdu])
    hdu.writeto(fitsimage.removesuffix('.fits') + '.fits', overwrite=True)
