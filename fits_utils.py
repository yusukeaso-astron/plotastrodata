import subprocess, shlex
import numpy as np
from astropy.io import fits
from astropy import constants, units, wcs

from mytools import listing, coord2xy


def Jy2K(header):
    bmaj = np.radians(header['BMAJ'])
    bmin = np.radians(header['BMIN'])
    bpa = np.radians(header['BPA'])
    omega = bmaj * bmin * np.pi / 4. * np.log(2.)
    if 'RESTFREQ' in header.keys(): freq = header['RESTFREQ']
    if 'RESTFRQ' in header.keys(): freq = header['RESTFRQ']
    lam = constants.c.to('m/s').value / freq
    a = units.Jy.to('J*s**(-1)*m**(-2)*Hz**(-1)') \
        * lam**2 / 2. / constants.k_B.to('J/K').value / omega
    return a


class MyFits:
    def __init__(self, fitsimage):
        self.fitsimage = fitsimage

    def gen_header(self):
        self.header = fits.open(self.fitsimage)[0].header

    def gen_beam(self, ang='deg'):
        self.bmaj, self.bmin, self.bpa = 0, 0, 0
        h = fits.open(self.fitsimage)[0].header
        if np.all(np.isin(['BMAJ', 'BMIN', 'BPA'], list(h.keys()))):
            u = 1
            if ang == 'arcsec': u = 3600.
            elif ang in ['rad', 'radian']: u = np.radians(1.)
            self.bmaj = h['BMAJ'] * u
            self.bmin = h['BMIN'] * u
            self.bpa = h['BPA']

    def gen_data(self, Tb=False, log=False, drop=False):
        self.data = []
        c = fits.open(self.fitsimage)[0]
        h = c.header
        d = c.data
        if drop == True: d = np.squeeze(d)
        if Tb == True: d *= Jy2K(h)
        if log == True: d = np.log10(d.clip(np.min(d[d > 0]), None))
        self.data = d

    def gen_grid(self, axes=(0, 1, 2), ang='deg', vel='km/s',
                 restfrq=0, center=''):
        h = fits.open(self.fitsimage)[0].header
        self.x, self.y, self.v = [], [], []
        self.dx, self.dy, self.dv = [], [], []
        g = []
        for i in [0, 1]:
            if i in axes:
                i = str(i + 1)
                s = np.arange(h['NAXIS' + i])
                if ang in ['abspix', 'abspixel']: g.append(s)
                s = (s - h['CRPIX' + i] + 1) * h['CDELT' + i] + h['CRVAL' + i]
                if ang in ['deg', 'degree', 'absdeg', 'absdegree']: g.append(s)
                if center != '':
                    cx, cy = coord2xy(center)
                    cntr = cx if i == '1' else cy
                else:
                    cntr = h['CRVAL' + i]
                s = s - cntr
                if ang in ['reldeg', 'reldegree']: g.append(s)
                if ang in ['pix', 'pixel', 'relpix', 'relpixel']:
                    g.append(np.round(s / h['CDELT' + i]).astype(np.int64))
                if ang in ['arcsec']: g.append(s * 3600.)
                if i == '1':
                    self.x = g[0]
                    self.dx = g[0][1] - g[0][0]
                else:
                    self.y = g[1]
                    self.dy = g[1][1] - g[1][0]
        if 2 in axes and h['NAXIS'] > 2:
            if h['NAXIS3'] > 1:
                s = np.arange(h['NAXIS3'])
                if vel in ['abspix', 'abspixel']: g.append(s)
                s = (s - h['CRPIX3'] + 1) * h['CDELT3'] + h['CRVAL3']
                if vel in ['Hz', 'absHz']: g.append(s)
                freq = 0
                if 'RESTFREQ' in h.keys(): freq = h['RESTFREQ']
                if 'RESTFRQ' in h.keys(): freq = h['RESTFRQ']
                if restfrq > 0: freq = restfrq
                s = s - freq
                if vel in ['relHz']: g.append(s)
                if vel in ['pix', 'pixel', 'relpix', 'relpixel']:
                    g.append(np.round(s / h['CDELT3']).astype(np.int64))
                if freq > 0:
                    s = -s / freq * constants.c.to('m/s').value
                    if vel in ['m/s']: g.append(s)
                    s = s / 1e3
                    if vel in ['km/s']: g.append(s)
                self.v = g[2]
                self.dv = g[2][1] - g[2][0]


def fits2data(fitsimage, Tb, center=''):
    fitsimage = listing(fitsimage)
    d, x, b = [], [], []
    for f in fitsimage:
        a = MyFits(f)
        a.gen_data(Tb=Tb, drop=True)
        d.append(a.data)
        a.gen_grid(ang='arcsec', center=center)
        x.append([a.x, a.y, a.v])
        a.gen_beam(ang='arcsec')
        b.append([a.bmaj, a.bmin, a.bpa])
    return [d, x, b]


def read_bunit(fitsimage='', bunit=''):
    fitsimage, bunit = listing(fitsimage, bunit)
    a = []
    for f, b in zip(fitsimage, bunit):
        h = fits.open(f)[0].header
        if 'BUNIT' in h.keys() and b == '': b = h['BUNIT']
        a.append(b)         
    return a


def data2fits(d=None, h={}, crpix=None, crval=None, cdelt=None, ctype=None,
              fitsname='test', foraplpy=False):
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
        for k in h.keys(): hdu.header[k]=h[k]
    hdu = fits.HDUList([hdu])
    hdu.writeto(fitsname.replace('.fits', '') + '.fits', overwrite=True)


def hdu4aplpy(fitslist, data):
    fitslist = listing(fitslist)
    c = []
    for n, d in zip(fitslist, data):
        h = fits.open(n)[0].header
        data2fits(d=d, h=h, foraplpy=True, fitsname='0')
        c.append(fits.open('0.fits')[0])
        subprocess.run(shlex.split('rm 0.fits'))
    return c

