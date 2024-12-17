from astropy import constants
from astropy import units

# astropy.units in the SI unit system
au = units.au.to('m')
pc = units.pc.to('m')
arcsec = units.arcsec.to('radian')
Jy = units.Jy.to('J*s**(-1)*m**(-2)*Hz**(-1)')
deg = units.deg.to('radian')
year = units.year.to('s')
gauss = units.gauss.to('T')
cm = units.cm.to('m')
km = units.km.to('m')
gram = units.gram.to('kg')
debye = units.debye.to('A*s*m')
erg = units.erg.to('J')

# astropy.constants in the SI unit system
G = constants.G.si.value
N_A = constants.N_A.si.value
R = constants.R.si.value
Ryd = constants.Ryd.si.value
a0 = constants.a0.si.value
alpha = constants.alpha.si.value
atm = constants.atm.si.value
b_wien = constants.b_wien.si.value
c = constants.c.si.value
e = constants.e.si.value
eps0 = constants.eps0.si.value
g0 = constants.g0.si.value
h = constants.h.si.value
hbar = constants.hbar.si.value
k_B = constants.k_B.si.value
m_e = constants.m_e.si.value
m_n = constants.m_n.si.value
m_p = constants.m_p.si.value
mu0 = constants.mu0.si.value
muB = constants.muB.si.value
sigma_T = constants.sigma_T.si.value
sigma_sb = constants.sigma_sb.si.value
u = constants.u.si.value
GM_earth = constants.GM_earth.si.value
GM_jup = constants.GM_jup.si.value
GM_sun = constants.GM_sun.si.value
L_bol0 = constants.L_bol0.si.value
L_sun = constants.L_sun.si.value
M_earth = constants.M_earth.si.value
M_jup = constants.M_jup.si.value
M_sun = constants.M_sun.si.value
R_earth = constants.R_earth.si.value
R_jup = constants.R_jup.si.value
R_sun = constants.R_sun.si.value
kpc = constants.kpc.si.value

# Metric prefix
quetta = 1e30
ronna = 1e27
yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deca = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21
yocto = 1e-24
ronto = 1e-27
quecto = 1e-30

# Others
mumol = 2.37 * m_p  # kg; Kauffmann et al. 2008, for sound speed
muH2 = 2.8 * m_p  # kg; Kauffmann et al. 2008, for mass-H2 conversion
c_kms = constants.c.to('km*s**(-1)').value