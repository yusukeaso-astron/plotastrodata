from plotastrodata.fitting_utils import EmceeCorner


def logl(p):
    x1, x2, x3 = p
    chi2 = (x1 / 1)**2 + (x2 / 2)**2 + (x3 / 4)**2
    return -0.5 * chi2

# initialization
fitter = EmceeCorner(bounds=[[-5, 5], [-10, 10], [-20, 20]],
                     logl=logl, progressbar=False, percent=[16, 84])

# Using emcee, corner, and dynesty
fitter.fit(nwalkersperdim=30, nsteps=11000, nburnin=1000,
           savechain='chain.npy')
print('best:', fitter.popt)
print('lower percentile:', fitter.plow)
print('50 percentile:', fitter.pmid)
print('higher percentile:', fitter.phigh)
fitter.getDNSevidence()
print('evidence:', fitter.evidence)
fitter.plotcorner(show=True, savefig='corner.png',
                  labels=['par1', 'par2', 'par3'],
                  cornerrange=[[-4, 4], [-8, 8], [-16, 16]])
fitter.plotchain(show=True, savefig='chain.png',
                 labels=['par1', 'par2', 'par3'],
                 ylim=[[-2, 2], [-4, 4], [-8, 8]])

# Calculating logl on a parameter grid
fitter.posteriorongrid(ngrid=[101, 201, 401])
print('best:', fitter.popt)
print('lower percentile:', fitter.plow)
print('50th percentile:', fitter.pmid)
print('higher percentile:', fitter.phigh)
print('evidence:', fitter.evidence)
fitter.plotongrid(show=True, savefig='grid.png',
                  labels=['par1', 'par2', 'par3'],
                  cornerrange=[[-4, 4], [-8, 8], [-16, 16]])
