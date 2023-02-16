import numpy as np
import plotly.offline as po
import plotly.graph_objs as go
from skimage import measure

from plotastrodata.plot_utils import kwargs2AstroData, kwargs2AstroFrame


def plot3d(levels: list = [3,6,12],
           xskip: int = 1, yskip: int = 1,
           eye_p: float = 0, eye_i: float = 180,
           outname: str = 'plot3d', show: bool = False, **kwargs):
    f = kwargs2AstroFrame(kwargs)
    d = kwargs2AstroData(kwargs)
    f.read(d, xskip, yskip)
    volume, x, y, v, rms = d.data, d.x, d.y, d.v, d.rms
    dx, dy, dv = x[1] - x[0], y[1] - y[0], v[1] - v[0]
    volume[np.isnan(volume)] = 0        
    if dx < 0: x, dx, volume = x[::-1], -dx, volume[:, :, ::-1]
    if dy < 0: y, dy, volume = y[::-1], -dy, volume[:, ::-1, :]
    if dv < 0: v, dv, volume = v[::-1], -dv, volume[::-1, :, :]
    s, ds = [x, y, v], [dx, dy, dv]
    deg = np.radians(1)
    xeye = -np.sin(eye_i * deg) * np.sin(eye_p * deg)
    yeye = -np.sin(eye_i * deg) * np.cos(eye_p * deg)
    zeye = np.cos(eye_i * deg)
    margin=dict(l=0, r=0, b=0, t=0)
    camera = dict(eye=dict(x=xeye, y=yeye, z=zeye), up=dict(x=0, y=1, z=0))
    xaxis, yaxis, zaxis = [dict(range=[t[0], t[-1]]) for t in s]
    scene = dict(aspectmode='cube', camera=camera,
                 xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)
    layout = go.Layout(margin=margin, scene=scene, showlegend=False)

    data = []
    for lev in levels:
        if lev * rms > np.max(volume): continue
        vertices, simplices, _, _ = measure.marching_cubes(volume, lev * rms)
        Xg, Yg, Zg = [t[0] + i * dt for t, i, dt
                      in zip(s, vertices.T[::-1], ds)]
        i, j, k = simplices.T
        mesh = dict(type='mesh3d', x=Xg, y=Yg, z=Zg, i=i, j=j, k=k,
                    intensity=Zg * 0 + lev,
                    colorscale='Jet', reversescale=False,
                    cmin=np.min(levels), cmax=np.max(levels),
                    opacity=0.08, name='', showscale=False)
        data.append(mesh)
        Xe, Ye, Ze = [], [], []
        for T in vertices[simplices]:
            Xe += [x[0] + dx * T[k % 3][2] for k in range(4)] + [None]
            Ye += [y[0] + dy * T[k % 3][1] for k in range(4)] + [None]
            Ze += [v[0] + dv * T[k % 3][0] for k in range(4)] + [None]
        lines=dict(type='scatter3d', x=Xe, y=Ye, z=Ze,
                   mode='lines', opacity=0.04, visible=True,
                   name='', line=dict(color='rgb(0,0,0)', width=1))
        data.append(lines)

    fig = dict(data=data, layout=layout)
    po.plot(fig, filename=outname + '.html', auto_open=show)
