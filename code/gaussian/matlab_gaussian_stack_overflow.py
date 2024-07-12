
from oct2py import octave
# to add a folder use:
octave.addpath('/Users/rebecca/Documents/PhD/Research/stf/')  # doctest: +SKIP
# to add folder with all subfolder in it use:
#octave.addpath(octave.genpath('/Users/rebecca/Documents/PhD/Research/stf/'))  # doctest: +SKIP
# to run the .m file :
octave.run('fit_multiple_gaussians.m', nout='max_nout')  # doctest: +SKIP

# import numpy as np
# from oct2py import octave
# x = np.array([[1, 2], [3, 4]], dtype=float)
# #use nout='max_nout' to automatically choose max possible nout
# octave.addpath('./example')  # doctest: +SKIP
# out, oclass = octave.roundtrip(x,nout=2)  # doctest: +SKIP
# import pprint  # doctest: +SKIP
# pprint.pprint([x, x.dtype, out, oclass, out.dtype])  # doctest: +SKIP
# [array([[1., 2.],
#         [3., 4.]]),
#     dtype('float64'),
#     array([[1., 2.],
#         [3., 4.]]),
#     'double',
#     dtype('<f8')]