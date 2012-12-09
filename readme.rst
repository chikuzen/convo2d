=====================================================
convo2d - Spatial convolution filter for VapourSynth
=====================================================

General spatial convolution (3x3 or 5x5) filter.::

    convo2d.Convolution(clip clip[, int[] matrix, float bias, float divisor, int[] planes])

Parameters:
-----------
    matrix: can be a 3x3 or 5x5 matrix with 9 or 25 integer numbers.
        default is [0, 0, 0, 0, 1, 0, 0, 0, 0].
    bias: additive bias to adjust the total output intensity.
        default is 0.0.
    divisor: divides the output of the convolution (calculated before adding bias). 0.0 means sum of the elements of the matrix or 1.0(when the sum is zero).
        default is 0.0.
    planes: planes which processes.
        default is all planes of the clip.

Examples:
---------
    >>> import vapoursynth as vs
    >>> core = vs.Core()
    >>> core.std.LoadPlugin('/path/to/convo2d.dll')
    >>> clip = something

    - blur(5x5) only Y(or R) plane:
    >>> matrix = [10, 10, 10, 10, 10,
                  10, 10, 10, 10, 10,
                  10, 10, 16, 10, 10,
                  10, 10, 10, 10, 10,
                  10, 10, 10, 10, 10]
    >>> blured = core.convo2d.Convolution(clip, matrix, planes=0)

    - Displacement UV(or GB) planes by quarter sample up:
    >>> matrix = [0, 1, 0,
                  0, 3, 0,
                  0, 0, 0]
    >>> clip = core.convo2d.Convolution(clip, matrix, planes=[1, 2])

    - edge detection with Sobel operator:
    >>> import math
    >>> def binalyze(val, thresh):
    ...     return 236 if val > thresh else 16
    ...
    >>> def get_lut(thresh):
    ...     lut = []
    ...     for y in range(256):
    ...         for x in range(256):
    ...             lut.append(binalyze(math.sqrt(x * x + y * y), thresh))
    ...     return lut
    ...
    >>> clip = core.resize.Point(clip, format=vs.GRAY8)
    >>> horizontal = [1, 2, 1, 0, 0, 0, -1, -2, -1]
    >>> vertical = [1, 0, -1, 2, 0, -2, 1, 0, -1]
    >>> edge_h = core.convo2d.Convolution(clip, horizontal, divisor=8)
    >>> edge_v = core.convo2d.Convolution(clip, vertical, divisor=8)
    >>> clip = core.std.Lut2([edge_h, edge_v], get_lut(16), 0)

How to compile:
---------------
    on unix system(include mingw/cygwin), type as follows::

    $ git clone git://github.com/chikuzen/convo2d.git
    $ cd ./convo2d
    $ ./configure
    $ make

    if you want to use msvc++, then

    - rename convo2d.c to convo2d.cpp
    - create vcxproj yourself


Author: Oka Motofumi (chikuzen.mo at gmail dot com)
