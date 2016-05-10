#!/usr/bin/python

import sys

## configure the great pyramid display from the user input but supply sensible
## defaults. note these are all positional parameters.
_n = 10	 if len(sys.argv) < 2 else int(sys.argv[1])
_p = '*' if len(sys.argv) < 3 else sys.argv[2]
_s = 'o' if len(sys.argv) < 4 else sys.argv[3]
_i = False 
if len(sys.argv) >= 5:
    if sys.argv[4] == '-1':
        _i = True
_f = 2  if len(sys.argv) < 6 else int(sys.argv[5])

## print out a descrption of the paramters being used
print 'Using: ', [_n, _p, _s, _i, _f], '\n'
interval = range(1, _n+1)
if _i:
    interval = range(_n, 0, -1)
## global numeric value used to assign sequential numbers to pyramids
idx = 1

## magic make function
makepyr = lambda f, r: '\n'.join(map(f, r))
makepyr_interval = lambda f: makepyr(f, interval)    

## common display routine
def display_pyr(f):
    global idx
    print 'Type %d:' %idx
    idx += 1
    print makepyr_interval(f), '\n'
    return 0 ## just to make it map-able
 
## pyramid definitions
pyr_types = [   
    lambda x: _p * x,
    lambda x: ' ' * (_n - x) + _p * x,
    lambda x: _p * x + _s * (_n - x),
    lambda x: ' ' * (_n - x) + '%s ' %_p * x,
    lambda x: ' ' * (_n - x % (_n/_f)) + '%s ' %_p * (x % (_n/_f))
]

## run all pyramids
map(display_pyr, pyr_types)