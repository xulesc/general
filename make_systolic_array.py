#!/usr/bin/python

'''
Un-generalized code that generates a systolic array output for VHDL.

@TODO:
  # input component specification
  # signal forwarding symantics

    Sxx Sxy Sxz Syx Syy Syz Szx Szy Szz
Sxx 0,0 0,1 0,2 0,3 0,4 0,5 0,6 0,7 0,8
Sxy 1,0 
Sxz 2,0 
Syx 3,0 	...........
Syy 4,0 	...........
Syz 5,0 	...........
Szx 6,0 
Szy 7,0 
Szz 8,0
 
since it is symmetric we need only the upper triangular part.

'''

XY_DIM = 9

print 'architecture rtl of sys_array is'
## make signals
for X in range(0, XY_DIM):
  signals = []
  for Y in range(X, XY_DIM):
    signals.append('s%d%d' %(X, Y));
    signals.append('p%d%d' %(X, Y));
    signals.append('d%d%d' %(X, Y));
  print '\tsignal %s : integer;' %(', '.join(signals))
print 'begin'
## make RTL
for X in range(0, XY_DIM):
  print '\t--Row %d' %X
  for Y in range(X, XY_DIM):
    print '\tN%d%d: node port map(inS(%d), inS(%d), s%d%d, p%d%d, d%d%d);' %(X, Y, X, Y, X, Y, X, Y, X, Y)    

## write everything
print 'end rtl;'