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
XY_PAIRS = ["xx","xy","xz","yx","yy","yz","zx","zy","zz"]

def write_out(top_msgs = ["---"], code_lines = [], trailing_msgs = ["----"]):
  for l in top_msgs + code_lines + trailing_msgs:
    print l

def make_sa_node():
  olines = []
  olines.append("library IEEE;\nuse IEEE.STD_LOGIC_1164.ALL;\nuse IEEE.STD_LOGIC_ARITH.ALL;\nuse IEEE.STD_LOGIC_UNSIGNED.ALL;");
  olines.append("entity node is")
  olines.append("\tPort (")
  olines.append("\t\tS1, S2 : in integer; -- input S pair")
  olines.append("\t\top : out integer -- output product")
  olines.append("\t);")
  olines.append("end node;")
  olines.append("--")
  olines.append("architecture rtl of node is")
  olines.append("begin")
  olines.append("\top <= S1 * S2;")
  olines.append("end rtl;")
  ##
  clines = []
  clines.append("\tcomponent node is")
  clines.append("\t\tPort (S1, S2 : in integer;")
  clines.append("\t\t\top : out integer);")
  clines.append("\tend component;")
  return [olines, clines]
  
def make_sa(component):
  olines = []
  olines.append("library IEEE;\nuse IEEE.STD_LOGIC_1164.ALL;\nuse IEEE.STD_LOGIC_ARITH.ALL;\nuse IEEE.STD_LOGIC_UNSIGNED.ALL;");
  olines.append("entity sys_array is")
  olines.append("\tPort (")
  olines.append("\t\t" + ", ".join(["inS%s" %x for x in XY_PAIRS]) + " : in integer;") 
  ##
  # make signals
  all_signals = []; t_count = 0;
  for X in range(0, XY_DIM):
    signals = []
    for Y in range(X, XY_DIM):
      signals.append('p%s%s' %(XY_PAIRS[X], XY_PAIRS[Y]));
      t_count += 1;
    all_signals.append(signals);
    #print '\tsignal %s : integer;' %(', '.join(signals))  
  ##
  for signals in all_signals:
    olines.append("\t\t" + ", ".join(signals) + " : out integer;")
  olines.append("\t);")
  olines.append("end sys_array;")
  olines.append("--")
  ##
  olines.append("architcture rtl of sys_array is")
  #for s in all_signals:
  #  olines.append("\tsignal %s : integer;" %(', '.join(s)));
  for s in component:
    olines.append(s)
  olines.append("begin")
  ##
  for X in range(0, XY_DIM):
    olines.append("\t--Row %d" %X)  
    for Y in range(X, XY_DIM):      
      olines.append('\tN%d%d: node port map(inS%s, inS%s, p%s%s);' %(X, Y, XY_PAIRS[X], XY_PAIRS[Y], XY_PAIRS[X], XY_PAIRS[Y]))
  olines.append("end rtl;")
  ##
  clines = ["\tcomponent sys_array is"]
  clines.append("\t\tPort (")
  clines.append("\t\t\t" + ", ".join(["inS%s" %x for x in XY_PAIRS]) + " : in integer;")
  for signals in all_signals:
    clines.append("\t\t\t" + ", ".join(signals) + " : out integer;")
  clines.append("\t\t);")
  clines.append("\tend component;")        
  ##
  ulines = []
  ifs = ["sa_%s" %x for x in XY_PAIRS]
  ulines.append("signal " + ", ".join(ifs) + " : integer;");
  for signals in all_signals:
    ulines.append("signal " + ", ".join(["sa_%s" %x for x in signals]) + " : integer;")
  uline = "SA0: sys_array port map("
  fs = ["sa_%s" %x for x in XY_PAIRS]
  for signals in all_signals:
    for s in signals:
      fs.append("sa_%s" %s)
  uline += ", ".join(ifs + fs) + ")";
  ulines.append(uline)
  return [olines, clines, ulines]

## 
entity_code_lines, component_code_lines = make_sa_node()
write_out(["----","node.vhd","----"], entity_code_lines)
##
entity_code_lines_sa, component_code_lines_sa, usage_comment_lines = make_sa(component_code_lines)
write_out(["----","sys_array.vhd","----"], entity_code_lines_sa)
write_out(code_lines = component_code_lines_sa)
write_out(code_lines = usage_comment_lines)

##

