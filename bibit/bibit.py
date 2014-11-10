#!/usr/bin/python

from bibit_utils import *
from sets import Set

def cli2bib(tag, tag_fields, input_data):
  out_str = '@%s{%s,\n' %(tag,input_data['key']) 
  for e in tag_fields:
      out_str += '\t%s = {{%s}},\n' %(e, input_data[e])
  out_str = out_str[:len(out_str)-2] + '\n}'
  return out_str
  
def verifybib():
  bib_tex_entry = []; bad_bib_map = {}; key_map = {}; duplicate_key_set = Set()
  
  # find bad bib entries
  for line in open(parse_input_for_verify(), 'r'):
    line = line.replace('\n','').strip()
    if len(line) == 0:
      continue
    
    if line[0] == '@':
      if bib_tex_entry != []:
	bibchecker(bib_tex_entry, bad_bib_map, key_map, duplicate_key_set)
	bib_tex_entry = []

    bib_tex_entry.append(line)
  bibchecker(bib_tex_entry, bad_bib_map, key_map, duplicate_key_set)
  
  # print out the bad entries
  print 'Found %d duplicate keys: ' %(len(duplicate_key_set))
  print duplicate_key_set
  print '\nFound %d errors' %len(bad_bib_map)
  for (k,v) in bad_bib_map.iteritems():
    print 'Bibtex entry with key: %s' %k
    for e in v:
      print '\tError in %s' %e
      
if __name__ == '__main__':
    
  validate(vars())
  
  ## if we are here everything is in order
  function_name = get_function_name()
  
  if function_name == 'cli2bib':
    tag = get_tag()
    tag_fileds = vars()[tag]    
    ## make map of input parameters
    input_data = parse_input_for_tag(vars())
    
    ## print out the bibtex
    print cli2bib(tag, tag_fileds, input_data)
  elif function_name == 'verifybib':
    verifybib()    
  else:
    print 'This cant be. Disorderly default impending!'
    