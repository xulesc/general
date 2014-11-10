#!/usr/bin/python
import sys, string, re
from bibit_globals import *

#bib_entry_part = re.compile(',(.*?=.*?[}"])')

def get_tag():
  return sys.argv[TAG_INDEX].lower()
  
def get_function_name():
  return sys.argv[OP_INDEX].lower()
  
def do_exit(msg, exit_number):
  print 'Exit Number: %d\n%s' %(exit_number, msg)
  sys.exit(exit_number)

def validate(gvars):
  arg_count = len(sys.argv)
  
  if arg_count < MIN_ARGS:
    do_exit(USAGE_MESSAGE, ILLEGAL_USAGE)
    
  if get_function_name() not in gvars:
    do_exit(USAGE_MESSAGE, UNSUPPORTED_OP)

## input parsers
def make_input_parm_tuple(start_at):
  args = sys.argv[start_at :]
  keys = args[0::2]
  values = args[1::2]
  args_data = zip(keys,values)
  
  if len(keys) != len(values):
    do_exit("Error in parameter list", WRONG_PARMS)

  return args_data

def parse_input_for_tag(gvars):   
  if get_tag() not in gvars:
    do_exit(USAGE_MESSAGE, UNKNOWN_TAG)  
    
  data_map = {}
  
  for z in make_input_parm_tuple(TAG_INDEX + 1):
    k = cli_data.get(z[0])
    if k == None: ## skip unknown parameters
      continue
    data_map[k] = z[1]
    
  return data_map
  
def parse_input_for_verify():
  data_map = {}
  
  for z in make_input_parm_tuple(OP_INDEX + 1):
    data_map[z[0]] = z[1]
    
  return data_map['-f']
  
## verification functions
def bibchecker(bib_entry, bad_bib_map, key_map, duplicate_key_set):
  str_bib_entry = " ".join(bib_entry)
  start_index = 1
  
  # find tag
  stop_index = string.find(str_bib_entry,'{')
  tag = str_bib_entry[start_index:stop_index].lower().strip()
  #print "tag: %s" %tag
  
  # find key
  start_index = stop_index + 1
  stop_index = string.find(str_bib_entry, ',',start_index)
  key = str_bib_entry[start_index:stop_index].lower()
  if key_map.get(key) == None:
    key_map[key] = 1
  else:
    duplicate_key_set.add(key)
    
  #print "key: %s" %key
  
  # find other fields
  str_bib_entry = str_bib_entry.lower()
  for x in globals()[tag]:
    part = re.match('.*(%s\s*=)'%x,str_bib_entry)
    if part == None:
      v = bad_bib_map.get(key)
      if v == None:
	bad_bib_map[key] = []
      bad_bib_map[key].append(x)
