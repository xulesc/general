#!/usr/bin/python

## Constants
MIN_ARGS = 2
TAG_INDEX = 2
OP_INDEX = 1
USAGE_MESSAGE = "Usage ./bibit.py {cli2bib|verifybib} \
{article|book|conference|inbook|incollection|inproceedings|manual|misc|online|\
phdthesis|proceedings|techreport|unpublished} parameters"

## Exits
ILLEGAL_USAGE = 1
UNKNOWN_TAG = 2
UNSUPPORTED_OP = 3
WRONG_PARMS = 4

## Tags supported
article = ['author','title','journal','volume','pages','year']
book = ['author','title','publisher','edition','year']
conference = ['author','title','booktitle','volume','pages','year']
inbook = ['author','title','booktitle','pages','year']
incollection = ['author','title','booktitle','volume','pages','year']
inproceedings = ['author','title','booktitle','volume','pages','year']
manual = ['author','title','organization','year']
misc = ['author','title','howpublished','year']
online = ['author','title','howpublished','year']
phdthesis = ['author','title','school','year']
proceedings = ['title','publisher','volume','year']
techreport = ['author','title','institution','type','year']
unpublished = ['author','title','note','year']


## Command line key:value pairs for cli2bib
cli_data = {'-a':'author','-t':'title','-j':'journal','-v':'volume','-p':'pages'
,'-y':'year','-pu':'publisher','-e':'edition','-bt':'booktitle'
,'-o':'organization','-hp':'howpublished','-s':'school','-i':'institution'
,'-ty':'type', '-n':'note','-k':'key'}