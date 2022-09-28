
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

import pynmrstar


def get_shifts(shifts):
	
	indices    = shifts.get_tag('Seq_ID')
	res_codes  = shifts.get_tag('Comp_ID')
	atm_codes  = shifts.get_tag('Atom_ID')
	chem_shift = shifts.get_tag('Val')
	
	assert(len(indices) == len(res_codes) == len(atm_codes) == len(chem_shift))
	
	shift_dict = dict()
	
	for id, res, atm, cs in zip(indices, res_codes, atm_codes, chem_shift):
		
		if id not in shift_dict: shift_dict[id] = dict()
		shift_dict[id]['res'] = res
		shift_dict[id][atm]   = cs	
		
	compact_dict = {'seq': '', 'peaks': []}
	
	for ind in shift_dict:
		compact_dict['seq'] += shift_dict[ind]['res']
		
		peak = []
		
		if 'N' not in shift_dict[ind]: peak.append(np.nan)
		else:                          peak.append(float(shift_dict[ind]['N']))
		
		if 'H' not in shift_dict[ind]: peak.append(np.nan)
		else:                          peak.append(float(shift_dict[ind]['H']))
		
		if 'C' not in shift_dict[ind]: peak.append(np.nan)
		else:                          peak.append(float(shift_dict[ind]['C']))
		
		if 'CA' not in shift_dict[ind]: peak.append(np.nan)
		else:                           peak.append(float(shift_dict[ind]['CA']))
		
		compact_dict['peaks'].append(peak)
	
	return compact_dict


def process_bmrb(file):
	with open(file, 'r') as fp:
		# read entry from file
		entry = pynmrstar.Entry.from_file(fp)
		
		# grab shift lists
		shift_loop = entry.get_loops_by_category("Atom_chem_shift")[0]
		results = get_shifts(shift_loop)
		results['bmrbid'] = entry.entry_id
		points = np.array(results['peaks'],dtype=np.float32)
		
		# create masks for null values
		mask = np.isnan(points)
		pts_masked = np.ma.array(points, mask=mask)
		
		# sort mask arrays by HN dimension
		pts_sorted = pts_masked[:,1].argsort(axis=0)
		
		# re-make peak lists with sorted mask array
		results['peaks'] = pts_masked[pts_sorted]
		
		return results

dataset = []

for file in os.listdir(sys.argv[1]):
	if 'bmrb' not in file: continue
	print(file)
	file = os.path.abspath(os.path.join(sys.argv[1], file))
	
	entry_peaks = process_bmrb(file)
	dataset.append(entry_peaks)

df = pd.DataFrame(dataset)
print(df.head(5))
print(df.peaks[:5])
print(type(df.peaks[0]))

seqlens = [len(seq) for seq in df.seq.tolist()]
max_len = np.amax(np.array(seqlens))
print(max_len)

for p, seq in zip(df.peaks.tolist(), df.seq.tolist()):
	print(f'>{seq[:10]}\t{len(seq)}')
	mask = p.mask
	sys.exit()

"""
with open(sys.argv[1], newline='') as csvfile:
	reader = csv.DictReader(csvfile, delimiter=';')
	
	for row in reader:
		dic = {'bmrid': '', 'seq': '', 'star': ''}
		#print(row.keys())
		print(row[' NMR structure sequence '])
		print()
		print(row[' BMRB id '])
		print()
		print(row[' path to shifts list '])
		
		paths = path_front+row[' path to shifts list ']
		cmd = f'wget {paths} /Users/kfraga/RESEARCH/assignformer/data/nesg_xray_pairs/'
		os.system(cmd)
		sys.exit()




def make_peak_lists(max_len=1024):
	
	peak_list = [''] * max_len
	
	for res in seq:
		n  = data['n'][res]
		h  = data['h'][res]
		c  = data['c'][res]
		ca = data['ca'][res]



for entry in frame:
"""