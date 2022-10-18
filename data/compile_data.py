import csv
import json
import os
import sys

import numpy as np
import pandas as pd

import pynmrstar

aa_dict = {
	'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 
	'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L', 
	'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R',
	'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'
}


class DatasetError(object):
	def __init__(
		self,
		seq=None,
		nesg=None,
		bmrb=None,
		shiftlist=None,
		msg=None):
		
		self.seq       = seq
		self.nesg      = nesg
		self.bmrb      = bmrb
		self.msg       = msg


def fasta_reader(path):
	seqs    = []
	seqlist = []
	
	fp = open(path)
	
	for line in fp.readlines():
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				seqlist.append(seq)
				name = line[1:]
				seqs = []
			else:
				name = line[1:]
		else:
			seqs.append(line)
	seqlist.append(''.join(seqs))
	fp.close()
	return seqlist


def read_seq(files_list, base_path):
	seqfile = [f for f in files_list if '.seq' in f]
	assert(len(seqfile) == 1)
	
	path = os.path.join(base_path, seqfile[0])
	
	sequences = fasta_reader(path)
	
	return sequences


def make_shift_dict(seq):
	dic = {}
	
	dic = {i+1: dict() for i in range(len(seq))}
	
	for ind, aa in zip(dic.keys(), seq):
		dic[ind]['res'] = aa
		dic[ind]['N']   = None
		dic[ind]['H']   = None
		dic[ind]['C']   = None
		dic[ind]['CA']  = None
	
	return dic


def get_shifts(shifts, sequence):
	
	indices    = shifts.get_tag('Seq_ID')
	res_codes  = shifts.get_tag('Comp_ID')
	atm_codes  = shifts.get_tag('Atom_ID')
	chem_shift = shifts.get_tag('Val')
	
	try:
		indices = [int(i) for i in indices]
	except:
		err = DatasetError(
			msg = 'indices not of base 10?, {indices}',
			bmrb=shifts.get_tag('Entry_ID')[0],
			seq=sequence)
		return err
	
	assert(len(indices) == len(res_codes) == len(atm_codes) == len(chem_shift))
	
	shift_dict = make_shift_dict(sequence)
	
	for id, res, atm, cs in zip(indices, res_codes, atm_codes, chem_shift):
		
		if id not in shift_dict:
			err = DatasetError(
				bmrb=shifts.get_tag('Entry_ID')[0],
				seq=sequence,
				msg=f'{id} not in dictionary')
			return err
		
		if res not in aa_dict:
			err = DatasetError(
				msg=f'{res} not in aa dictionary, at {id}',
				bmrb=shifts.get_tag('Entry_ID')[0],
				seq=sequence)
			return err
		
		if aa_dict[res] != shift_dict[id]['res']:
			err = DatasetError(
				bmrb=shifts.get_tag('Entry_ID')[0],
				seq=sequence,
				msg=''.join((
					f"sequences do not agree <><><> {id}, {res},",
					f" {aa_dict[res]}, {shift_dict[id]['res']}")))
			return err
		
		if atm not in shift_dict[id]: continue
		shift_dict[id][atm] = cs	
	
	compact_dict = {'seq': sequence, 'peaks': []}
	
	for ind in shift_dict:
		
		peak = []
		
		if shift_dict[ind]['N'] is None: 
			peak.append(np.nan)
		else:
			peak.append(float(shift_dict[ind]['N']))
		
		if shift_dict[ind]['H'] is None: 
			peak.append(np.nan)
		else:
			peak.append(float(shift_dict[ind]['H']))
		
		if shift_dict[ind]['C'] is None: 
			peak.append(np.nan)
		else:
			peak.append(float(shift_dict[ind]['C']))
		
		if shift_dict[ind]['CA'] is None:
			peak.append(np.nan)
		else:
			peak.append(float(shift_dict[ind]['CA']))
		
		compact_dict['peaks'].append(peak)
	
	return compact_dict


def read_bmrb(files_list, base_path, seq):
	bmrbfile = [f for f in files_list if '.bmrb' in f]
	assert(len(bmrbfile) == 1)
	
	path = os.path.join(base_path, bmrbfile[0])
	with open(path, 'r') as fp:
		
		# read entry from file
		try:
			entry = pynmrstar.Entry.from_file(fp)
		except: 
			err = DatasetError(seq=seq, msg='pynmrstar failed')
			return err
		
		# grab shift lists
		shift_loop = entry.get_loops_by_category("Atom_chem_shift")[0]
		results = get_shifts(shift_loop, seq)
		if isinstance(results, DatasetError): return results
		
		results['bmrbid'] = entry.entry_id
		points = np.array(results['peaks'],dtype=np.float32)
		
		# create masks for null values
		mask = np.isnan(points)
		pts_masked = np.ma.array(points, mask=mask)
		
		# sort mask arrays by HN dimension
		pts_sorted = pts_masked[:,1].argsort(axis=0)
		
		# re-make peak lists with sorted mask array
		results['peaks'] = pts_masked[pts_sorted]
		results['order'] = pts_sorted
		return results


def process_shiftx2_preds(path, seq):
	# NUM,RES,ATOMNAME,SHIFT
	grp = ['N', 'H', 'C', 'CA']
	preds = {}
	
	pred_peaks = []
	with open(path, 'r') as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			if int(row['NUM']) not in preds: preds[int(row['NUM'])] = dict()
			preds[int(row['NUM'])]['aa'] = row['RES']
			if row['ATOMNAME'] not in grp: continue
			if row['ATOMNAME'] not in preds[int(row['NUM'])]:
				preds[int(row['NUM'])][row['ATOMNAME']] = float(row['SHIFT'])
	
	for ind, aa in enumerate(seq):
		ii = ind+1
		if ii not in preds:
			pred_peaks.append([np.nan, np.nan, np.nan, np.nan])
			continue
		
		if preds[ii]['aa'] == 'B':
			if aa != 'B' and aa != 'C':
				print('problem with cysteine')
				print(f'seq: {aa}')
				print(f'pdb: {preds[ind]["aa"]}')
				sys.exit()
		elif preds[ii]['aa'] != aa:
			return DatasetError(seq=seq, nesg='', msg=''.join((
				f'sequences dont match in shiftx2, aa: {aa}, index: {ii}, '
				f'shiftx2 aa: {preds[ii]["aa"]}')))
		
		p = []
		if 'N' not in preds[ii]:  p.append(np.nan)
		else:                     p.append(preds[ii]['N'])
		
		if 'H' not in preds[ii]:  p.append(np.nan)
		else:                     p.append(preds[ii]['H'])
		
		if 'C' not in preds[ii]:  p.append(np.nan)
		else:                     p.append(preds[ii]['C'])
		
		if 'CA' not in preds[ii]: p.append(np.nan)
		else:                     p.append(preds[ii]['CA'])
		
		pred_peaks.append(p)
	
	return pred_peaks


shiftx2 = '/Users/kfraga/RESEARCH/shiftX2-usage/shiftx2-mac/shiftx2.py'
dataset = []
errors  = []
none = 0
for root, dirs, files in os.walk(sys.argv[1]):
	if len(files) == 0: continue
	if len(dirs) != 0:  continue
	print(root, files)
	
	seq = read_seq(files, os.path.abspath(root))
	if len(seq) > 1:
		print('dimer?')
		sys.exit()
	seq = seq[0]
	
	peaks = read_bmrb(files, os.path.abspath(root), seq)
	if isinstance(peaks, DatasetError):
		peaks.nesg = root
		errors.append(peaks.__dict__)
		continue
	
	pdb = [f for f in files if f.endswith('.pdb')]
	assert(len(pdb) == 1)
	pdb = pdb[0]
	
	out = ''.join(pdb.split('/')[:-1])
	out += f'{pdb}.cs'
	cmd = f'python2 {shiftx2} -i {os.path.join(os.path.abspath(root), pdb)}'
	os.system(cmd)
	
	preds = process_shiftx2_preds(os.path.join(os.path.abspath(root), out), seq)
	if isinstance(preds, DatasetError):
		preds.nesg = root
		errors.append(preds.__dict__)
		continue
	peaks['predictions'] = np.array(preds)
	#sys.exit()
	
	dataset.append(peaks)

df = pd.DataFrame(dataset)
df.to_pickle('./data.pkl')

with open('errors.json', 'w') as fp:
	json.dump(errors, fp, indent=2)

"""
if 'bmrb' not in file: continue
print(file)
file = os.path.abspath(os.path.join(sys.argv[1], file))

entry_peaks = process_bmrb(file)
if entry_peaks is None: sys.exit()
dataset.append(entry_peaks)
"""