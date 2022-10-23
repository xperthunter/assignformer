
import json
import matplotlib.pyplot as plt
import sys
import warnings

from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
from numba import njit
import numpy as np
import pandas as pd


def rc_models(data=None, index=None):
	assert(index != None)
	assert(type(index) == int)
	assert(index <= 3)
	
	total = np.ma.concatenate(data.peaks.tolist())

	avg = np.mean(total[:,index].compressed())
	sd  = np.std(total[:,index].compressed())
	
	return (avg, sd)


def seq_models(data=None, index=None):
	assert(index != None)
	assert(type(index) == int)
	assert(index <= 3)
	
	model = dict()
	
	for peaks, seqs in zip(data.peaks.tolist(), data.seq.tolist()):
		for i, aa in enumerate(seqs):
			if i == 0: continue
			if peaks[i][index] is np.ma.masked: continue
			
			aa_prev = seqs[i-1]
			if aa_prev not in model:     model[aa_prev]     = dict()
			if aa not in model[aa_prev]: model[aa_prev][aa] = list()
			
			model[aa_prev][aa].append(peaks[i][index])
	
	for k1 in model:
		for k2, vals in model[k1].items():
			avg = np.mean(np.array(vals))
			sd  = np.std(np.array(vals))
			model[k1][k2] = (float(avg), float(sd))
	
	return model


def make_predictions(data=None, index=None, model_rc=None, model_seq=None):
	assert(index != None)
	assert(type(index) == int)
	assert(index <= 3)
	assert(model_rc != None)
	assert(model_seq != None)
	
	for i, (preds, seqs) in enumerate(
		zip(
			data.predictions.tolist(),
			data.seq.tolist())):
		
		for j, aa in enumerate(seqs):
			if j == 0: 
				data['seq_preds'][i][j][index] = model_rc[0]
				continue
			
			aa_prev = seqs[i-1]
			
			if aa_prev not in model_seq:
				data['seq_preds'][i][j][index] = model_rc[0]
			elif aa not in model_seq[aa_prev]:
				data['seq_preds'][i][j][index] = model_rc[0]
			else:
				data['seq_preds'][i][j][index] = model_seq[aa_prev][aa][0]
	
	return data


def metric(obs, pred, scale=dict()):
	dis = 0
	count = 0
	for i, (o, p) in enumerate(zip(obs, pred)):
		# just put a zero in if its masked
		if np.isnan(o) or np.isnan(p):
			dis += 0.0
			continue
		elif np.ma.is_masked(o):
			dis += 0.0
			continue
		count += 1
		dis += ((o - p)/scale[i])**2
	
	if count == 0: dis = np.nan	
	else:          dis = np.sqrt(dis) / count
	
	return dis


def metric_matrix(observed, preds, scale=None):
	
	matrix = np.ma.array(
		np.zeros(
			(observed.shape[0], preds.shape[0])
		),
		mask=np.zeros(
			(observed.shape[0],preds.shape[0])
		)
	)
	
	for i, obs in enumerate(observed):
		masked = obs.mask.sum()
		if masked == 4:
			matrix[i,:] = np.ma.masked
			continue
		
		for j, pred in enumerate(preds):
			matrix[i, j] = metric(obs, pred, scale=scale)
	
	return matrix


@njit()
def numba_for_loop(x,y):
	m = x.shape[0]
	k = y.shape[0]
	D = np.zeros((m,k))
	
	for i in range(m):
		for j in range(k):
			dis   = 0
			count = 0
			for o, p in zip(x[i,:], y[j,:]):
				if np.isnan(o) or np.isnan(p):
					dis += 0.0
					continue
				if o == -np.inf:
					dis += 0.0
					continue
				count += 1
				dis += ((o - p))**2
			
			if count == 0: D[i, j] = np.nan
			else:          D[i, j] = np.sqrt(dis) / count
	
	return D


def scale_shifts(shifts, scale):
	scales = np.tile(scale, (shifts.shape[0], 1))
	return np.divide(shifts, scales)


def two_step(row=None, matrix=None, row_i=None, threshold=False):
	
	row_assignment = np.zeros(row.shape[0])
	two_step_agreements = []
	
	row_min = row.min(fill_value=np.inf)
	
	if threshold:
		assert(type(threshold) == float)
		assert(threshold < 1.0)
		
		upper_bound = row_min * (1 + threshold)
		
		diffs = upper_bound - row
		mins = np.where(diffs > 0)[0]
	else:
		mins = np.where(row == row_min)[0]
	
	#if mins.shape[0] == row.shape[0]: return None, None
	
	for i in mins:
		col_j = matrix[:,i]
		col_min = col_j.min(fill_value=np.inf)
		
		mins = np.where(col_j == col_min)[0]
		if mins.shape[0] > 1: continue
		else:
			if mins[0] == row_i:
				two_step_agreements.append(i)
	
	if len(two_step_agreements) == 0:
		matrix[row_i,:] = np.ma.masked
		return row_assignment, matrix
	
	row_assignment[two_step_agreements] = 1/len(two_step_agreements) 
	matrix[row_i,:] = np.ma.masked
	return row_assignment, matrix


def make_assignment(metrics, assignment, threshold=False):
	
	global_min = metrics.min(fill_value=np.inf)
	if np.ma.is_masked(global_min): return assignment
	
	globes_ij = np.where(metrics == global_min)
	
	counts = {}
	
	for ri in globes_ij[0]:
		if ri not in counts: counts[ri] = 0
		counts[ri] += 1
	
	for rk in counts:
		assign_i, metrics = two_step(
			row=metrics[rk,:],
			matrix=metrics,
			row_i=rk,
			threshold=threshold)
		
		if assign_i is None:
			print('none here')
			return None
		assignment[rk,:] = assign_i
	
	global_min = metrics.min(fill_value=np.inf)
	if np.ma.is_masked(global_min): return assignment
	else:
		return make_assignment(metrics, assignment, threshold=threshold)


df = pd.read_pickle(sys.argv[1])
print(df.columns)


preds = df.peaks.tolist()
for i in range(len(preds)): preds[i] = np.zeros(preds[i].shape)


df['seq_preds'] = preds
scale = dict()
for i in range(4):
	rc_model  = rc_models(data=df, index=i)
	seq_model = seq_models(data=df, index=i)
	
	df = make_predictions(
		data=df,
		index=i,
		model_rc=rc_model,
		model_seq=seq_model)
	
	scale[i] = float(rc_model[1])

print(json.dumps(scale,indent=2))
scalearr = np.array(list(scale.values()))


seqs_metrics    = []
shiftx2_metrics = []
for peaks, s2pred, seqpred in zip(
	df.peaks.tolist(),
	df.predictions.tolist(),
	df.seq_preds.tolist()
	):
	
	pscale = scale_shifts(peaks, scale=scalearr)
	predscale = scale_shifts(s2pred, scale=scalearr)
	seqscale = scale_shifts(seqpred, scale=scalearr)
	
	pcopy = pscale.copy()
	pcopy.filled(fill_value=-np.inf)
	
	numbas2 = numba_for_loop(pcopy, predscale)
	numbaseq = numba_for_loop(pcopy, seqscale)
	
	shiftx2_metrics.append(numbas2)
	seqs_metrics.append(numbaseq)

df['shiftx2_matrics'] = shiftx2_metrics
df['seq_matrices']    = seqs_metrics


headers = [
	'bmrbid',
	'assignable',
	'seq raw',
	'seq % raw',
	'seq pres.',
	'seq % pres.',
	'sx2 raw',
	'sx2 % raw',
	'sx2 pres.',
	'sx2 % pres.'
]
thresh = 0.50
results = []
for bmrbid, shiftx2, seqm, order in zip(
	df.bmrbid.tolist(),
	df.shiftx2_matrics.tolist(),
	df.seq_matrices.tolist(),
	df.order.tolist()):
	
	result = []
	# mask nans -- some reason they arent getting masked
	shiftx2 = np.ma.masked_invalid(shiftx2)
	seqm    = np.ma.masked_invalid(seqm)
	
	# get the mask for the shiftx2 and seqm to apply to answers
	shiftx2m  = np.ma.getmask(shiftx2).copy()
	seqm_mask = np.ma.getmask(seqm).copy()
	
	# make the correct assignment matrix
	answers = np.zeros((shiftx2.shape[0], shiftx2.shape[1]))
	for i, ind in enumerate(order): answers[i, ind] = 1.0
	shiftx2_answers = np.ma.array(answers, mask=shiftx2m)
	seqm_answers = np.ma.array(answers, mask=seqm_mask)
	
	# get number of assignable rows in assignment matrix
	shiftx2_assignable =  np.array(
		[not shiftx2[i,:].mask.all() for i in range(shiftx2.shape[0])])
	shiftx2_assignable = shiftx2_assignable.sum()
	
	seqm_assignable = np.array(
		[not seqm[i,:].mask.all() for i in range(seqm.shape[0])])
	seqm_assignable = seqm_assignable.sum()
	
	# create place holders for the assignment predictions
	assignment = np.ma.array(
		np.zeros((shiftx2.shape[0], shiftx2.shape[1])),
		mask=shiftx2m)
	
	# make the shiftx2 assignment prediction and score it
	shiftx2_assign = make_assignment(shiftx2, assignment, threshold=thresh)
	if shiftx2_assign is None: shiftx2_score = -1
	shiftx2_score = np.multiply(shiftx2_assign, shiftx2_answers)
	shiftx2_score = shiftx2_score.sum()
	
	row_scores = np.multiply(shiftx2_assign, shiftx2_answers).sum(axis=1)
	sx2_pres   = np.where(row_scores > 0)[0].shape[0]
	#print(sx2_pres)
	
	# make the seq-based assignment prediction and score it
	assignment = np.ma.array(
		np.zeros((seqm.shape[0], seqm.shape[1])),
		mask=seqm_mask)	
	
	seqm_assign = make_assignment(seqm, assignment, threshold=thresh)
	if seqm_assign is None: seqm_score = -1
	seqm_score = np.multiply(seqm_assign, seqm_answers)
	seqm_score = seqm_score.sum()
	
	row_scores = seqm_assign.sum(axis=1)
	seqm_pres = np.where(row_scores > 0)[0].shape[0]
	#print(seqm_pres)
	
	result.append(bmrbid)
	result.append(shiftx2_assignable)
	result.append(f'{seqm_score:.4f}')
	result.append(f'{(seqm_score / seqm_assignable) * 100.0:.4f}')
	result.append(f'{seqm_pres:.4f}')
	result.append(f'{(seqm_pres / seqm_assignable) * 100.0:.4f}')
	result.append(f'{shiftx2_score:.4f}')
	result.append(f'{(shiftx2_score / shiftx2_assignable) * 100.0:.4f}')
	result.append(f'{sx2_pres:.4f}')
	result.append(f'{(sx2_pres / shiftx2_assignable) * 100.0:.4f}')
	
	results.append(result)

results.sort(key=lambda x: x[1],reverse=True)
writer = MarkdownTableWriter(
	table_name='Nearest Assignment Results',
	headers=headers,
	value_matrix=results,
	margin=1,
	column_styles=[Style(align="center")] * len(headers)
)
writer.write_table()

threshold_values = [False, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9]

results = []
for bmrbid, shiftx2, order in zip(
	df.bmrbid.tolist(),
	df.shiftx2_matrics.tolist(),
	df.order.tolist()):
	
	result = []
	# mask nans -- some reason they arent getting masked
	shiftx2 = np.ma.masked_invalid(shiftx2)

	# get the mask for the shiftx2 and seqm to apply to answers
	shiftx2m  = np.ma.getmask(shiftx2).copy()

	# make the correct assignment matrix
	answers = np.zeros((shiftx2.shape[0], shiftx2.shape[1]))
	for i, ind in enumerate(order): answers[i, ind] = 1.0
	shiftx2_answers = np.ma.array(answers, mask=shiftx2m)

	# get number of assignable rows in assignment matrix
	shiftx2_assignable =  np.array(
		[not shiftx2[i,:].mask.all() for i in range(shiftx2.shape[0])])
	shiftx2_assignable = shiftx2_assignable.sum()

	# create place holders for the assignment predictions
	assignment = np.ma.array(
		np.zeros((shiftx2.shape[0], shiftx2.shape[1])),
		mask=shiftx2m)
	
	result.append(bmrbid)
	result.append(shiftx2_assignable)
	# make the shiftx2 assignment prediction and score it
	for t in threshold_values:
		sx2c = shiftx2.copy()
		assignment = np.ma.array(
			np.zeros((shiftx2.shape[0], shiftx2.shape[1])),
			mask=shiftx2m)
		shiftx2_assign = make_assignment(sx2c, assignment, threshold=t)
		if shiftx2_assign is None: shiftx2_score = -1
		shiftx2_score = np.multiply(shiftx2_assign, shiftx2_answers)
		shiftx2_score = shiftx2_score.sum()
		
		row_scores = np.multiply(shiftx2_assign, shiftx2_answers).sum(axis=1)
		sx2_pres   = np.where(row_scores > 0)[0].shape[0]
		
		line = ''.join((
			f'{shiftx2_score:.2f}-',
			f'{(shiftx2_score / shiftx2_assignable) * 100.0:.2f}-',
			f'{sx2_pres:.2f}-',
			f'{(sx2_pres / shiftx2_assignable) * 100.0:.2f}'))
		
		result.append(line)
	
	results.append(result)

headers = [
	'bmrbid', 'len', 'None', '0.01', '0.1',
	'0.2', '0.3', '0.4', '0.5', '0.9']

results.sort(key=lambda x: x[1],reverse=True)
writer = MarkdownTableWriter(
	table_name='Nearest Assignment Results with thresholding',
	headers=headers,
	value_matrix=results,
	margin=1,
	column_styles=[Style(align="center")] * len(headers)
)
writer.write_table()




"""
	
		#print(numbas2)
	#print(matshiftx2)
	#print(numbas2 - matshiftx2)
	#print(np.allclose(numbas2, matshiftx2, equal_nan=True))
	
	#if not np.allclose(numbas2, matshiftx2, equal_nan=True):
	#	sys.exit()
	
	#print(numbaseq)
	#print(np.allclose(numbaseq, matseq, equal_nan=True))
	#if not np.allclose(numbaseq, matseq, equal_nan=True):
	#	sys.exit()
	
	
	#sys.exit()
	
	
	# if matshiftx2.mask.all():
# 		print(matshiftx2)
# 		print(predictions)
# 		print(peaks)
# 		print(bmrb)
# 		sys.exit()
	
	#predscale.filled(fill_value=-np.inf)
	#seqscale.filled(fill_value=-np.inf)
	#print(pscale.shape)
	#print(predscale.shape)
	#sys.exit()
	
	#print(type(pcopy))

	#matshiftx2 = metric_matrix(peaks, s2pred, scale=scale)
	#matseq     = metric_matrix(peaks, seqpred, scale=scale)
	
	#sys.exit()
	#print(peaks)
	#print(predictions)
	#print(scalearr)

print(df.predictions[0][25])
print()
print(df.peaks[0][25])

print(df.predictions[0][26])
print()
print(df.peaks[0][26])

print(df.predictions[0][27])
print()
print(df.peaks[0][27])

print(df.predictions[0][28])
print()
print(df.peaks[0][28])
"""

"""


# nstd --> from only true n chemical shifts

def peak_metric(obs, pred):
	
	sqrt((nobs - npred / nstd)**2, hnobs - hnpred / hnstd, c, ca)
	

preds = np.zeros((len(seq), len(seq)))

metrics.mask == beforehand
[1 0 -- 0.4]
[0 5 -- 0.1]
[3.2 -- 9.1]

for i in range(len(seq)):
	mins = metrics.argmin(axis=None, fill_value=1e9)
	
	preds[mins] = 1.0
	
	metrics[mins] = np.ma.masked
	
trues = np.zeros((len(seq), len(seq)))

ans = np.multiply(preds, trues)

accuracy = ans.sum() / len(seq)
"""





"""

peaks vs pred

mis referenced carbon shifts are a particular
all things need to be referenced properly
nesg nmr xray pairs should have good referencing
initially use nmr xray pairs should be helpful
however we can also use lacs (nmrfam?) 
distribution of carbon chemical shifts and identify correction
5-10 ppm lacs could be a problem
lacs < 2 ppm
if > 2, then that list has problems


1. pred chemical shifts
2. compute distances (peaks vs the predictions)
3. nearest match (ignore the columns that are not reliable)
	-- maybe fill with random coil


avg n h c ca --> rc
n[i-1, i]  avg
h[i-1, i]  avg
c[i-1, i]  avg
ca[i-1, i] avg


preds
obs
if obs has a mask, ignore
obs x pred



score metric matrix
score assignment matrix
score assignment accuracy

for every row find min
for every col find min

for every min_r
if min_r == min_c mask
else continue

row min multiples
check where there is one minimum
and row and col agree

but what to do about duplicates??

need to find what rows were masked

sys.exit()
	
	row_mask = np.array(
		[np.ma.is_masked(metrics[i,:]) for i in range(metrics.shape[0])])
	#print(row_mask)
	row_mins = np.ma.array(
		metrics.argmin(axis=1, fill_value=np.inf),
		mask=row_mask)
	#print(row_mins)
	
	col_mins = metrics.argmin(axis=0, fill_value=1e9)
	#print(row_mins)
	#print(col_mins)
	
	# find reciporcals
	for i, rc in enumerate(row_mins):
		if np.ma.is_masked(row_mins[i]): continue
		if col_mins[rc] == i:
			print('found!')
			print(i, rc, col_mins[rc])
			two_step(row=metrics[i,:], matrix=metrics, row_i=i)
			#print(metrics[i,:])
			#print(metrics[:,rc])
			if np.ma.is_masked(assignment[i, rc]):
				print('how is this masked??')
				sys.exit()
			assignment[i, rc] = 1.0
			#print(assignment[i,:])
			#print(np.sum(assignment[i,:]))
			#print(assignment.sum())
			#print('*******')
			#print('>look here')
			#print(assignment)
			print(metrics[:,rc])
			print(metrics[i,rc])
			mm = metrics[i, :]
			print(metrics[i,:].tolist().count(metrics[i,rc]))
			continue
			metrics[i,:] = np.ma.masked
			print('////////////')
			print('> does metrics go over to assignment>>>>>>???????')
			print(assignment)
			print(assignment.sum())
			#print(metrics[i,:])
			#sys.exit()
	
	#print(metrics)
	#print('*****************************')
	#print('>>>>final')
	#print(assignment)
	#print(np.sum(assignment))
	#print(row_mask.sum())
	sys.exit()


counts = {}
	
	for i in range(matshiftx2.shape[0]):
		counts = {}
		for j in range(matshiftx2.shape[0]):
			if np.ma.is_masked(matshiftx2[i][j]): continue
			if np.isnan(matshiftx2[i][j]): continue
			if matshiftx2[i][j] not in counts: counts[matshiftx2[i][j]] = 0
			counts[matshiftx2[i][j]] += 1
			if counts[matshiftx2[i][j]] > 1:
				#print(matshiftx2[i,:])
				val = matshiftx2[i][j]
				wheres = np.where(matshiftx2 == matshiftx2[i][j])
				print()
				print(f'bmrbid: {bmrb}')
				print(f'val: {val}')
				print(f'row: {matshiftx2[wheres]}')
				print(f'peaks: {peaks[wheres[0]]}')
				print(f'predictions: {predictions[wheres[1]]}')
				print()
				#sys.exit()

# for bmrb, predictions in zip(df.bmrbid.tolist(), df.predictions.tolist()):
# 	print(type(predictions))
# 	predictions = np.ma.masked_invalid(predictions)
# 	mat = metric_matrix(predictions, predictions, scale=scale)
# 	#print(mat)
# 	mat = mat + np.identity(mat.shape[0])
# 	zeros = np.where(mat == 0.0)
# 	print(zeros)
# 	#sys.exit()

#sys.exit()


	#print(shiftx2_assign)
	print()
	print(shiftx2_assign.sum(axis=1))
	#plt.imshow(shiftx2_assign)
	#plt.show()
	#print(shiftx2_score)
	#sys.exit()

	# find which rows in the assignment matrix are assignable
	shiftx2_assignable =  np.array(
		[not shiftx2[i,:].mask.all() for i in range(shiftx2.shape[0])])
	shiftx2_assignable = shiftx2_assignable.sum()
	
	seqm_assignable = np.array(
		[not seqm[i,:].mask.all() for i in range(seqm.shape[0])])
	seqm_assignable = seqm_assignable.sum()


"""