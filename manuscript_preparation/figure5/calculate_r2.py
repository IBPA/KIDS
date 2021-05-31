from sklearn.metrics import r2_score

fraction_of_positives = [
	12/121,
	4/28,
	6/18,
	17/26,
	25/33]

ideal_line = [
	0.1,
	0.3,
	0.5,
	0.7,
	0.9]

print(fraction_of_positives)
print(r2_score(ideal_line, fraction_of_positives))

hypotheses_26 = [
	(0.000000+0.000316+0.022114+0.000000+0.000000+0.000023)/6,
	0.223327,
	0.573165,
	(0.701304+0.761767+0.797500+0.755129)/4,
	(0.981132+0.939076+0.995238+0.973245+0.995238+0.930385+0.802814+0.995238+0.995238+0.995238+0.821742+0.981132+0.994769+0.986443)/14
]

print(hypotheses_26)
print(r2_score(ideal_line, hypotheses_26))
