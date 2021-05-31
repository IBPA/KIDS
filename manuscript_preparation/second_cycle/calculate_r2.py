from sklearn.metrics import r2_score

fraction_of_positives = [
	2 / 26,
	8 / 20,
	6 / 11,
	9 / 10,
	4 / 4]

ideal_line = [
	0.1,
	0.3,
	0.5,
	0.7,
	0.9]


fraction_of_positives = [
	14/147,
	12/48,
	12/29,
	26/36,
	29/37]

print(fraction_of_positives)
print(r2_score(ideal_line, fraction_of_positives))
