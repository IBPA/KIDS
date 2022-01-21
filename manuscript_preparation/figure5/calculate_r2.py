from sklearn.metrics import r2_score

ideal_line = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9]

fraction_of_positives = [
    0.085889571,
    0.235294118,
    0.413793103,
    0.722222222,
    0.783783784]

print(r2_score(ideal_line, fraction_of_positives))


ideal_line = [
    0.05,
    0.15,
    0.25,
    0.35,
    0.45,
    0.55,
    0.65,
    0.75,
    0.85,
    0.95]

fraction_of_positives = [
    0.079365079,
    0.108108108,
    0.185185185,
    0.291666667,
    0.294117647,
    0.583333333,
    0.642857143,
    0.772727273,
    0.8,
    0.78125]

print(r2_score(ideal_line, fraction_of_positives))
