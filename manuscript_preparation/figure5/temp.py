import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
iris = sns.load_dataset("iris")
print(iris.head())

species = iris.pop("species")
print(species.head())

g = sns.clustermap(iris)

plt.show()
