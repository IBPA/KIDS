from ete3 import Tree, TreeStyle
import csv
from collections import defaultdict
from pprint import pprint
import pandas as pd

def tree():
    return defaultdict(tree)

def tree_add(t, path):
  for node in path:
    t = t[node]

def pprint_tree(tree_instance):
    def dicts(t): return {k: dicts(t[k]) for k in t}
    pprint(dicts(tree_instance))

def csv_to_tree(filename):
    t = tree()

    pd_data = pd.read_csv(filename, sep='\t', names=['Antibiotic', 'Taxonomy'])
    pd_data = pd_data.dropna()
    pd_data['concat'] = pd_data.apply(lambda row: row['Taxonomy'] + ':' + row['Antibiotic'], axis=1)

    for index, row in pd_data.iterrows():
        tree_add(t, row['concat'].replace(',', ' ').split(':'))

    return t

def tree_to_newick(root):
    items = []
    for k in root.keys():
        s = ''
        if len(root[k].keys()) > 0:
            sub_tree = tree_to_newick(root[k])
            if sub_tree != '':
                s += '(' + sub_tree + ')'
        s += k
        items.append(s)
    return ','.join(items)

def csv_to_weightless_newick(filename):
    t = csv_to_tree(filename)
    return tree_to_newick(t)

if __name__ == '__main__':
    newick = csv_to_weightless_newick('./result.txt') + ';'
    t = Tree(newick, format=8)

    for n in t.traverse():
        n.dist = 0.1

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.mode = "c"
    ts.arc_start = 270
    ts.root_opening_factor = 0.5
    ts.optimal_scale_level = 'full'
    # ts.scale = 100

    print(t.get_leaf_names())

    t.show(tree_style=ts)
    t.render("/home/jyoun/Jason/UbuntuShare/mytree_circular.svg", tree_style=ts)
    # t.show()
    # t.render("/home/jyoun/Jason/UbuntuShare/mytree_vertial.svg")
