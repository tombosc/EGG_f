import unittest
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations, product
from collections import defaultdict

def compute_transitivity_exact(order_dict, total):
    """ Brute force.
    """
    # turn the string keys into tuples
    order_dict = {
        tuple([int(r) for r in k.split(',')]): v for k, v in
        order_dict.items()
    }
    to_combine = {}
    n = max([max(k) for k in order_dict.keys()])
    for comb in combinations(range(n+1), 2):
        S = tuple(sorted(comb))
        reverse = (S[1], S[0])
        to_combine[S] = [
            (S, order_dict.get(S, 0)),
            (reverse, order_dict.get(reverse, 0)),
        ]

    max_val = 0
    max_edges = {}
    for P in product(*to_combine.values()):
        edges = [k for k, _ in P]
        if dfs(edges) == []:  # failure!
            continue
        val = sum([n for _, n in P])
        val -= sum([order_dict.get((k[1], k[0]), 0) for k, _ in P])
        if val > max_val:
            max_val = val
            max_edges = edges
    return max_val / float(total), set(max_edges)

def compute_transitivity_greedy(order_dict, total):
    """ Contains a dict with keys "0,1" "1,0", "0,2", "2,0", "1,2", "2,1", and more.
    Computes transitivity.
    """
    # TODO ugly that it takes such a dict of str instead of tuples...
    seen = set()
    sum_counts = 0
    final_edges = set()
    for k, count in order_dict.items():
        role1, role2 = [int(r) for r in k.split(',')]
        S = tuple(sorted((role1, role2)))
        if S in seen:
            continue
        opposite_k = str(role2) + "," + str(role1)
        opposite_count = order_dict.get(opposite_k, 0)
        two_counts = (count, opposite_count)
        sum_counts += (max(two_counts) - min(two_counts))
        #  print(count, opposite_count)
        if count == max(two_counts):
            final_edges.add((role1, role2))
        else:
            final_edges.add((role2, role1))
        seen.add(S)
    # need to check transitivity: 0, 1
    if dfs(final_edges) == []:  # failure!
        return 0, {}
    return sum_counts / float(total), final_edges

def dfs(edges):
    nodes = set([a for a, b in edges]).union(set([b for a, b in edges]))
    temp_marks = set()
    perm_marks = set()
    L = []
    
    def visit(e):
        if e in temp_marks:
            return 1  # not a DAG
        if e in perm_marks:
            return 0
        temp_marks.add(e)
        for (a, b) in edges:
            if e == a:
                V = visit(b)
                if V == 1:
                    return 1
        temp_marks.remove(e)
        perm_marks.add(e)
        L.insert(0, e)
    
    while True:
        left_nodes = False
        for e in nodes:
            if e in perm_marks:
                continue
            left_nodes = True
            if visit(e) == 1:
                return []
        if not left_nodes:
            break
    return L
    
class Tester(unittest.TestCase):
    def test_transitivity(self):
        # in all these examples, the total is pretty random (except in the
        # "perfect" one)
        # real example
        ex = {"0,1": 1314, "0,2": 247, "2,1": 312,
              "2,0": 75, "1,0": 16, "1,2": 18}
        T_score, edges = compute_transitivity_greedy(ex, 2000)
        T_score_exact, edges_exact = compute_transitivity_exact(ex, 2000)
        assert(T_score_exact == T_score)
        O = dfs(edges)
        self.assertTrue(O == [0, 2, 1])
        ex = {"0,1": 1000, "0,2": 200, "2,1": 300,
              "1,0": 50, "2,0": 50, "1,2": 100}
        T_score, edges = compute_transitivity_greedy(ex, 1700)
        T_score_exact, edges_exact = compute_transitivity_exact(ex, 1700)
        assert(T_score_exact == T_score)
        self.assertTrue(T_score == (950 + 150 + 200) / 1700)
        O = dfs(edges)
        self.assertTrue(O == [0, 2, 1])
        perfect = {"0,1": 300, "0,2": 200, "2,1": 1000,
                   "1,0": 0, "2,0": 0, "1,2": 0}
        n = sum(perfect.values())
        T_score, edges = compute_transitivity_greedy(perfect, n)
        T_score_exact, edges_exact = compute_transitivity_exact(perfect, n)
        print(T_score, T_score_exact, edges, edges_exact)
        assert(T_score_exact == T_score)
        self.assertTrue(T_score == 1.0)
        # when one key is missing, it should still work!
        ex = {"0,1": 1701, "0,2": 411, "2,1": 517, "2,0": 2, "1,2": 3}
        T_score, edges = compute_transitivity_greedy(ex, 3500) 
        T_score_exact, edges_exact = compute_transitivity_exact(ex, 3500)
        assert(T_score_exact == T_score)
        self.assertTrue(T_score != 0.0)

    def test_intransitive(self):
        # in this test, the heuristic from compute_transitivity_greedy fails and it
        # returns a non transitive / acyclic set of edges.
        # dfs should detect this, although we do not receover from the mistake
        # made by the greedy compute_transitivity_greedy.
        # (the fact that the alg fails means that the transitive subgraph is
        # not "obvious")
        topo_order = dfs(set({(0, 2), (2, 1), (1, 0)}))
        self.assertTrue(topo_order == [])
        topo_order = dfs(set({(0, 2), (2, 1), (0, 1)}))
        self.assertTrue(topo_order == [0, 2, 1])
