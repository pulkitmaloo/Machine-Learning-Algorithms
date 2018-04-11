#! /bin/bash/env python3

# Load libraries
import pandas as pd
from load_data import load_data
import itertools
import time
import pprint


# ### Config ######
MAX_K = 4
cnt = 0
# #################


def candidate_generation_v1(f_k_1, f_1):
    """ Generate candidate itemsets using F(k-1) X F(1) Method """
    c_k = []
    for itemset_k_1 in f_k_1:
        for i, itemset_1 in enumerate(f_1):
            if itemset_1[0] > itemset_k_1[-1]:
                c_k.append(itemset_k_1 + itemset_1)
    return c_k


def candidate_generation_v2(f_k_1):
    """ Generate candidate itemsets using F(k-1) X F(k-1) Method """
    c_k = []
    for i in range(len(f_k_1)):
        for j in range(i+1, len(f_k_1)):
            if f_k_1[i][: -1] == f_k_1[j][: -1] and f_k_1[i][-1] != f_k_1[j][-1]:
                new_cand = f_k_1[i][: -1] + sorted([f_k_1[i][-1], f_k_1[j][-1]])
                c_k.append(new_cand)
                # todo more pruning check book
    return c_k


def candidate_pruning(F_k, c_k, k):
    """ Prune the generated candidates """
    c_k_pruned = []
    for cand in c_k:
        flag = True
        # Generate candidate subsets
        cand_subsets = [list(x) for x in itertools.combinations(cand, k-1)]
        # if any of candidate_subsets in infrequent prune candidate
        for cand_subset in cand_subsets:
            if cand_subset not in F_k[k-1]:  # check if infrequent
                flag = False
                break
        if flag:
            c_k_pruned.append(cand)
    return c_k_pruned


def apriori_gen(F_k, k, method="v2"):
    if method == "v1":
        c_k = candidate_generation_v1(f_k_1=F_k[k-1], f_1=F_k[1])
    elif method == "v2":
        c_k = candidate_generation_v2(f_k_1=F_k[k-1])

    c_k_pruned = candidate_pruning(F_k, c_k, k)

    return c_k_pruned, len(c_k)


def subset(c_k, k, t):
    """ Identify all candidates that belong to Transaction t """
    t = t[t == 1].index.values
    transaction_itemsets = list(map(sorted, itertools.combinations(t, k)))
    return [x for x in c_k if x in transaction_itemsets]


def get_frequent_itemset(df, MINSUP, method="v2"):
    print("Generating frequent itemsets using",
          {"v1": "F(k-1) x F1", "v2": "F(k-1) x F(k-1)"}[method])
    items = df.columns

    k = 1
    F_k = dict()
    F_k[k] = [[i] for i in sorted(items[df.sum(axis=0) >= len(df) * MINSUP].values)]
    support = {(x[0], ): x[1] for x in df.sum(axis=0).iteritems()
               if x[1] >= len(df) * MINSUP}

    print("k", k, "->", len(F_k[1]), "frequent itemsets")
    cand_cnt = 0
    while MAX_K > k:
        k += 1
        c_k, cnt = apriori_gen(F_k, k, method)
        cand_cnt += cnt


        for t in range(len(df)):
            c_t = subset(c_k, k, df.iloc[t, :])
            for cand in c_t:
                support[tuple(cand)] = support.get(tuple(cand), 0) + 1
        fk = [c for c in c_k if support.get(tuple(c), 0) >= len(df) * MINSUP]
        if len(fk) == 0:
            break
        print("k", k, "->", len(fk), "frequent itemsets")
        F_k[k] = fk

    print()
    print(cand_cnt, "candidates generated")
    print(len(list(itertools.chain.from_iterable(F_k.values()))),
          "frequent itemsets generated")
    print(len(closed_freq(F_k, support)), "closed frequent itemsets")
    print(len(maximal_freq(F_k)), "maximal frequent itemsets")

    return F_k, support


def ap_genrules(f_k, H_m, support, rules, MINCONF):
    global cnt
    k = len(f_k)
    try:
        m = len(H_m[0])
    except:
        return
    if k > (m+1):
        H_m1 = candidate_generation_v2(H_m)

        cnt += len(H_m1)
#        print(len(H_m1))

        for h_m1 in H_m1:
            h_m1_t = sorted(set(f_k.copy()) - set(h_m1))
#            print(h_m1_t, "->", h_m1)
            try:
                conf = support[tuple(f_k)]/support[tuple(h_m1_t)]
            except:
                conf = 0
            if conf >= MINCONF:
#                print(h_m1_t, "->", tuple(h_m1), "\tConfidence", round(conf, 2))
                rules.append((conf, conf/support[tuple(h_m1)], h_m1_t, h_m1))
            else:
                H_m1.remove(h_m1)

        ap_genrules(f_k, H_m1, support, rules, MINCONF)


def rule_generation(F_k, support, MINCONF):
    print("Generating rules...", end=" ")
    rules = []
    global cnt
    cnt = 0
    for itemset in itertools.chain.from_iterable(F_k.values()):
        if len(itemset) == 1:
            continue
        H_1 = [[item] for item in itemset]
        cnt += len(H_1)
        # Generate rules of consequent size 1
        for h_1 in H_1:
            h_t = sorted(set(itemset.copy()) - set(h_1))
            conf = support[tuple(itemset)]/support[tuple(h_t)]
            if conf >= MINCONF:
#                print(h_t, "->", tuple(h_1), "\tConfidence", round(conf, 2))
                rules.append((conf, conf/support[tuple(h_1)], h_t, h_1))

        ap_genrules(itemset, H_1, support, rules, MINCONF)
    print(len(rules), "rules generated!")
    return rules


def maximal_freq(F_k):
    maximal_freq_dict = {}
    for key, value in F_k.items():
        for val in value:
            maximal_freq_dict[tuple(val)] = 0

    for k in range(len(F_k), 1, -1):
        for fk in F_k[k]:
            subsets = itertools.combinations(fk, k-1)
            for subset in subsets:
                maximal_freq_dict[tuple(subset)] = maximal_freq_dict[tuple(subset)] + 1

    return [item for item, val in maximal_freq_dict.items() if val == 0]


def closed_freq(F_k, support):
    maximal_freq_dict = {}
    for key, value in F_k.items():
        for val in value:
            maximal_freq_dict[tuple(val)] = 0

    for k in range(len(F_k), 1, -1):
        for fk in F_k[k]:
            subsets = itertools.combinations(fk, k-1)
            for subset in subsets:
                if support[tuple(subset)] == support[tuple(fk)]:
                    maximal_freq_dict[tuple(subset)] = maximal_freq_dict[tuple(subset)] + 1

    return [item for item, val in maximal_freq_dict.items() if val == 0]


def print_rules(rules_df):
    print("Rules Sorted by Confidence")
    print(rules_df.sort_values(by="Confidence", ascending=False).reset_index(drop=True).head())
    print("Rules Sorted by Lift")
    print(rules_df.sort_values(by="Lift", ascending=False).reset_index(drop=True).head())


def apriori(data="test", MINSUP=0.6, MINCONF=0.6, cand_method="v2"):
    print("minsup  =", MINSUP, "\nminconf =", MINCONF)

    df = load_data(data)

    tic = time.time()
    F_k, support = get_frequent_itemset(df, MINSUP, cand_method)
    toc = time.time()
#    pprint.pprint(F_k)
#    print("Time taken: ", round(toc - tic, 2), "seconds\n")

    tic = time.time()
    rules = rule_generation(F_k, support, MINCONF)
    rules_df = pd.DataFrame(rules, columns=["Confidence", "Lift", "Antecedent", "Consequent"])
    print_rules(rules_df)
    toc = time.time()
#    print("Time taken: ", round(toc - tic, 2), "seconds")
    return F_k, rules_df, support


def part_ef():
    thres_dict = {"car": itertools.product([0.02, 0.05, 0.1], [0.1, 0.3, 0.6]),
                  "nursery": itertools.product([0.02, 0.05, 0.1], [0.1, 0.3, 0.6]),
                  "chess": itertools.product([0.02, 0.05, 0.1], [0.1, 0.3, 0.6])
                  }
    for data in ("car", "nursery", "chess"):
        print("####################", data.title(), "Dataset #########################")
        for minsup, minconf in thres_dict[data]:
            F_k, rules, support = apriori(data, minsup, minconf, "v2")
            print("==========================================================")


def part_d():
    thres_dict = {"car": itertools.product([0.02, 0.05, 0.1], [0.1, 0.3, 0.6]),
                  "nursery": itertools.product([0.02, 0.05, 0.1], [0.1, 0.3, 0.6]),
                  "chess": itertools.product([0.02, 0.05, 0.1], [0.1, 0.3, 0.6])
                  }
    for data in ("car", "nursery", "chess"):
        print("\n", data)
        for minsup, minconf in thres_dict[data]:
            F_k, rules, support = apriori(data, minsup, minconf, "v2")


if __name__ == "__main__":
    part_ef()
