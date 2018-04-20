import json


def _find_frequent_itemsets(itemsets, dataset, required_amount):
    result = []
    # Count how frequently every itemset appears in samples of dataset
    for itemset in itemsets:
        occurrences = 0
        for sample in dataset:
            if all(item in sample for item in itemset):
                occurrences += 1
        if occurrences >= required_amount:
            result.append(itemset)
    return result


def _next_itemsets(itemsets):
    # Find next itemsets based on previously found ones
    # Split every itemset to 2 parts - 1 element and rest. Save them to dictionary
    subsets = {}
    for itemset in itemsets:
        for i in range(len(itemset)):
            element = itemset[i]
            rest = [*itemset[:i], *itemset[i + 1:]]
            key = json.dumps(rest)
            if key not in subsets:
                subsets[key] = set()
            subsets[key].add(element)

    # Create new itemsets using saved parts
    new_itemsets = {}
    for key, value in subsets.items():
        if len(value) > 1:
            prefix = json.loads(key)
            sort = list(sorted(value))
            for i in range(len(sort)):
                for j in range(i + 1, len(sort)):
                    new_itemset = list(sorted([*prefix, sort[i], sort[j]]))
                    new_key = repr(new_itemset)
                    if new_key not in new_itemsets:
                        new_itemsets[new_key] = new_itemset

    # Filter valid itemsets, i.e. ones whose subsets without one element
    # were in the previous list of itemsets
    previous = set([repr(itemset) for itemset in itemsets])
    result = []
    for itemset in new_itemsets.values():
        for i in range(len(itemset)):
            subset = [*itemset[:i], *itemset[i + 1:]]
            key = repr(subset)
            if key not in previous:
                break
        else:
            result.append(itemset)
    return result


def apriori(dataset, minspan):
    """
    Apriori algorithm by Rakesh Agrawal and Ramakrishnan Srikant

    Finds all frequent itemsets in the dataset with specified minspan.
    Itemset is a set of elements which appears in not less that minspan-part of the dataset.

    Based on the article `"Fast algorithms for mining association rules." <http://www.vldb.org/conf/1994/P487.PDF>`_.

    Parameters
    ----------
    dataset: list
        List of size n_samples whose elements are sets of integers.
        Each set represents a sample from the dataset.
    minspan: float
        MinSpan value. Algorithm will select sets of items that appear in not less
        than (MinSpan * n_samples) samples.

    Returns
    -------
    itemsets: list
        List of frequent itemsets. Every itemset is a list of integers in increasing order.
    """

    n_samples = len(dataset)
    required_amount = int(n_samples * minspan)
    all_items = set(item for sample in dataset for item in sample)
    result = []

    # Start with one-element frequent subsets
    current = [[item] for item in all_items]
    previous = _find_frequent_itemsets(current, dataset, required_amount)
    result.extend(previous)

    while len(previous) > 0:
        # Generate next datasets and select frequent ones among them
        current = _next_itemsets(previous)
        previous = _find_frequent_itemsets(current, dataset, required_amount)
        result.extend(previous)
    return result
