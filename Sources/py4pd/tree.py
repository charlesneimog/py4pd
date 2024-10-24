import pd


def get_list_depth(t):
    if not isinstance(t, list):
        return 0
    return 1 + max((get_list_depth(item) for item in t), default=0)


def sum_tree(tree):
    if type(tree) == float or type(tree) == int:
        return tree
    tree_sum = 0
    for unit in tree:
        if type(unit) == float or type(unit) == int:
            tree_sum += unit
        else:
            tree_sum += unit[0]
    return tree_sum


def tuple_to_list(t):
    if not isinstance(t, tuple):
        return t
    return [tuple_to_list(item) for item in t]


def recursive_tree(tree, last_onset, unit_dur):
    onsets = []
    last_dur = 0
    for subtree in tree:
        if isinstance(subtree, list):
            subunit = subtree[0] * unit_dur
            subtree = subtree[1]
            subsum = sum_tree(subtree)
            subunit_dur = subunit / subsum
            level = get_list_depth(tree)
            if level != 1:
                subonsets, last_dur = recursive_tree(subtree, last_onset, subunit_dur)
                for onset in subonsets:
                    onsets.append(onset)
                last_onset = last_dur
            else:
                for unit in subtree:
                    onsets.append(last_onset)
                    last_dur = unit * subunit_dur
                    last_onset += last_dur
        else:
            onsets.append(last_onset)
            last_dur = subtree * unit_dur
            last_onset += last_dur
    return onsets, last_onset


def omtree(tree, figure, bpm):
    pd.clear_player()
    tree = " ".join(map(str, tree))
    tree = tree.replace("(", "[")
    tree = tree.replace(")", "]")
    tree = tree.replace(" ", ", ")
    tree = eval(tree)
    tree = tuple_to_list(tree)
    level = get_list_depth(tree)
    if level < 2:
        raise ValueError("Tree is wrong")
    if level == 2:
        tree = tuple(tree)
        raise ValueError("Tree is wrong")

    last_onset = 0
    onsets = []
    for measure in list(tree):
        num = measure[0]
        den = measure[1] / figure
        measure_tree = measure[2]
        measure_t = (60 / bpm) / den  # figura de tempo
        measure_total_dur = measure_t * num
        tree_sum = sum_tree(measure_tree)
        unit_dur = measure_total_dur / tree_sum
        if type(measure_tree) == float or type(measure_tree) == int:
            measure_tree = [measure_tree]
        sublevel = get_list_depth(measure_tree)
        if sublevel == 1:
            for unit in measure_tree:
                onsets.append(last_onset)
                last_onset += unit * unit_dur
        else:
            subonsets, last_onset = recursive_tree(measure_tree, last_onset, unit_dur)
            for onset in subonsets:
                onsets.append(onset)

    for onset in onsets:
        if onset == 0:
            onset = 0.001
        pd.add_to_player(onset * 1000, "bang")

    pd.out(onsets, out_n=1)
    return
