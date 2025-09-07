# utils.py
def pauli_commutes(p1, p2):
    diff_count = 0
    for a,b in zip(p1, p2):
        if a == 'I' or b == 'I': continue
        if a != b: diff_count += 1
    return (diff_count % 2) == 0

def group_commuting(pauli_list):
    groups = []
    for p in pauli_list:
        placed = False
        for g in groups:
            if all(pauli_commutes(p, q) for q in g):
                g.append(p); placed = True; break
        if not placed:
            groups.append([p])
    return groups
