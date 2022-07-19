def otonal_diamond(limit, diagonal, fundamental):
    otonal = []
    for o in range(1, limit, 2):
        tonality = []
        for u in range (1, limit, 2):
            tonality.append(fundamental * (u / o))
        otonal.append(tonality)
        tonality = [] 
    return otonal[diagonal]


def utonal_diamond(limit, diagonal, fundamental):
    utonal = []
    for o in range(1, limit, 2):
        tonality = []
        for u in range (1, limit, 2):
            tonality.append(fundamental * (o / u))
        utonal.append(tonality)
        tonality = [] 
    return utonal[diagonal]
    