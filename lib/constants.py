"""
/*
*       Coded by : Jaspreet Singh Kalsi.
*
*       "Thesis  Chapter-2 Part A
*       (Image Fragmentation using Inverted Dirichlet Distribution using Markov Random Field as a Prior).
*
*       ```python core.py <Image-Name>```
*
*/

"""

CONST = {}
CONST['K'] = 4
CONST['IMG'] = '1.jpg'
CONST['cluster_drop_cond'] = 0.0001
CONST['algConverge'] = 0.01
CONST['PIXEL_COLOR_ONE'] = [[0.30332078,  0.27290306,  0.43117691],
                            [0.13345756, 0.01805083, 0.08127872],
                            [0.03548756, 0.43605482, 0.43984528],
                            [0.05775268, 0.42134079, 0.35717922]]