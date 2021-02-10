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

from numpy import sum as SUM
from numpy import subtract as SUBS
from numpy import zeros as ZEROS
from numpy import pi as PI
from numpy.linalg import det as DET
from numpy import exp as EXP
from numpy import log as LOG
from numpy.linalg import inv as INVERSE
from numpy import matmul as MATMUL
from numpy import asarray as ASARRAY
from numpy import real as REAL
from numpy import exp as EXP
from numpy import diag as DIAGONAL
import warnings
warnings.filterwarnings("error")

"""
/**
 * This function add the array's element and return them in the form of a String.
 * @param  {Integer} a.
 * @return {String} which contains the Sum of Array.
 */
"""


class Gaussian:

    def __init__(self, k, dim, mean, cov, pixels):
        self.k = k
        self.dim = dim
        self.mean = mean
        self.cov = cov
        self.pixels = pixels

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def pdf_fetcher(self):
        prob = ZEROS((len(self.pixels), self.k))
        for m_index, (m_v, c_v) in enumerate(zip(self.mean, self.cov)):
            for p_index, pixel in enumerate(self.pixels):
                prob[p_index][m_index] = self.pdf(ASARRAY(pixel).reshape(1, self.dim),
                                                  ASARRAY(m_v).reshape(1, self.dim),
                                                  ASARRAY(c_v, dtype=complex).reshape(self.dim, self.dim),
                                                  self.dim)
        return prob


    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def pdf(p_v, m_v, c_v, dim):
        foo = EXP(-(dim/2)*LOG(2*PI) - REAL(LOG(DET(c_v))/2) - MATMUL((p_v - m_v), MATMUL(INVERSE(c_v), (p_v - m_v).T))/2)
        return REAL(foo[0][0])