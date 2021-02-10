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


import sys
import pickle
from math import pow as POW
from scipy.special import polygamma as POLYGAMMA
from numpy import sum as SUM
from numpy import log as LOG
from numpy import zeros as ZEROS
from numpy import transpose as TRANSPOSE
from numpy import asarray as ASARRAY
from numpy import mean as MEAN, var as VAR
from numpy import full as FULL
from numpy import concatenate as CONCAT
from sklearn.preprocessing import normalize as NORMALIZE
from numpy.linalg import inv as INVERSE
from numpy import diag as DIAGONAL
from numpy import matmul as MATMUL
from numpy import subtract as SUBTRACT
from numpy import exp as EXP
from numpy import mean as MEAN
from numpy import cov as COVARIANCE
from numpy import square as SQUARE
from numpy import absolute as ABS
from PIL import Image
from numpy.random import rand as RANDOM
import pathlib



class helpers:


    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def display(self, paramOne, paramTwo, paramThree):
        print(('{0}\'s {1} : {2}').format(paramOne, paramTwo, paramThree))

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def logger(self, data):
        file = open('./dataset/logs.txt', 'wb')
        pickle.dump(data, file)
        file.close()

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def rGB_Normalized_Space(self, imgPixels):
        red = imgPixels[:, 0]
        green = imgPixels[:, 1]
        blue = imgPixels[:, 2]
        colorSUM = red + green + blue
        normalizesPixels = [color/ sum  for color, sum in zip(imgPixels[:, :-2], colorSUM)]
        return normalizesPixels

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def initial_mix(cluster_set, p_size, k):
        mix = [len(cluster_set[cluster])/p_size for cluster in cluster_set]
        return ASARRAY([FULL((p_size, 1), pi) for pi in mix]).T.reshape(p_size, k)

    @staticmethod
    def initial_mean(cluster_set):
        return ASARRAY([MEAN(cluster_set[cluster], axis=0) for cluster in cluster_set])


    @staticmethod
    def initial_covariance(mean, cluster_set):
        # return ASARRAY([DIAGONAL((SUM(SQUARE(cluster_set[cluster]), axis=0)/len(cluster_set[cluster]))
        #                 - SQUARE(mean[cluster])) for cluster in cluster_set])
        return ASARRAY([COVARIANCE(ASARRAY(cluster_set[j]).T) for j in cluster_set])
    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def split_pixels_based_on_label(labels, pixels):
        clusters_obj = {}
        for index, label in enumerate(labels):
            if not (label in clusters_obj):
                clusters_obj[label] = []
            clusters_obj[label].append(pixels[index])
        return clusters_obj, len(clusters_obj[0][0])

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def posterior_estimator(pdf, mix, p_size, k):
        return ASARRAY([(m * p_v)/SUM(m * p_v) for m, p_v in zip(mix, pdf)]).reshape(p_size, k)

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def mix_estimator(post_prob, mrf):
        return ASARRAY([(p_v + m_v)/SUM(p_v + m_v) for p_v, m_v in zip(post_prob, mrf)])

    @staticmethod
    def mean_estimator(posterior, img_pixels, clusters):
        return ASARRAY([SUM(posterior[:, j:j+1] * img_pixels, axis=0) / SUM(posterior[:, j:j+1]) for j in range(clusters)])

    @staticmethod
    def cov_estimator(pixels, posterior, mean, dim):
        foo = []
        for j, m_v in enumerate(mean):
            z = posterior[:, j:j+1]
            doo = ([MATMUL(ASARRAY(p_v - m_v).reshape(1, dim).T,
                           z[i] * ASARRAY(p_v - m_v).reshape(1, dim))
                    for i, p_v in enumerate(pixels)])
            foo.append(SUM(doo, axis=0)/SUM(z))
        return ASARRAY(foo)


    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def g_estimation(self, imgPixels, pixelSize, alphaSet, posteriorProbability, DIM, K):
        G = []
        pixels = CONCAT((imgPixels, FULL((pixelSize, 1), 1)), axis=1)
        pixel_Log = ASARRAY([LOG(pixel / (1 + SUM(pixel[:DIM]))) for pixel in pixels])
        for index, alpha in enumerate(alphaSet):
            G.append(self.gMatrixGenerator(alpha, posteriorProbability[:, [index]], pixel_Log, DIM))
        return ASARRAY(G)

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def gMatrixGenerator(self, alpha, posterior, logPixels, DIM):
        return SUM(posterior * (POLYGAMMA(0, SUM(alpha)) - POLYGAMMA(0, alpha) + logPixels), axis=0).reshape(DIM + 1, 1)

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def hessian(dim, posterior, alpha):
        diagonal = []
        constant = []
        h_a = []
        a_t = []
        for index, a_v in enumerate(alpha):
            p_sum = SUM(posterior[:, index]) + sys.float_info.epsilon
            a_t_gamma = POLYGAMMA(1, a_v)
            diagonal.append(DIAGONAL(-1 * EXP(- LOG(p_sum) - LOG(a_t_gamma))).reshape(dim + 1, dim + 1))
            a_t_gamma_sum = POLYGAMMA(1, SUM(a_v))
            constant.append((a_t_gamma_sum * SUM(1 / a_t_gamma) - 1) * a_t_gamma_sum * p_sum)
            a = ((-1 / p_sum) * 1/a_t_gamma).reshape(1, dim + 1)
            h_a.append(a)
            a_t.append(TRANSPOSE(a))

        return ASARRAY(diagonal), constant, ASARRAY(h_a), ASARRAY(a_t)

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def hessian_inverse(self, K, hessian_Diagonal, hessian_constant, hessian_a, hessian_a_T):
        return ASARRAY([hessian_Diagonal[j] + hessian_constant[j]
                        * MATMUL(hessian_a_T[j], hessian_a[j]) for j in range(K)])

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def alpha_updator(self, alphaSet, hessian_Inverse, G, K, DIM):
        return ASARRAY([SUBTRACT(alphaSet[j].reshape(DIM + 1, 1), MATMUL(hessian_Inverse[j], G[j]))
                        for j in range(K)]).reshape(K, DIM + 1)

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """

    def markov_random_fld(self, posterior, img_height, img_width, k, mix):
        return ASARRAY([self.window(posterior[:, j: j + 1].reshape(img_height, img_width),
                                    img_height, img_width, mix[:, j:j + 1].reshape(img_height, img_width))
                        for j in range(k)]).T.reshape(img_height * img_width, k)

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def window(posterior, img_height, img_width, mix):
        a = 0
        b = 5
        result = []
        for h in range(img_height):
            c = 0
            d = 5
            for w in range(img_width):
                result.append(EXP((12/(2 * posterior[a:b, c:d].size)) * SUM(posterior[a:b, c:d] + mix[a:b, c:d])))
                c = c + 1
                d = d + 1
            a = a + 1
            b = b + 1
        return ASARRAY(result).reshape(img_height * img_width, 1)

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def clusterDropTest(self, mix, alpha, dropingCriteria, K, DIM, pixelSize):
        # print("Inside a ClusterDropTest()!")
        mixInfo = []
        alphaInfo = []
        # print("\n######### Inside Cluster Drop Test ##########\n")
        for j in range(K):
            if SUM(mix[:, j: j+1]) > dropingCriteria:
                mixInfo.append(mix[:, j: j+1])
                alphaInfo.append(alpha[j])
            else:
                print("Cluster having  alpha :", alpha[j], " & Mix :", j, " is removed!")
        return (ASARRAY(mixInfo).T).reshape(pixelSize, len(alphaInfo)), ASARRAY(alphaInfo).reshape(len(alphaInfo), DIM + 1), len(mixInfo)

    @staticmethod
    def log_likelihood(posterior, pdf, mix, m_r_field):
        return SUM(posterior * (LOG(mix) + LOG(pdf)) + m_r_field * LOG(mix))



    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def predictLabels(self, posteriorProbability):
        return posteriorProbability.argmax(axis=1)

    def hessianTest(self, alphaSet, DIM, K):
        return ASARRAY([INVERSE(FULL((DIM+1)**2, POLYGAMMA(1, SUM(aVector))).reshape(DIM+1, DIM+1) - DIAGONAL(POLYGAMMA(1, aVector))) for aVector in alphaSet])

    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def convergence_test(likelihood):
        l_size = len(likelihood)
        result = 0
        if l_size >= 2:
            print("Converence :>", ABS(likelihood[l_size - 1] - likelihood[l_size - 2]))
            result = ABS(likelihood[l_size - 1] - likelihood[l_size - 2])
        return result

    @staticmethod
    def log_likelihood(posterior, pdf, mix, m_r_field):
        return SUM(posterior * (LOG(mix) + LOG(pdf))) + SUM(m_r_field * LOG(mix))

    def generateImg(self, K, predictLabels, imageW,  imageH, imgPixels, imgName, imgExtension, counter, pColor):
        pColor = RANDOM(K, 3)
        pColor *= 0.45
        print("pColor :>", pColor)
        for cluster in range(K):
            for labelIndex, labelRecord in enumerate(predictLabels):
                if cluster == labelRecord:
                    imgPixels[labelIndex][0] = int(round(pColor[cluster][0] * 255))
                    imgPixels[labelIndex][1] = int(round(pColor[cluster][1] * 255))
                    imgPixels[labelIndex][2] = int(round(pColor[cluster][2] * 255))

        # Save image
        image = Image.new("RGB", (imageW, imageH))

        for y in range(imageH):
            for x in range(imageW):
                image.putpixel((x, y), (int(imgPixels[y * imageW + x][0]),
                                        int(imgPixels[y * imageW + x][1]),
                                        int(imgPixels[y * imageW + x][2])))
        pathlib.Path('./output/output_' + imgName).mkdir(parents=True, exist_ok=True)
        print("./output/output_" + imgName + "/" + str(counter) + imgExtension)
        image.save("./output/output_" + imgName + "/" + str(counter) + imgExtension)
        return self