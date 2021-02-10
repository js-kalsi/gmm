"""
/*
*       Coded by : Jaspreet Singh Kalsi.
*
*       "Thesis  Chapter-2 Part A
*       (Image Fragmentation using Inverted Dirichlet Distribution using Markov Random Field as a Prior).
*
*       ```python core.py <Image-Name>```
*
*
*   EM Algorithm Inverted Dirichlet Mixture Model.
*       1) Convert image pixels into array.
*       2) Normalize it.
*       3) Assume Number of Cluster(K) =  10 & apply KMeans clustering algorithm
*          to obtain K clusters for Initialization purposes.
*       4) Use `Method of Moments` for obtaining the initial values for Mixing Parameters.
*       5) Expectation Step:
*                           => Compute the Posterior Probability.
*       6) Maximization Step:
*                           => Update the Mixing Parameter.
*                           => Update the Alpha Parameter using `Newton Raphson` Method.
*       7) If Mixing Parameter of any Cluster < Cluster-Skipping Threshold:
*                           => Skip that particular Cluster.
*       8) Compute the Log Likelihood and check for Convergence by comparing the difference
*          between last two likelihood values with the Convergence Threshold.
*       9) If algorithm(converge) :terminate
*       10) Go-to Step 5(Expectation Step).
*/

"""
import sys
from dataSet import DataSet  # Importing DataSet
from KMeans import KMeans as KM # contains KNN related functionality.
from gaussian import Gaussian as gaussian
from lib.helpers import helpers as HELPER  # class contains the logic like performanceMeasure, Precision etc.
from lib.constants import CONST  # contains the constant values.
from sklearn.preprocessing import normalize as NORMALIZE
from numpy import sum as SUM
from numpy import asarray as ASARRAY


"""
/**
 * This function contains the logic for `Initial Algorithm`.
 * @param  {Integer} K: Number of Clusters.
 * @param {Float Array} which image pixels in array form having dimension  of (N * DIM).
 * @return {Float Array} alphaSet in array form having dimension  of (N * DIM).
 * @return {Float Array} imgPixels in array form having dimension  of (N * DIM).
 * @return {Integer} DIM: Number of Features 
 * @return {Integer} pixelSize: Size of img Array.
 * @return {Float Array} mix: Mixture Components value in (1 x N) dimention.
 * @return {Integer} imageW: Contains the width of an image.
 * @return {Integer} imageH: Contains the height of an image.
 */
"""


def initial_algorithm(k, img_name):
    pixels, image_width, image_height = DataSet(img_name).pixel_extractor()
    # pixels = NORMALIZE(pixels)
    # pixels += sys.float_info.epsilon
    p_size = len(pixels)
    p_labels = ASARRAY(KM(pixels, k).predict()).reshape(1, p_size)
    cluster_set, dim = HELPER().split_pixels_based_on_label(p_labels[0], pixels)
    mix_initial = HELPER().initial_mix(cluster_set, p_size, k)
    mean_initial = HELPER().initial_mean(cluster_set)
    cov_initial = HELPER().initial_covariance(mean_initial, cluster_set)
    return pixels, dim, p_size, mix_initial, mean_initial, cov_initial, image_width, image_height


"""
/**
 * This function Contains the Estimation Step logic.
 * @param  {Integer} K.
 * @param  {Integer} mix.
 * @param  {Integer} alphaSet.
 * @param  {Integer} imgPixels.
 * @param  {Integer} pixelSize.
 * @return {String} pdfMatrix.
 * @return {String} posteriorProbability.
 */
"""


def estimation_step(k, mix_param, mean_param, cov_param, pixels, p_size, image_width, image_height, dim):
    pdf_e_step = gaussian(k, dim, mean_param, cov_param, pixels).pdf_fetcher()
    # print("pdf_e_step :>", pdf_e_step)
    posterior_e_step = HELPER().posterior_estimator(pdf_e_step, mix_param, p_size, k)
    mrf_e_step = HELPER().markov_random_fld(posterior_e_step, image_height, image_width, k, mix_param)
    return pdf_e_step, posterior_e_step, mrf_e_step


"""
/**
 * This function Contains the Maximization Step logic.
 * @param  {Integer} K.
 * @param  {Integer} alphaSet.
 * @param  {Integer} imgPixels.
 * @param  {Integer} DIM.
 * @param  {Integer} posteriorProb.
 * @param  {Integer} pixelSize.
 * @param  {Integer} imageH.
 * @param  {Integer} imageW.
 * @param  {Integer} mix.
 * @return {String} mix.
 * @return {String} alpha.
 */
"""


def maximization_step(k, pixels, dim, posterior_param, mrf_param):
    mix_m_step = HELPER().mix_estimator(posterior_param, mrf_param)  # Checked: Working Fine!
    # print("mix_m_step :>", mix_m_step)
    mean_m_step = HELPER().mean_estimator(posterior_param, pixels, k)
    # print("mean_m_step :>", mean_m_step)
    cov_m_step = HELPER().cov_estimator(pixels, posterior_param, mean_m_step, dim)
    # print("cov_m_step :>", cov_m_step)
    return mix_m_step, mean_m_step, cov_m_step


"""
/**
 * This function add the array's element and return them in the form of a String.
 * @param  {Integer} a.
 * @return {String} which contains the Sum of Array.
 */
"""

if __name__ == '__main__':
    K = CONST["K"]

    if len(sys.argv) == 2:
        img = sys.argv[1]
    else:
        img = CONST['IMG']
    imgArr = img.split(".")

    if len(imgArr) == 2:
        imgName = imgArr[0]
        imgExtension = '.' + imgArr[1]
    else:
        imgName = imgArr[0]
        imgExtension = '.jpg'

    img_pixels, dimension, pixel_size, mix, mean, covariance, img_width, img_height = initial_algorithm(K, img)
    # print("Cov :>", len(covariance), covariance)
    counter = 1
    obj = {
            'logLikelihood': [],
            'alpha': [],
            'previous_likelihood': []
          }
    while True:
        pdf, posterior, mrf = estimation_step(K, mix, mean, covariance, img_pixels,
                                              pixel_size, img_width, img_height, dimension)
        mix, mean, covariance = maximization_step(K, img_pixels, dimension, posterior, mrf)
        if counter:
            predictLabels = HELPER().predictLabels(posterior)
            print("Counter : ", counter)
            generateImg = HELPER().generateImg(K, predictLabels, img_width, img_height, img_pixels, imgName,
                                               imgExtension, counter, CONST["PIXEL_COLOR_ONE"])
            counter = counter + 1
            if counter == 3:
                exit()
