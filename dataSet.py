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


from PIL import Image
from numpy import ndarray as ND_ARRAY
from numpy import asarray as ASARRAY
from numpy import power as POWER
from itertools import combinations as  COMBINATIONS
from numpy import sum as SUM
import sys


class DataSet:

    def __init__(self, img_name):
        self.img_name = img_name

    """
        /**
         * `loadCsv` function of dataSet Class.
         * @return  {Integer Vector} vector X & Y.
        */
    """
    def pixel_extractor(self):
        image = Image.open('/home/bugsbunny/Projects/masterThesis/BSDS500/data/images/test/' + self.img_name)
        img_width = image.size[0]
        img_height = image.size[1]
        # img_width = 3
        # img_height = 3

        # Initialise data vector with attribute r,g,b,x,y for each pixel
        img_pixels_v = ND_ARRAY(shape=(img_width * img_height, 5), dtype=float)


        """
            /**
             * Populate data vector with data from input image dataVector has 5 fields: red, green, blue, x coord, y coord
             * @return
            */
        """
        for y in range(0, img_height):
            for x in range(0, img_width):
                xy = (x, y)
                pixels = image.getpixel(xy)
                # rgb_sum = SUM(pixels)
                # if rgb_sum == 0:
                #     rgb = [0, 0, 0]
                # else:
                #     rgb = (pixels / rgb_sum) * 255.0
                rgb = self.photo_color_invariant(pixels)
                img_pixels_v[x + y * img_width, 0] = rgb[0]
                img_pixels_v[x + y * img_width, 1] = rgb[1]
                img_pixels_v[x + y * img_width, 2] = rgb[2]
                img_pixels_v[x + y * img_width, 3] = x
                img_pixels_v[x + y * img_width, 4] = y
        return img_pixels_v[:, :3], img_width, img_height

    @staticmethod
    def photo_color_invariant(rgb):
        color_invariant = ASARRAY([POWER(rgb[j[0]] - rgb[j[1]], 2) for j in list(COMBINATIONS(range(3), 2))])
        if SUM(color_invariant) == 0:
            return [0, 0, 0]
        return color_invariant/SUM(color_invariant)
