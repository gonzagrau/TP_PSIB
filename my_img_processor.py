import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from typing import Self

class My_Image(np.ndarray):
    def __new__(cls, input_val: np.ndarray | str):
        """
        Creates a new image object from either a numpy array or a filepath
        """

        if type(input_val) != str:
            input_array = input_val
        else:
            input_array = cv2.imread(input_val, 0)

        if input_array.ndim == 3:
            input_array = input_array.mean(axis=2)
        
        obj = np.asarray(input_array).view(cls)
        return obj

    def get_hist(self,
                 vmin: int=0,
                 vmax: int=255,
                 N_bins: int=0,
                 plot: bool=True) -> tuple[np.ndarray, np.ndarray]:
        """
        vmin: minimum intensity
        vmax: maximum intensity
        N_bins: amount of values to calculate their frequency

        return: bins, hist
        """
        if N_bins == 0:
            N_bins = vmax - vmin + 1

        bins = np.linspace(vmin, vmax, N_bins, dtype='uint8')
        img_ravel = self.ravel()
        hist = np.zeros(N_bins)

        for px in img_ravel:
            hist[bins == px] += 1

        if plot:
            fig, ax = plt.subplots()
            ax.bar(bins, hist, width=1)
            plt.show()

        return bins, hist   

    def zero_pad(self, d: int) -> Self:
        """
        d: size of kernel

        return: zero padded image suitable for convoluting with a d-sized kernel
        """
        N, M = self.shape
        pad_num = d//2
        img_pad = np.zeros((N + 2*pad_num, M + 2*pad_num))
        img_pad[pad_num:-pad_num, pad_num:-pad_num] = self
        return My_Image(img_pad)  

    def expand_pad(self, d: int) -> Self:
        """
        img: image represented as a numpy array of dimensions NxM
        d: size of kernel

        return: zero padded image suitable for convoluting with a d-sized kernel
        """
        img_pad = self.zero_pad(d)
        pad_num = d//2
        north = pad_num - 1
        south = pad_num + self.shape[0]
        east =  pad_num + self.shape[1]
        west = pad_num - 1

        for layer in range(pad_num):
            # fill the borders
            img_pad[north, west + 1 : east] = img_pad[north + 1, west + 1 : east]
            img_pad[south, west + 1 : east] = img_pad[south - 1, west + 1 : east]
            img_pad[north + 1: south, east] = img_pad[north + 1: south, east - 1]
            img_pad[north + 1: south, west] = img_pad[north + 1: south, west + 1]

            # then the corners
            img_pad[north, west] = img_pad[north + 1, west + 1]
            img_pad[north, east] = img_pad[north + 1, east - 1]
            img_pad[south, west] = img_pad[south - 1, west + 1]
            img_pad[south, east] = img_pad[south - 1, east - 1]

            # update borders
            north -= 1
            south += 1
            west -= 1
            east += 1

        return My_Image(img_pad)

    def filter_w_kernel(self: np.ndarray,
                   kernel: np.ndarray, *,
                   pad_mode: str='zero',
                   output_dtype: np.number | None = None) -> Self:
        """
        kernel: kernel matrix to convolute with the image
        pad_mode: chosen padding method, either 'zero' or 'expand'
        output_dtype: specifies image depth for the output

        return: the filtered image after convoluting with the kernel
        """
        if output_dtype is None:
            output_dtype = self.dtype

        d = kernel.shape[0]
        N, M = self.shape
        pad_num = d//2
        dtype_data = np.iinfo(output_dtype)
        MIN_VAL, MAX_VAL = dtype_data.min, dtype_data.max

        if pad_mode == 'zero':
            padded_img = self.zero_pad(d)
        elif pad_mode =='expand':
            padded_img = self.expand_pad(d)
        else:
            raise ValueError(f"'{pad_mode}' is not a valid padding mode," +
                            "should  be either 'zero' or 'expand'")

        filtered_img = np.zeros_like(self, dtype=output_dtype)
        for i in range(pad_num, N):
            for j in range(pad_num, M):
                window = padded_img[i - pad_num : i + pad_num + 1,
                                    j - pad_num : j + pad_num + 1]
                new_value = np.sum(window*kernel)

                # map the output value to the valid range (MIN_VAL, MAX_VAL)
                if new_value < MIN_VAL:
                    new_value = MIN_VAL
                elif new_value > MAX_VAL:
                    new_value = MAX_VAL

                filtered_img[i, j] = new_value

        return My_Image(filtered_img)
    
    def median_filter(self, window_size: float, pad_mode: str='zero') -> Self:
        """
        window_size: dimension of the window to convolve
        pad_mode: chosen padding method, either 'zero' or 'expand'

        return: the filtered image after applying the filter
        """
        N, M = self.shape
        pad_num = window_size//2

        if pad_mode == 'zero':
            padded_img = self.zero_pad(window_size)
        elif pad_mode =='expand':
            padded_img = self.expand_pad(window_size)
        else:
            raise ValueError(f"'{pad_mode}' is not a valid padding mode," +
                            "should  be either 'zero' or 'expand'")

        filtered_img = np.zeros_like(self)
        for i in range(pad_num, N):
            for j in range(pad_num, M):
                window = padded_img[i - pad_num : i + pad_num + 1,
                                    j - pad_num : j + pad_num + 1]
                filtered_img[i, j] = np.median(window)

        return My_Image(filtered_img)

    def canny_filter(self, 
                     t1: int, 
                     t2: int, 
                     gauss_kernel: np.ndarray | None = None,
                     show_proggress: bool = False) -> Self:
        """
        Performs the canny border detection algorithm

        Args:
            t1 (int): lower threshold
            t2 (int): upper threshold
            gauss_kernel (np.ndarray | None, optional): prefilter to apply. Defaults to None.

        Returns:
            My_Image: filtered image 
        """
        # Step 0: prefilter
        print(('Prefiltering...' + '\n')*show_proggress, end='')
        if gauss_kernel is not None:
            img = self.filter_w_kernel(gauss_kernel)
        else:
            img = self

        # Step 1: get gradient
        print(('Getting gradient...' + '\n')*show_proggress, end='')
        KSV = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        KSH = KSV.T
        Gx = img.filter_w_kernel(KSH, output_dtype=np.int16)
        Gy = img.filter_w_kernel(KSV, output_dtype=np.int16)

        G = np.zeros_like(Gx)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                G[i, j] = np.sqrt(Gx[i, j]**2 + Gy[i, j]**2)

        #Step 2: Get gradiente angle
        # IMPORTANT: The angle is rounded to one of four angles representing
        # vertical, horizontal, and the two diagonals (0°, 45°, 90°, and 135°).
        # An edge direction falling in each color region will be set to a specific angle
        # value, for instance, θ in [0°, 22.5°] or [157.5°, 180°] maps to 0°.
        print(('Getting angles...' + '\n')*show_proggress, end='')
        first_ang = 0
        last_ang = 135
        step = 45
        valid_angles = np.arange(first_ang, last_ang + 1, step)
        rad_2_deg = 180/np.pi

        ang = np.zeros_like(Gx, dtype='uint8')
        for i in range(Gx.shape[0]):
            for j in range(Gx.shape[1]):
                if Gy[i, j] == Gx[i, j] == 0:
                    continue
                atan = np.arctan2(Gy[i, j], Gx[i, j])
                if atan < 0: #importa direccion, no sentido
                    atan += np.pi
                    atan *= rad_2_deg
                if (first_ang <= atan < first_ang + step/2 or atan > last_ang + step/2):
                # caso particular 180 = 0
                    ang[i, j] = first_ang
                else:
                    for valid_angle in valid_angles[1:]:
                        if np.abs(atan - valid_angle) < step/2:
                            ang[i, j] = valid_angle
                            break

        
        # Step 3: Selective suppression
        print(('Applying selective suppression...' + '\n')*show_proggress, end='')
        def _0_ang_check(x: np.ndarray) -> bool:
            """
            checks for changes in the 0° for a 3x3 window of an array
            """
            return x[1, 0] > x[1, 1] or x[1, 1] < x[1, -1]


        def _45_ang_check(x: np.ndarray) -> bool:
            """
            checks for changes in the 45° for a 3x3 window of an array
            """
            return x[1, 1] < x[0, -1] or x[-1, 0] > x[1, 1]


        def _90_ang_check(x: np.ndarray) -> bool:
            """
            checks for changes in the 90° for a 3x3 window of an array
            """
            return x[0, 1] > x[1, 1] or x[1, 1] < x[-1, 1]


        def _135_ang_check(x: np.ndarray) -> bool:
            """
            checks for changes in the 135° for a 3x3 window of an array
            """
            return x[0, 0] > x[1, 1] or x[1, 1] < x[-1, -1]

        ang_func_dict = {0:_0_ang_check, 45:_45_ang_check,
                        90:_90_ang_check, 135:_135_ang_check}
        
        N, M = img.shape
        dtype_data = np.iinfo(img.dtype)
        MIN_VAL, MAX_VAL = dtype_data.min, dtype_data.max

        supressed_img = np.zeros_like(G)
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                window = G[i - 1 : i + 1 + 1,
                        j - 1 : j + 1 + 1]
                angle = ang[i, j]
                check_func = ang_func_dict[angle]
                if check_func(window):
                    supressed_img[i, j] = MIN_VAL
                else:
                    supressed_img[i, j] = G[i, j]

        # Step 4: Histeresis
        print(('Applying the histeresis...' + '\n')*show_proggress, end='')
        histeresis_img = np.zeros_like(img)
        indeces = []
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                if supressed_img[i, j] > t2:
                    indeces.append((i, j))
                    histeresis_img[i, j] = MAX_VAL

            for (i, j) in indeces:
                window_supp = supressed_img[i - 1 : i + 1 + 1,
                                            j - 1 : j + 1 + 1]
                window[window_supp > t1] = 255
                histeresis_img[i - 1 : i + 1 + 1, j - 1 : j + 1 + 1] = window

        return My_Image(histeresis_img)

    def equalize_hist(self) -> Self:
        """
        Applies the histograme equalizing algorithm
        """
        # define constants
        N, M = self.shape
        dtype_data = np.iinfo(self.dtype)
        L = dtype_data.max - dtype_data.min + 1
        k_min = self.min()

        # get original histogram and integrate it
        bins, hist = self.get_hist(plot=False)
        hist_cdf = np.cumsum(hist)
        cdf_min = hist_cdf[k_min]

        # define the transformation, and apply it
        hist_eq = np.round(((hist_cdf - cdf_min)/(M*N - cdf_min))*(L - 1))
        return My_Image(hist_eq[self])


def main():
    imagen1 = My_Image('img25.png')
    bins, hist_orig = imagen1.get_hist(plot=False)
    cdf_orig = hist_orig.cumsum()
    img_eq = imagen1.equalize_hist()
    bins, hist_eq = img_eq.get_hist(plot=False)
    cdf_eq = hist_eq.cumsum()


    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs[0, 0].imshow(imagen1, vmin=0, vmax=255, cmap='gray')
    axs[0, 0].set_title('Imagen original')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img_eq, vmin=0, vmax=255, cmap='gray')
    axs[0, 1].set_title('Imagen ecualizada')
    axs[0, 1].axis('off')

    axs[1, 0].bar(bins, hist_orig, width=1)
    axs[1, 0].plot(bins, cdf_orig, color='red')
    axs[1, 0].set_title('Histograma original')

    axs[1, 1].bar(bins, hist_eq, width=1)
    axs[1, 1].plot(bins, cdf_eq, color='red')
    axs[1, 1].set_title('Histograma ecualizado')

    plt.show()



if __name__ == '__main__':
    main()