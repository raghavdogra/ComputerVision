# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    # Write histogram equalization here
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_in)

    hi, bins = np.histogram(v, 256)
    cdf = np.cumsum(hi)
    cdf_normalized = cdf * float(hi.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    vnew = cdf[v]
    img_in = cv2.merge((h, s, vnew))

    img_in = cv2.cvtColor(img_in, cv2.COLOR_HSV2BGR)

    img_out = img_in  # Histogram equalization result

    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def displayimage(img):
    cv2.imshow("ss", img)
    cv2.waitKey()


def low_pass_filter(img_in):
    rows, cols, layers = img_in.shape
    img_out = np.zeros((rows, cols, layers))
    # Low pass filter result
    # Write low pass filter here
    dft0 = cv2.dft(np.float32(img_in[:, :, 0]), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft1 = cv2.dft(np.float32(img_in[:, :, 1]), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(img_in[:, :, 2]), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift0 = np.fft.fftshift(dft0)
    dft_shift1 = np.fft.fftshift(dft1)
    dft_shift2 = np.fft.fftshift(dft2)

    crow, ccol = rows / 2, cols / 2

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1

    fshift0 = dft_shift0 * mask
    fshift1 = dft_shift1 * mask
    fshift2 = dft_shift2 * mask
    f_ishift0 = np.fft.ifftshift(fshift0)
    f_ishift1 = np.fft.ifftshift(fshift1)
    f_ishift2 = np.fft.ifftshift(fshift2)

    img_back0 = cv2.idft(f_ishift0)
    img_back1 = cv2.idft(f_ishift1)
    img_back2 = cv2.idft(f_ishift2)
    img_out[:, :, 0] = cv2.magnitude(img_back0[:, :, 0], img_back0[:, :, 1])
    img_out[:, :, 1] = cv2.magnitude(img_back1[:, :, 0], img_back1[:, :, 1])
    img_out[:, :, 2] = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

    img_out = np.divide(img_out * 255, np.max(img_out))

    return True, img_out


def high_pass_filter(img_in):

    # Write high pass filter here
    rows, cols, layers = img_in.shape
    img_out = np.zeros((rows, cols, layers))

    dft0 = cv2.dft(np.float32(img_in[:, :, 0]), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft1 = cv2.dft(np.float32(img_in[:, :, 1]), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(img_in[:, :, 2]), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift0 = np.fft.fftshift(dft0)
    dft_shift1 = np.fft.fftshift(dft1)
    dft_shift2 = np.fft.fftshift(dft2)

    crow, ccol = rows / 2, cols / 2

    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0

    fshift0 = dft_shift0 * mask
    fshift1 = dft_shift1 * mask
    fshift2 = dft_shift2 * mask
    f_ishift0 = np.fft.ifftshift(fshift0)
    f_ishift1 = np.fft.ifftshift(fshift1)
    f_ishift2 = np.fft.ifftshift(fshift2)

    img_back0 = cv2.idft(f_ishift0)
    img_back1 = cv2.idft(f_ishift1)
    img_back2 = cv2.idft(f_ishift2)
    img_out[:, :, 0] = cv2.magnitude(img_back0[:, :, 0], img_back0[:, :, 1])
    img_out[:, :, 1] = cv2.magnitude(img_back1[:, :, 0], img_back1[:, :, 1])
    img_out[:, :, 2] = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

    img_out = np.divide(img_out * 255, np.max(img_out))
    return True, img_out


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)


def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def deconvolution(img_in):
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    # Write deconvolution codes here
    imf = ft(img_in, (img_in.shape[0], img_in.shape[1]))
    gkf = ft(gk, (img_in.shape[0],
                  img_in.shape[1]))  # so we can multiple easily
    imconvf = imf / gkf
    blurred = ift(imconvf)

    img_out = blurred * 255  # Deconvolution result
    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)
    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================
# Laplacian pyramid


def laplacian_pyramid_blending(img_in1, img_in2):
    img_in1 = img_in1[:, :img_in1.shape[0]]
    img_in2 = img_in2[:img_in1.shape[0], :img_in1.shape[0]]

    # Gaussian pyramid
    G = img_in1.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)  # Blur and downsample in one
        gpA.append(G)

    G = img_in2.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)  # Blur and downsample in one
        gpB.append(G)

# generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
# generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

# Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)


# now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    img_out = ls_  # Blending result

    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
