#!/usr/bin/env python
# coding: utf-8

# Import libraries
from PIL import Image, ImageStat, ImageFilter, ImageOps
import numpy as np
import streamlit as st

def main():
    selected_box = st.sidebar.selectbox('Select from dropdown', ('Underwater Image Enhancer', 'About the App'))   
    if selected_box == 'About the App':
        about() 
    if selected_box == 'Underwater Image Enhancer':
        image_enhancer()

def about():
    st.title("Welcome!")
    st.caption("Underwater Image Enhancement Web App")
    with st.expander("Abstract"):
        st.write("""Underwater images find application in various fields, like marine research, inspection of
                aquatic habitat, underwater surveillance, identification of minerals, and more. However,
                underwater shots are affected a lot during the acquisition process due to the absorption
                and scattering of light. As depth increases, longer wavelengths get absorbed by water;
                therefore, the images appear predominantly bluish-green, and red gets absorbed due to
                higher wavelength. These phenomenons result in significant degradation of images due to
                which images have low contrast, color distortion, and low visibility. Hence, underwater
                images need enhancement to improve the quality of images to be used for various
                applications while preserving the valuable information contained in them.""")
    with st.expander("Block Diagram"):
        st.write("Please upload the block diagram image if available.")
        block_diagram = st.file_uploader("Upload Block Diagram", type=["jpg", "png"], key="block_diagram")
        if block_diagram:
            st.image(Image.open(block_diagram), use_container_width=True)
    with st.expander("Results On Sample Images"):
        st.write("Please upload sample result images to display.")
        result1 = st.file_uploader("Upload Result Image 1", type=["jpg", "png"], key="result1")
        if result1:
            st.image(Image.open(result1), use_container_width=True)
        result2 = st.file_uploader("Upload Result Image 2", type=["jpg", "png"], key="result2")
        if result2:
            st.image(Image.open(result2), use_container_width=True)
    with st.expander(" "):
        pass

def image_enhancer():
    st.header("Underwater Image Enhancement Web App")
    file = st.file_uploader("Please upload a RGB underwater image file", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        if image.mode != 'RGB':
            st.text("Please upload RGB image")
        else:
            st.text("Uploaded Image")
            st.image(image, use_container_width=True)
            imtype = st.radio("Select one", ('Greenish Image', 'Bluish Image'))
            if imtype == "Greenish Image":
                flag=0
            else:
                flag=1
            if(st.button("Enhance Uploaded Image")):
                pcafused, averagefused = underwater_image_enhancement(image, flag)
                st.text("Enhanced Image Using PCA Based Fusion")
                st.image(pcafused, use_container_width=True)
                st.text("Enhanced Image Using Averaging Based Fusion")
                st.image(averagefused, use_container_width=True)

def compensate_RB(image, flag):
    imager, imageg, imageb = image.split()
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    x, y = image.size
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = (imageR[i][j] - minR) / (maxR - minR)
            imageG[i][j] = (imageG[i][j] - minG) / (maxG - minG)
            imageB[i][j] = (imageB[i][j] - minB) / (maxB - minB)
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    if flag == 0:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
                imageB[i][j] = int((imageB[i][j] + (meanG - meanB) * (1 - imageB[i][j]) * imageG[i][j]) * maxB)
        for i in range(0, y):
            for j in range(0, x):
                imageG[i][j] = int(imageG[i][j] * maxG)
    if flag == 1:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
        for i in range(0, y):
            for j in range(0, x):
                imageB[i][j] = int(imageB[i][j] * maxB)
                imageG[i][j] = int(imageG[i][j] * maxG)
    compensateIm = np.zeros((y, x, 3), dtype="uint8")
    compensateIm[:, :, 0] = imageR
    compensateIm[:, :, 1] = imageG
    compensateIm[:, :, 2] = imageB
    return Image.fromarray(compensateIm)

def gray_world(image):
    imager, imageg, imageb = image.split()
    imagegray = image.convert('L')
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)
    x, y = image.size
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = np.mean(imageGray)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = int(imageR[i][j] * meanGray / meanR)
            imageG[i][j] = int(imageG[i][j] * meanGray / meanG)
            imageB[i][j] = int(imageB[i][j] * meanGray / meanB)
    whitebalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whitebalancedIm[:, :, 0] = imageR
    whitebalancedIm[:, :, 1] = imageG
    whitebalancedIm[:, :, 2] = imageB
    return Image.fromarray(whitebalancedIm)

def sharpen(wbimage, original):
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)
    smoothedr, smoothedg, smoothedb = smoothed_image.split()
    imager, imageg, imageb = wbimage.split()
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    smoothedR = np.array(smoothedr, np.float64)
    smoothedG = np.array(smoothedg, np.float64)
    smoothedB = np.array(smoothedb, np.float64)
    x, y = wbimage.size
    for i in range(y):
        for j in range(x):
            imageR[i][j] = 2 * imageR[i][j] - smoothedR[i][j]
            imageG[i][j] = 2 * imageG[i][j] - smoothedG[i][j]
            imageB[i][j] = 2 * imageB[i][j] - smoothedB[i][j]
    sharpenIm = np.zeros((y, x, 3), dtype="uint8")
    sharpenIm[:, :, 0] = imageR
    sharpenIm[:, :, 1] = imageG
    sharpenIm[:, :, 2] = imageB
    return Image.fromarray(sharpenIm)

def hsv_global_equalization(image):
    hsvimage = image.convert('HSV')
    Hue, Saturation, Value = hsvimage.split()
    equalizedValue = ImageOps.equalize(Value, mask=None)
    x, y = image.size
    equalizedIm = np.zeros((y, x, 3), dtype="uint8")
    equalizedIm[:, :, 0] = Hue
    equalizedIm[:, :, 1] = Saturation
    equalizedIm[:, :, 2] = equalizedValue
    hsvimage = Image.fromarray(equalizedIm, 'HSV')
    rgbimage = hsvimage.convert('RGB')
    return rgbimage

def average_fusion(image1, image2):
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)
    x, y = image1R.shape
    for i in range(x):
        for j in range(y):
            image1R[i][j] = int((image1R[i][j] + image2R[i][j]) / 2)
            image1G[i][j] = int((image1G[i][j] + image2G[i][j]) / 2)
            image1B[i][j] = int((image1B[i][j] + image2B[i][j]) / 2)
    fusedIm = np.zeros((x, y, 3), dtype="uint8")
    fusedIm[:, :, 0] = image1R
    fusedIm[:, :, 1] = image1G
    fusedIm[:, :, 2] = image1B
    return Image.fromarray(fusedIm)

def pca_fusion(image1, image2):
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()
    image1R = np.array(image1r, np.float64).flatten()
    image1G = np.array(image1g, np.float64).flatten()
    image1B = np.array(image1b, np.float64).flatten()
    image2R = np.array(image2r, np.float64).flatten()
    image2G = np.array(image2g, np.float64).flatten()
    image2B = np.array(image2b, np.float64).flatten()
    mean1R = np.mean(image1R)
    mean1G = np.mean(image1G)
    mean1B = np.mean(image1B)
    mean2R = np.mean(image2R)
    mean2G = np.mean(image2G)
    mean2B = np.mean(image2B)
    imageR = np.array((image1R, image2R))
    imageG = np.array((image1G, image2G))
    imageB = np.array((image1B, image2B))
    x, y = imageR.shape
    for i in range(y):
        imageR[0][i] -= mean1R
        imageR[1][i] -= mean2R
        imageG[0][i] -= mean1G
        imageG[1][i] -= mean2G
        imageB[0][i] -= mean1B
        imageB[1][i] -= mean2B
    covR = np.cov(imageR)
    covG = np.cov(imageG)
    covB = np.cov(imageB)
    valueR, vectorR = np.linalg.eig(covR)
    valueG, vectorG = np.linalg.eig(covG)
    valueB, vectorB = np.linalg.eig(covB)
    if valueR[0] >= valueR[1]:
        coefR = vectorR[:, 0] / sum(vectorR[:, 0])
    else:
        coefR = vectorR[:, 1] / sum(vectorR[:, 1])
    if valueG[0] >= valueG[1]:
        coefG = vectorG[:, 0] / sum(vectorG[:, 0])
    else:
        coefG = vectorG[:, 1] / sum(vectorG[:, 1])
    if valueB[0] >= valueB[1]:
        coefB = vectorB[:, 0] / sum(vectorB[:, 0])
    else:
        coefB = vectorB[:, 1] / sum(vectorB[:, 1])
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)
    x, y = image1R.shape
    for i in range(x):
        for j in range(y):
            image1R[i][j] = int(coefR[0] * image1R[i][j] + coefR[1] * image2R[i][j])
            image1G[i][j] = int(coefG[0] * image1G[i][j] + coefG[1] * image2G[i][j])
            image1B[i][j] = int(coefB[0] * image1B[i][j] + coefB[1] * image2B[i][j])
    fusedIm = np.zeros((x, y, 3), dtype="uint8")
    fusedIm[:, :, 0] = image1R
    fusedIm[:, :, 1] = image1G
    fusedIm[:, :, 2] = image1B
    return Image.fromarray(fusedIm)

def underwater_image_enhancement(image, flag):
    st.text("Compensating Red/Blue Channel Based on Green Channel...")
    compensatedimage = compensate_RB(image, flag)
    st.text("White Balancing the compensated Image using Grayworld Algorithm...")
    whitebalanced = gray_world(compensatedimage)
    st.text("Enhancing Contrast of White Balanced Image using Global Histogram Equalization...")
    contrastenhanced = hsv_global_equalization(whitebalanced)
    st.text("Sharpening White Balanced Image using Unsharp Masking...")
    sharpenedimage = sharpen(whitebalanced, image)
    st.text("Performing Average Based Fusion of Sharped Image & Contrast Enhanced Image...")
    averagefused = average_fusion(sharpenedimage, contrastenhanced)
    st.text("Performing PCA Based Fusion of Sharped Image & Contrast Enhanced Image...")
    pcafused = pca_fusion(sharpenedimage, contrastenhanced)
    return pcafused, averagefused

if __name__ == "__main__":
    main()
