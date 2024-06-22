# Gaussian Blur

## Project Description
This project demonstrates the application of Gaussian blur using CUDA and OpenCV. It can be used as a template for other CUDA projects requiring image processing.

## Code Organization

```bin/```
This folder the executable gaussian_blur which performs a gaussian blur on the dataset of .tiff images.

```data/```
This folder holds all the images as input data:
The Sequences volume contains 69 images in 4 sequences. Three sequences consists of 16, 32, and 11 256x256 images. One sequence consists of 10 512x512 images.

Sequence 6.2 consists of 32 images but only the first 16 appear to be a true motion sequence. Image 17 through 32 show some motion but not in any clear direction. They are included in the database only because they have been part of it for several years.

```lib/```
This folder is for libraries that are not installed via the Operating System-specific package manager. Currently, it is empty as all required libraries are installed via the package manager.

```src/```
The source code for the project is placed here in a hierarchical fashion, as appropriate. The primary source file for this project is gaussian_blur.cu..

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
it has rudimentary scripts for building your project's code in an automatic fashion.

