# Robust Optical Flow Computation: A Higher-Order Differential Approach


## Overview

This is a Python implementation of the optical flow computation derived using the second-order Taylor series approximation. Given an image sequence as input, this program calculates flow vectors (u, v) representing pixel motion between consecutive frames.

## Example usage: Test image and corresponding result image

<p align="center">
  <img src="/Datasets/test images/car1.jpg" alt="Test Image" width="45%" />
  <img src="/results/Car/Optical_Flow.png" alt="Result Image" width="45%" />
</p>

## Features

- Consider two input images of consecutive frames, and compute optical flow.
- Utilizes the Horn-Schunck method for optical flow estimation.
- Regularization constant (alpha) to control the smoothness of the output flow vectors.
- Easily adjustable parameters for customized results.

## Installation

Before running the program, ensure you have Python (>= 3.6) installed on your system. Clone this repository and install the required dependencies:

```bash
git clone https://github.com/chanukaravishan/Optical-flow-plus.git
cd Optical-flow-plus
pip install -r requirements.txt
```

## Usage

To compute optical flow for your image sequences, follow these steps:

1. Prepare your image sequences: Make sure you have consecutive image frames in a directory.

2. Open the `OF++.py` script and set the `alpha` parameter according to your requirements. The higher the value of `alpha`, the smoother the output flow vectors will be.

3. Run the script by providing the path to your image sequence directory:



## Contributing

Contributions to this project are welcome! Feel free to open issues and submit pull requests to address bugs, add new features, or improve the existing implementation.

## Acknowledgments

Special thanks to the original authors of the Horn-Schunck method for their valuable research and contribution to the computer vision community.
