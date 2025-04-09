import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_dct as dct  # Assuming torch_dct is properly installed

# Generate a simple image: a block of white on black background
image = np.zeros((32, 32))
image[8:24, 8:24] = 1

# Perform DCT
dct_image = dct.dct_2d(torch.tensor(image)).numpy()

# Perform DFT
dft_image = np.fft.fft2(image)
dft_image = np.fft.fftshift(dft_image)  # Shift zero frequency to center

# Zero out all small coefficients and inverse transform
threshold = 5
dct_thresholded = dct_image * (abs(dct_image) > threshold)
dft_thresholded = dft_image * (abs(dft_image) > threshold)

# Inverse DCT
idct_image = dct.idct_2d(torch.tensor(dct_thresholded)).numpy()

# Inverse DFT
idft_image = np.fft.ifft2(np.fft.ifftshift(dft_thresholded)).real

# Plotting
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title("Original Image")
ax[0, 1].imshow(np.log1p(abs(dct_image)), cmap='gray')
ax[0, 1].set_title("DCT Coefficients (Log Scale)")
ax[0, 2].imshow(idct_image, cmap='gray')
ax[0, 2].set_title("Reconstructed Image from DCT")

ax[1, 0].imshow(image, cmap='gray')
ax[1, 0].set_title("Original Image")
ax[1, 1].imshow(np.log1p(abs(dft_image)), cmap='gray')
ax[1, 1].set_title("DFT Coefficients (Log Scale)")
ax[1, 2].imshow(idft_image, cmap='gray')
ax[1, 2].set_title("Reconstructed Image from DFT")

plt.tight_layout()
plt.savefig('compare.png')
plt.show()
