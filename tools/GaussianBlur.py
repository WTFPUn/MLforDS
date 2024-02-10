import numpy as np
import cv2


class GaussianBlur:
    def __init__(self, kernel_size: int, sigma: int):
        """
        kernel_size must be odd and square
        """
        self.kernel_size = np.array(kernel_size)
        self.sigma = np.array(sigma)
        self.C = self.__C()
        print(self.C)
        self.kernel = self.__generate_kernel()

    def __generate_kernel(self):
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                # Adjust indices to be relative to center
                i_adjusted = i - center
                j_adjusted = j - center
                kernel[i][j] = self.__G_ij(i_adjusted, j_adjusted)

        return kernel

    def __C(self):
        sum_of_sum = np.array(
            [
                self.__g_ij(i, j)
                for i in range(-self.kernel_size, self.kernel_size + 1)
                for j in range(-self.kernel_size, self.kernel_size + 1)
            ]
        )
        return sum_of_sum.sum()

    def __g_ij(self, i: int, j: int):
        return np.exp(-((i**2 + j**2)) / (2 * self.sigma**2))

    def __G_ij(self, i: int, j: int):
        return (self.__g_ij(i, j) / self.__g_ij(0, 0)) / self.C

    def filter(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, kernel)


if __name__ == "__main__":
    gau = GaussianBlur(5, 3)
    print(gau.kernel)
    print(gau.kernel.shape)

    import matplotlib.pyplot as plt

    plt.imshow(gau.kernel, cmap="gray")
