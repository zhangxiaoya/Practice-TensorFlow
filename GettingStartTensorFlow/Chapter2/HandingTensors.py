import matplotlib.image as mp_image
import matplotlib.pyplot as plt

filename = "packt.jpg"
input_image = mp_image.imread(filename)

print("input dim = {}".format(input_image.ndim))
print("input.shape = {}".format(input_image.shape))

plt.imshow(input_image)
plt.show()