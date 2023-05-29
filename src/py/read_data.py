import numpy as np
import matplotlib.pyplot as plt
import numpy as np


image_size = 28 # width and length
data_path = "data/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
print(test_data[10])


fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(train_imgs[-1].reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")


plt.show()
# np.save(train_imgs, "data/train_data.npy")

print(type(train_imgs))

np.save("data/train.npy", train_imgs)
np.save("data/test.npy", test_imgs)
np.save("data/train_labels.npy", train_labels)
np.save("data/test_labels.npy", test_labels)
