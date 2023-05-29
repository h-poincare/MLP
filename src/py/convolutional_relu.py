import numpy as np 
from fully_connected import oneHotEncode
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 



Xdata = np.load("data/train.npy")
ydata = np.load("data/train_labels.npy")

ylabels = oneHotEncode(ydata)
y = ylabels.T
# Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, y, test_size=0.3, random_state=42)

# new_data = np.zeros((Xdata.shape[0], 11*11))

def relu(x):
    return max(0, np.max(x)) 

# #2D CONVOLUTION + RELU LAYER 
# TODO: ADD PADDING FOR CONV 
for row_num,_ in enumerate(Xdata):

    print(f"FEATURE ENGINEERING ITERATION : {row_num}")
    filter1 = np.random.randn(3,3)
    # rand_row = np.random.randint(0, Xtrain.shape[0])
    x = Xdata[row_num, :].reshape(28, 28)

    #example of convolution using filter1 (no padding.. stride=1)
    post_conv = []
    stride=2
    i=0
    while i + filter1.shape[0] <= x.shape[0]:
        row_conv = []
        j=0
        while j + filter1.shape[1] <= x.shape[1]:
            sub_mat = x[i:i+filter1.shape[0], j:j+filter1.shape[1]]
            conv = np.multiply(sub_mat, filter1)
            convolved = np.sum(conv)
            row_conv.append(convolved)
            if j + filter1.shape[1] + stride > x.shape[1]:
                post_conv.append(row_conv)
                break
            else:
                j+=stride
        i+=stride


    #compare original plot vs convolved plot
    post_conv = np.array(post_conv)
    relu_func = lambda x: relu(x+1) 
    relu_vec = np.vectorize(relu_func)
    post_conv = relu_vec(post_conv)
    # print(np.array(post_conv).shape)
    # plt.imshow(x, interpolation="nearest")
    # plt.title("ORIGINAL")
    # plt.show()
    # plt.imshow(post_conv, interpolation="nearest")
    # plt.title("CONVOLVED")
    # plt.show()

    #MAX POOLING APPLIED - STRIDE=1, PADDING=0, F=3
    max_pooling = np.zeros((3,3))
    post_pooling = []
    i=0
    stride=1
    while i + max_pooling.shape[0] <= post_conv.shape[0]:
        row_pooling = []
        j=0
        while j + max_pooling.shape[1] <= post_conv.shape[1]:
            sub_mat = post_conv[i:i+max_pooling.shape[0], j:j+max_pooling.shape[1]]
            max_val = np.max(sub_mat)
            row_pooling.append(max_val)

            if j + max_pooling.shape[1] + stride > post_conv.shape[1]:
                post_pooling.append(row_pooling)
                break
            else:
                j+=stride
        i+=stride

    #view image post conv + pooling 
    post_pooling = np.array(post_pooling)
    # print(np.array(post_pooling).shape)
    # plt.imshow(post_pooling, interpolation="nearest")
    # plt.title("CONVOLVED + POOLING")
    # plt.show()

    # NEXT:
    #       - PADDING (missing)
    #       - MULTIPLE CHANNELS 
    #       - HOW FILTERS RELATE TO FC LAYERS

    x_i_flat = post_pooling.reshape(1, post_pooling.shape[0]*post_pooling.shape[1])

    if row_num == 0:
        x_flat = x_i_flat
    else:
        x_flat = np.concatenate((x_flat, x_i_flat),axis=0)

np.save("data/Xdata_Conv.npy", x_flat)



