import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def oneHotEncode(data_labels):
    num_labels = np.unique(data_labels).tolist()
    y_vector = np.zeros((len(num_labels), data_labels.shape[0]))
    for i,dl in enumerate(data_labels):
        for j,nl in enumerate(num_labels):
            if (dl == nl):
                y_vector[j, i] = 1
            else:
                y_vector[j, i] = 0
    return y_vector


#class to implement fully connected network with specified architecture 
class Fully_Connected_Network():
    '''Train Multi-layer perceptron fully connected neural network with given network architecture'''
    def __init__(self, n_inputs: int, n_hidden_units: list, n_outputs: int, activation_func: str):
        self.n_inputs = n_inputs
        self. n_hidden_units = n_hidden_units
        self.n_outputs = n_outputs
        self.activation_func = activation_func

        self.net = self.create_net()
        self.weights= self.init_weights(self.net)
        
    #function returns net structure 
    def create_net(self):
        '''Returns list, where each element is a list representing a single layer '''
        net = []
        first_layer = np.zeros((self.n_inputs, 1))
        output_layer = np.zeros((self.n_outputs, 1))
        net.append(first_layer)
        for _,hidden_unit in enumerate(self.n_hidden_units):
            hidden_layer = np.zeros((hidden_unit, 1))
            net.append(hidden_layer)
        net.append(output_layer)

        return net

    #initialization: random gaussian N(0, var=2/#hidden units)
    def init_weights(self, net):
        '''Initialize network weights randomly, scale variance by 2/#hidden units. '''
        num_thetas = len(net) - 1
        weights = []
        for i in range(num_thetas):
            #scale variance: 2/#hidden units... impact
            theta = np.sqrt(2/len(net[i])) * np.random.randn(len(net[i + 1]), len(net[i]) + 1)
            weights.append(theta)
        return weights
    
    def activation(self, x, func, dx=False):
        '''Activation function, supports Sigmoid and Tanh, supports derivative'''
        if func == "sigmoid":
            if dx==True:
                return np.multiply(x, (1 - x))
            else:
                return 1 / (1 + np.exp(-x))
        elif func == "tanh":
            if dx:
                return (1 - self.activation(x, 'tanh')**2)
            else:
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  
        else: 
            print(f"ERROR : ACTIVATION FUNCTION NOT RECOGNIZED {self.activation_func} ")
    
    def feedforward(self, x, j):
        '''Forward propagation through network'''
        one = np.array([[1]])
        self.net[0] = x[[j]].T
        self.net[0] = np.append(one, self.net[0], axis=0)
        for i,theta in enumerate(self.weights):
            z = np.dot(theta, self.net[i])
            a = self.activation(z, self.activation_func, dx=False)
            self.net[i + 1] = a
            if i + 1 < len(self.net)-1:
                self.net[i + 1] = np.append(one, self.net[i + 1], axis=0)

        output = self.net[len(self.net)-1]

        return output
        
    def backprop(self, y, output, j):
        '''Backpropagation algorithm, calculate partial derivative of cost function w.r.t weights'''
        delta_list = []
        self.weightsGrad = [np.zeros((self.weights[t].shape)) for t,_ in enumerate(self.weights)]
        delta = output - y[[j]].T
        delta_list.append(delta)
        for k in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(self.weights[k].T, delta) * self.activation(self.net[k], self.activation_func, dx=True)

            delta = delta[1:len(delta)]
            delta_list.append(delta)
        # compute the gradient of the cost function
        delta_list = list(reversed(delta_list))
        for m,_ in enumerate(self.weightsGrad):
            self.weightsGrad[m] = np.dot(delta_list[m], self.net[m].T)

    def gradient_descent(self, l_rate):
        '''Update network weights using gradient descent algorithm'''
        for i,_ in enumerate(self.weights):
            self.weights[i] += -(l_rate * self.weightsGrad[i])
        
    def accuracy(self, y, output, j):
        '''Check if model prediciton is correct, returns 1 if yes else 0'''
        if np.argmax(y[[j]].T) == np.argmax(output.T, axis=-1):
            return 1
        else:
            return 0

    def train(self, x_train, y_train, x_test, y_test, batch_size, epochs=1_000 
              , l_rate=0.01, viz=False):
        '''Train network on train set over set number of epochs, evaluate on test set'''
        self.epochs = epochs 
        self.batch_size = batch_size
        num_batches = x_train.shape[0] // self.batch_size
        epoch_list = []
        error_hist = []
        x_vals = 0
        for i in range(self.epochs):

            epoch_list.append(x_vals)
            #shuffle 
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                #Forward prop
                output = self.feedforward(x, j)
                # Backprop
                self.backprop(y, output, j)
                # Optimize
                self.gradient_descent(l_rate=l_rate)

            #Make predictions.. evaluate model performance
            #Train 
            train_pred = 0
            for ind,x in enumerate(x_train):
                output = self.feedforward(x_train, ind)
                train_acc_count = self.accuracy(y_train, output, ind)
                train_pred += train_acc_count
                
            train_acc = train_pred / x_train.shape[0]
            error_hist.append(1-train_acc)
            error_hist[i] = 1-train_acc

            print(f"TRAINING: EPOCH {i}  :   TRAIN ACCURACY = {(train_acc):.2%}")
            # if i % 100 == 0: 
            #     print(f"TRAINING: EPOCH {i}  :   TRAIN ACCURACY = {train_acc:.1%}")
            
            #plot training error live 
            if viz == True:
                plt.plot(epoch_list, error_hist, color='blue')
                plt.ylim(0, 1)
                plt.pause(0.000000001)
                plt.title(f"NN: Train Error Over {self.epochs} Epochs")    
                plt.xlabel("Epochs")
            x_vals +=1
        plt.show()

        #Test performance
        test_pred = 0
        for ind,x in enumerate(x_test):
            output = self.feedforward(x_test, ind)
            test_acc_count = self.accuracy(y_test, output, ind)
            test_pred += test_acc_count

        test_acc = test_pred / x_test.shape[0]
        print(f"TEST ACCURACY = {test_acc:.2%}")

        plt.plot(epoch_list, error_hist, color='b')
        plt.ylim(0, 1)
        plt.title(f"NNTrain Error Over {(self.epochs)} Epochs")    
        plt.xlabel("Epochs")
        plt.show()



if __name__ == '__main__':
    #8X8 pixel images
    Xdata = np.load("data/in/digit_images.npy")
    ydata = np.load("data/in/digit_labels.npy")

    ylabels = oneHotEncode(ydata)
    y = ylabels.T
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, y, test_size=0.33, random_state=42)
    nn1 = Fully_Connected_Network(64, [25], 10, "tanh")
    nn1.train(Xtrain, ytrain, Xtest, ytest, batch_size=Xtrain.shape[0], epochs = 2_500)

    # MNIST SUBSET - 28X28 pixel images.. 60K train 10K test 
    # Xdata = np.load("data/train.npy")
    # ydata = np.load("data/train_labels.npy")

    # ylabels = oneHotEncode(ydata)
    # y = ylabels.T
    # Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, y, test_size=0.33, random_state=42)
    # nn2 = Fully_Connected_Network(784, [25], 10, 'tanh')
    # nn2.train(Xtrain, ytrain, Xtest, ytest, batch_size=Xtrain.shape[0], epochs = 5_000)


    #28*28 pixel images.. after applying one 3X3 conv filter, and 3X3 max poooling  --> flatten (28*28 + 13*13 + 11*11)
    # Xdata = np.load("data/Xdata_Conv.npy")
    # ydata = np.load("data/train_labels.npy")

    # ylabels = oneHotEncode(ydata)
    # y = ylabels.T
    # Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, y, test_size=0.33, random_state=42)
    # nn3 = Fully_Connected_Network(121, [25], 10, 'tanh')
    # nn3.train(Xtrain, ytrain, Xtest, ytest, batch_size=Xtrain.shape[0], epochs = 5_000)







