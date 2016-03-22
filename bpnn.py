import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)



class BPNN(object):
    '''三层神经网络'''
    def __init__(self, sizes):
        super().__init__()
        self.ni = sizes[0]
        self.nh = sizes[1]
        self.no = sizes[2]

    # Helper function to evaluate the total loss on the dataset
    def calculate_loss(self, X, y, model):
        num_examples = len(X)
        lamda = 0.01 # regularization strength
        
        Wi, bh, Wh, bo = model['Wi'], model['bh'], model['Wh'], model['bo']
        # Forward propagation to calculate our predictions
        neth = np.dot(X, Wi) + bh
        lh = np.tanh(neth)
        neto = np.dot(lh, Wh) + bo
        lo = np.exp(neto)
        probs = lo / np.sum(lo, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += lamda/2 * (np.sum(np.square(Wi)) + np.sum(np.square(Wh)))
        return 1./num_examples * data_loss
        
    # 预测
    def predict(self, x):
        model = self.model
        Wi, bh, Wh, bo = model['Wi'], model['bh'], model['Wh'], model['bo']
        # 前向传播
        neth = np.dot(x, Wi) + bh
        lh = np.tanh(neth)
        neto = np.dot(lh, Wh) + bo
        lo = np.exp(neto)
        probs = lo / np.sum(lo, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
        
    # 训练
    def fit(self, X, y, print_loss=False):
        num_examples = len(X) # training set size
        
        # 初始化参数
        Wi = np.random.randn(self.ni, self.nh) / np.sqrt(self.ni)
        bh = np.zeros((1, self.nh))
        Wh = np.random.randn(self.nh, self.no) / np.sqrt(self.nh)
        bo = np.zeros((1, self.no))

        # Gradient descent parameters (I picked these by hand)
        epsilon = 0.01 # learning rate for gradient descent
        lamda = 0.01 # regularization strength 
        
        # This is what we return at the end
        model = {}
        
        # 训练 -- 批量梯度下降
        for i in range(2000):

            # 前向传播
            neth = np.dot(X, Wi) + bh
            lh = np.tanh(neth)
            neto = np.dot(lh, Wh) + bo
            lo = np.exp(neto)
            probs = lo / np.sum(lo, axis=1, keepdims=True)

            # 后向传播
            delta3 = probs
            delta3[range(num_examples), y] -= 1
            dW2 = np.dot(lh.T, delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = np.dot(delta3, Wh.T) * (1 - np.power(lh, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # 添加正则化项(b1和b2没有正规化)
            dW2 += lamda * Wh
            dW1 += lamda * Wi

            # 更新权值、偏置
            Wh += -epsilon * dW2
            Wi += -epsilon * dW1
            
            bo += -epsilon * db2
            bh += -epsilon * db1
            
            
            # Assign new parameters to the model
            model = { 'Wi': Wi, 'bh': bh, 'Wh': Wh, 'bo': bo}
            
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 100 == 0:
              print("Loss after iteration %i: %f" %(i, self.calculate_loss(X, y, model)))
        
        self.model = model


if __name__ == '__main__':
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()



    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    plot_decision_boundary(lambda X: clf.predict(X))
    plt.show()



    nn = BPNN([2,3,2])
    # Build a model with a 3-dimensional hidden layer
    nn.fit(X, y, print_loss=True)
    plot_decision_boundary(lambda X: nn.predict(X))
    plt.show()
