import numpy as np
import pickle

D = 28 * 28 + 1 # input dimensions + bias
H1 = 200 # size of the first hidden layer
H2 = 100 # size of the second hidden layer
N = 10 # number of classes

learning_rate = 1e-5

model_file = 'model/mnist_model.pkl'
resume = True

def save(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def load(file_name, encoding='bytes'):
    with open(file_name, 'rb') as file:
        return pickle.load(file, encoding=encoding)

def init_model():
    return {
        'w1': np.random.randn(D, H1) / (D + H1),
        'w2': np.random.randn(H1, H2) / (H1 + H2),
        'w3': np.random.randn(H2, N) / (H2 + N)
    }

def preprocess(x):
    x = x.astype(np.float32)
    x -= x.mean()
    x /= x.std()
    return np.append(x, np.ones((len(x), 1)), axis=1) # insert an extra column of ones that will be used to add bias

def forward(x):
    h1 = x @ model['w1'] # the last row of the weight matrix is the bias
    h1[h1 < 0] = 0 # ReLU
    h1[:, -1] = 1 # instead of appending ones just use the last column to add bias, making the effective size of H1 one less

    h2 = h1 @ model['w2']
    h2[h2 < 0] = 0
    h2[:, -1] = 1

    class_scores = h2 @ model['w3']
    pred_y = softmax(class_scores)
    
    return pred_y, h2, h1

def backward(y, pred_y, h2, h1, x):
    dclass_scores = pred_y - y # the gradient of the softmax and cross entropy loss functions
    dw3 = h2.T @ dclass_scores

    dh2 = dclass_scores @ model['w3'].T
    dh2[:, -1] = 0  # drop the gradient of the column used to add bias
    dh2[h2 <= 0] = 0
    dw2 = h1.T @ dh2

    dh1 = dh2 @ model['w2'].T
    dh1[:, -1] = 0
    dh1[h1 <= 0] = 0
    dw1 = x.T @ dh1

    return {'w1': dw1, 'w2': dw2, 'w3': dw3}

def softmax(x):
    x -= x.max() # for numerical stability
    np.exp(x, x)
    x /= x.sum(axis=1, keepdims=True)
    return x

def loss(y, pred_y):
    return (y * -np.log(pred_y)).sum() / len(y)

def train(epoch):
    pred_y, h2, h1 = forward(train_x)
    print(f'[ep{epoch}] loss: {loss(train_y, pred_y)}')

    gradient = backward(train_y, pred_y, h2, h1, train_x)
    for k, v in gradient.items():
        model[k] -= learning_rate * v

def test():
    pred_y, *_ = forward(test_x)
    print(f'[test] loss: {loss(test_y, pred_y)}')

    pred_labels = pred_y.argmax(axis=1)
    accuracy = test_y[range(len(test_y)), pred_labels].sum() / len(test_y)
    print(f'[test] accuracy: {round(accuracy * 100, 2)}%')

np.seterr(all='raise')

(train_data, train_labels), _, (test_data, test_labels) = load('data/mnist_data.pkl')

train_x = preprocess(train_data)
train_y = np.eye(N)[train_labels]

test_x = preprocess(test_data)
test_y = np.eye(N)[test_labels]

model = load(model_file) if resume else init_model()

for epoch in range(1000):
    train(epoch)
    if epoch % 10 == 0:
        test()
        save(model, model_file)
        print('[model saved]')
