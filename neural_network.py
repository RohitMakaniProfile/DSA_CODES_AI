import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42, lr=0.1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1/np.sqrt(input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, 1/np.sqrt(hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    @staticmethod
    def sigmoid(z): 
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def d_sigmoid(a):  
        return a * (1 - a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def d_relu(z):
        return (z > 0).astype(z.dtype)

    def feedforward(self, X):

        z1 = X @ self.W1 + self.b1    
        a1 = self.relu(z1)                
        z2 = a1 @ self.W2 + self.b2        
        a2 = self.sigmoid(z2)              
        return z1, a1, z2, a2

    def predict(self, X):
        _, _, _, a2 = self.feedforward(X)
        return a2

    def loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def train(self, X, y, epochs=1000, verbose_every=100):
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            z1, a1, z2, y_hat = self.feedforward(X)
            L = self.loss(y_hat, y)
            dL_dyhat = (2.0 / n) * (y_hat - y)
            dL_dz2 = dL_dyhat * self.d_sigmoid(y_hat)        
            dL_dW2 = a1.T @ dL_dz2                             
            dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)      

            dL_da1 = dL_dz2 @ self.W2.T                         
            dL_dz1 = dL_da1 * self.d_relu(z1)                  

            dL_dW1 = X.T @ dL_dz1                              
            dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)      

            self.W2 -= self.lr * dL_dW2
            self.b2 -= self.lr * dL_db2
            self.W1 -= self.lr * dL_dW1
            self.b1 -= self.lr * dL_db1

            if verbose_every and epoch % verbose_every == 0:
                print(f"epoch {epoch:5d}  loss={L:.6f}")

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)

    nn = NeuralNetwork(input_dim=2, hidden_dim=4, output_dim=1, lr=0.1)
    nn.train(X, y, epochs=5000, verbose_every=500)

    preds = nn.predict(X)
    print("Predictions:\n", preds)
