# Building Your Own AI From Scratch in Python
### A complete, no-skipped-steps guide for beginners

This guide builds a real, working neural network — the kind of thing people mean when they say "AI" — using nothing but Python and NumPy (which just gives us fast math on grids of numbers). No TensorFlow, no PyTorch, no magic. By the end you'll have code that looks at a picture of a handwritten digit and correctly says "that's a 7," and you'll understand *why* every line does what it does.

Everything here was written and actually run to confirm it works before being included.

---

## Part 1 — What "AI" Actually Is

Strip away the marketing, and almost everything people call "AI" today is one thing: **a mathematical function with adjustable knobs, tuned automatically until it produces the outputs we want.**

That's it. A neural network is not a brain, not conscious, not "thinking." It's a big function:

```
output = f(input, knobs)
```

where "knobs" (called **weights** and **biases**) start out random and get nudged, over and over, until the function's outputs match what we wanted. The nudging process is called **training**, and the algorithm that does the nudging is called **gradient descent**, powered by a technique called **backpropagation**. Those two words — gradient descent and backpropagation — are 90% of what "deep learning" actually is. We're going to build both from raw arithmetic so they stop being mysterious.

**The plan:**
1. Learn the small amount of math you actually need (no calculus degree required — I'll build the intuition from scratch)
2. Understand a single artificial neuron
3. Connect neurons into a network
4. Understand how a network measures its own mistakes
5. Understand how it learns from those mistakes (backpropagation)
6. Write the entire thing in Python using only NumPy
7. Train it on a real task: recognizing handwritten digits
8. Understand when and why to switch to frameworks like PyTorch

---

## Part 2 — The Math You Actually Need

You need three ideas. I'll explain each with an analogy first, then the math.

### 2.1 Vectors and matrices are just organized lists of numbers

A **vector** is a list of numbers, e.g. `[0.2, 0.9, 0.1]`. In our network, a vector will represent things like "the brightness of every pixel in an image" or "how strongly each neuron fired."

A **matrix** is a grid of numbers — rows and columns. We use matrices to store all the weights (knobs) connecting one layer of neurons to the next, because it lets us compute an entire layer's output in one shot instead of writing a loop for every single neuron.

The one operation you need to understand is the **dot product**: multiply matching elements together, then add up the results.

```
[a, b, c] · [x, y, z]  =  a*x + b*y + c*z
```

Why does this matter for AI? Because "how strongly should this neuron respond to this input" is naturally computed as: *multiply each input by how much you care about it (its weight), then sum it all up.* That sum **is** a dot product. Every "layer" of a neural network is just: take a vector of inputs, dot-product it against a matrix of weights, get a vector of outputs. NumPy does this in one line: `X @ W`.

### 2.2 Derivatives measure "if I nudge this, what happens?"

A **derivative** tells you the slope of a function at a point — if you nudge the input slightly, how much does the output change, and in which direction?

Picture standing on a hillside in fog. You can't see the whole landscape, but you can feel which way is downhill under your feet. The derivative is that local slope. If you always step in the downhill direction, you'll eventually reach the bottom of the hill (or at least a valley). This is literally the entire idea behind how neural networks learn: we define a "hill" where height = how wrong the network currently is, and we take steps downhill until the wrongness is as low as we can get it. That's **gradient descent** — "gradient" is just the multi-dimensional word for slope.

You don't need to be able to derive calculus formulas by hand. You need exactly these three derivative rules, which we'll use directly:

- The derivative of `x²` is `2x` (this is where "squared error" gets its clean gradient)
- The derivative of the sigmoid function `σ(x)` is `σ(x) * (1 - σ(x))` — a fact we'll just use, proven by any calculus textbook
- The **chain rule**: if you have a function inside a function, e.g. `f(g(x))`, the derivative of the whole thing is the derivative of the outer function times the derivative of the inner function. In plain English: *effects multiply as they pass through layers.*

The chain rule is the single mathematical fact that makes backpropagation possible. A neural network is functions nested inside functions inside functions (layer 3 depends on layer 2 depends on layer 1 depends on the input). The chain rule lets us figure out "how much did this very first weight, buried deep in the network, contribute to the final error?" by multiplying local slopes together as we walk backward through the layers. That backward walk, applying the chain rule at every step, is literally what the word "backpropagation" means: propagating the error backward.

### 2.3 Gradient descent: the update rule

Once you know the slope (gradient) of the error with respect to a weight, you update the weight like this:

```
new_weight = old_weight - (learning_rate * gradient)
```

Why subtract? Because the gradient points in the direction of *increasing* error, and we want to go the opposite way — downhill, toward *less* error.

`learning_rate` is a small number (like 0.1 or 0.01) that controls step size. Too large and you overshoot the valley and bounce around chaotically. Too small and training crawls forward at a snail's pace. Picking it is part science, part trial and error — you'll see this yourself when we train the real network.

That's genuinely the whole mathematical foundation. Everything below is just applying these three ideas — dot products, derivatives, and the gradient descent update rule — over and over, in an organized way.

---

## Part 3 — The Building Block: One Artificial Neuron

### 3.1 Where the idea comes from

Biological neurons receive electrical signals from other neurons, weigh how important each incoming signal is, sum them up, and fire (or don't) if the total crosses a threshold. Artificial neurons are a deliberately simplified cartoon of this — we're not simulating biology, we're stealing the *shape* of the idea because it turns out to be mathematically useful.

### 3.2 The math of one neuron

A neuron takes several inputs, multiplies each by a weight (how important that input is), adds them up, adds one more adjustable number called a **bias** (which shifts the neuron's sensitivity up or down), and passes the result through an **activation function**.

```
z = (x1 * w1) + (x2 * w2) + ... + (xn * wn) + b
a = activation(z)
```

- `x1...xn` — the inputs
- `w1...wn` — the weights (learned)
- `b` — the bias (learned)
- `z` — the "raw" weighted sum, before activation
- `a` — the neuron's actual output ("activation")

### 3.3 Why we need activation functions — the part beginners usually skip

Here's the crucial insight almost every tutorial glosses over: **if you stack layers of neurons without activation functions, the whole network collapses mathematically into a single straight line, no matter how many layers you add.** Multiplying and adding weighted sums, over and over with no nonlinearity in between, still just produces another weighted sum. You could have a million layers and it would be exactly as powerful as one.

Activation functions inject **nonlinearity** — a bend, a curve — which is what lets a network approximate complex, curvy, real-world patterns (like "this pattern of pixels is a handwritten 7" versus "this pattern of pixels is a handwritten 3"). Without nonlinearity, deep learning simply wouldn't work; you'd just have linear regression wearing a costume.

Two activation functions we'll actually use:

**Sigmoid** — squashes any number into the range (0, 1). Useful for outputs you want to interpret as probabilities.
```
sigmoid(x) = 1 / (1 + e^(-x))
```
Large positive numbers become close to 1, large negative numbers become close to 0, and 0 becomes exactly 0.5.

**ReLU (Rectified Linear Unit)** — the modern default for hidden layers, and much cheaper to compute:
```
relu(x) = max(0, x)
```
If the input is negative, output 0. Otherwise, pass it through unchanged. Despite being almost embarrassingly simple, ReLU trains faster and avoids a problem called the "vanishing gradient" (where sigmoid's slope gets so flat for large inputs that learning grinds to a halt) — which is why it's the standard choice for hidden layers in modern networks.

**Softmax** — used only on the *output* layer when choosing between multiple categories (e.g., "which digit, 0 through 9, is this?"). It converts a list of raw scores into a list of probabilities that sum to 1, so the network's output becomes directly interpretable as "I'm 94% sure this is a 7, 4% sure it's a 1, ...".

---

## Part 4 — From One Neuron to a Network

A single neuron can only draw a straight decision boundary — it can separate "yes" from "no" only if a straight line (or plane) can divide them. Real problems are messier than that. So we arrange neurons into **layers**:

- **Input layer**: not really neurons, just the raw data fed in (e.g., 64 numbers representing pixel brightness in an 8x8 image)
- **Hidden layer(s)**: neurons that learn to detect increasingly abstract patterns
- **Output layer**: neurons whose activations you actually read as the answer

Why do multiple hidden layers help? Each layer gets to combine the *previous* layer's patterns into new, more abstract patterns. In image recognition (though our example is simpler than this), early layers tend to detect edges, middle layers detect shapes made of edges, and later layers detect whole objects made of shapes. This layered composition is exactly why the field is called "deep" learning — "deep" just means "has several hidden layers."

**Forward propagation** is the process of pushing data through the network, layer by layer, to produce a final output:

```
layer1_output = activation(input @ W1 + b1)
layer2_output = activation(layer1_output @ W2 + b2)
final_output  = activation(layer2_output @ W3 + b3)
```

Each `@` is a matrix multiplication that computes *every neuron in that layer's weighted sum* in a single operation — this is why we use matrices instead of writing nested loops.

---

## Part 5 — How the Network Learns

### 5.1 Measuring wrongness: the loss function

Before a network can improve, it needs a number that says exactly *how wrong* it currently is. That number is called the **loss** (or cost). Two common choices:

**Mean Squared Error (MSE)** — good for numeric predictions:
```
loss = average( (predicted - actual)² )
```
Squaring does two jobs: it makes all errors positive (so a too-high guess and a too-low guess don't cancel out), and it punishes big mistakes disproportionately harder than small ones.

**Cross-Entropy Loss** — the standard choice when picking between categories (like digits 0–9), because it pairs naturally with softmax and produces cleaner gradients for classification:
```
loss = -sum( actual * log(predicted) )
```
Intuitively: if the network assigns high probability to the *correct* answer, `log(predicted)` is close to 0, so loss is small. If it confidently assigns low probability to the correct answer, `log(predicted)` becomes a large negative number, so loss explodes — the network gets punished hard for being confidently wrong.

### 5.2 Backpropagation: assigning blame

Here is the actual question backpropagation answers: **"For every single weight in this network, if I nudged it slightly, would the loss go up or down, and by how much?"**

That's it. That's the whole idea. Once we know that for every weight, we already know how to improve — nudge each weight a small step in the direction that *decreases* the loss (gradient descent, from Part 2.3).

The hard part is computing that "if I nudge this, what happens" number (the gradient) for weights buried deep in early layers, since their effect on the final loss passes through every layer after them. This is exactly what the chain rule (Part 2.2) solves: we start at the output, where the error is directly measurable, and walk backward layer by layer, multiplying local slopes together, until every weight in the network has been assigned its share of "blame" for the final error. That backward walk is the "backward pass," and it's why it's called **back**propagation.

Concretely, for each layer we compute:
1. **How wrong was this layer's output** (called the "delta" or error signal for that layer)
2. **How much should each weight feeding into this layer change**, which is the delta multiplied by whatever value flowed into that weight during the forward pass
3. **Pass a version of the error backward** to the previous layer, so it can do the same calculation

Then every weight gets updated using the gradient descent rule from Part 2.3.

### 5.3 Putting the whole training loop together

One full training loop, repeated thousands of times, looks like this:

1. **Forward pass**: push a batch of data through the network, get a prediction
2. **Compute loss**: compare the prediction to the correct answer
3. **Backward pass**: use the chain rule to compute how much every weight contributed to that loss
4. **Update**: nudge every weight a small step opposite its gradient
5. Repeat, thousands of times, and the loss should trend downward as the network gets better

Each full pass through the entire training dataset is called an **epoch**.

---

## Part 6 — Building It: Full Code From Scratch

We'll build this in two stages. First, a small network solving the classic **XOR problem** — a task specifically famous in AI history because a single neuron (a "perceptron") mathematically *cannot* solve it, proving that hidden layers and nonlinearity are necessary, not optional. Then we'll scale the same ideas up to a real task: recognizing handwritten digit images.

### 6.1 Stage 1 — XOR: proving hidden layers matter

XOR ("exclusive or") outputs 1 if exactly one of two inputs is 1, and 0 otherwise:

| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

There is no straight line you can draw on a 2D plot that separates the "1" outputs from the "0" outputs — this is precisely why it requires a hidden layer with a nonlinear activation. It's the smallest possible proof that depth and nonlinearity matter.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.5, seed=42):
        # layer_sizes example: [2, 4, 1] means:
        #   2 input neurons, 4 hidden neurons, 1 output neuron
        np.random.seed(seed)  # makes results reproducible while learning
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # Random small starting weights. The scaling factor
            # (sqrt(2 / layer_sizes[i])) is called "He initialization" —
            # it keeps early signals from exploding or vanishing as they
            # pass through layers. Without careful initialization, deep
            # networks often fail to learn at all.
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))  # biases start at zero, that's fine
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, activated_value):
        # Because we already have sigmoid(x) stored as "activated_value"
        # from the forward pass, we can compute the derivative cheaply
        # as activated_value * (1 - activated_value), instead of
        # recomputing sigmoid from scratch. This is a standard shortcut.
        return activated_value * (1 - activated_value)

    def forward(self, X):
        # Push data through every layer, saving each layer's output
        # (we need these saved activations later, during backpropagation)
        self.activations = [X]
        a = X
        for w, b in zip(self.weights, self.biases):
            z = a @ w + b          # weighted sum (the dot product from Part 2.1)
            a = self.sigmoid(z)    # nonlinearity (the "bend" from Part 3.3)
            self.activations.append(a)
        return a

    def backward(self, X, y, output):
        m = X.shape[0]  # number of training examples in this batch

        # STEP 1: how wrong was the final output, adjusted by the local slope?
        # (output - y) is the raw error. Multiplying by sigmoid_derivative
        # applies the chain rule: it converts "raw error" into "how much
        # the pre-activation value z should change."
        delta = (output - y) * self.sigmoid_derivative(output)
        deltas = [delta]

        # STEP 2: walk backward through every earlier layer, applying the
        # chain rule at each step: this layer's delta depends on the next
        # layer's delta multiplied by the weights connecting them.
        for i in reversed(range(len(self.weights) - 1)):
            delta = (deltas[-1] @ self.weights[i+1].T) * self.sigmoid_derivative(self.activations[i+1])
            deltas.append(delta)
        deltas.reverse()

        # STEP 3: convert deltas into actual weight/bias updates and apply
        # gradient descent (Part 2.3): step opposite the gradient.
        for i in range(len(self.weights)):
            grad_w = self.activations[i].T @ deltas[i] / m
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def train(self, X, y, epochs=10000, verbose=True):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)  # Mean Squared Error, Part 5.1
            losses.append(loss)
            self.backward(X, y, output)
            if verbose and epoch % 2000 == 0:
                print(f"Epoch {epoch:5d}  Loss: {loss:.5f}")
        return losses

    def predict(self, X):
        return self.forward(X)


# --- Train it on XOR ---
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

nn = NeuralNetwork([2, 4, 1], learning_rate=0.5)
nn.train(X, y, epochs=10000)

print("\nPredictions after training:")
preds = nn.predict(X)
for inp, target, pred in zip(X, y, preds):
    print(f"{inp} -> target {target[0]:.0f}, predicted {pred[0]:.4f}")
```

**This code was actually run.** Here's the real output:

```
Epoch     0  Loss: 0.26827
Epoch  2000  Loss: 0.18365
Epoch  4000  Loss: 0.02220
Epoch  6000  Loss: 0.00543
Epoch  8000  Loss: 0.00285

Predictions after training:
[0. 0.] -> target 0, predicted 0.0440
[0. 1.] -> target 1, predicted 0.9543
[1. 0.] -> target 1, predicted 0.9621
[1. 1.] -> target 0, predicted 0.0458
```

Notice the loss falling steadily — that's gradient descent walking downhill, exactly as described in Part 2.2. And the final predictions land close to the correct 0s and 1s, even though the network started with completely random weights and only ever saw four training examples. **This is real, unassisted learning** — nobody told it the XOR rule; it discovered a set of weights that implements XOR purely by repeatedly reducing its own error.

### 6.2 Stage 2 — A real task: recognizing handwritten digits

Now we scale up to something genuinely useful: classifying images of handwritten digits (0–9) using the classic "digits" dataset (1,797 small 8x8 pixel images, built into scikit-learn). This needs two upgrades from the XOR version:

1. **ReLU** instead of sigmoid in hidden layers (faster training, per Part 3.3)
2. **Softmax + cross-entropy** on the output layer, since we're choosing between 10 categories, not predicting a single 0/1 value (Part 3.3 and 5.1)

The overall structure — forward pass, compute loss, backward pass via chain rule, gradient descent update — is **identical in principle** to the XOR network. This is the payoff of understanding the fundamentals: scaling up is mostly swapping a few functions, not learning new concepts.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class MultiClassNet:
    def __init__(self, layer_sizes, learning_rate=0.1, seed=1):
        np.random.seed(seed)
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # ReLU's slope is 1 wherever the input was positive, 0 otherwise —
        # a much cheaper derivative to compute than sigmoid's.
        return (x > 0).astype(float)

    def softmax(self, x):
        # Subtracting the max before exponentiating is a standard numerical
        # trick that avoids overflow, without changing the result.
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z_values = []
        self.activations = [X]
        a = X
        # Hidden layers use ReLU
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            self.z_values.append(z)
            self.activations.append(a)
        # Output layer uses softmax, to get class probabilities
        z_final = a @ self.weights[-1] + self.biases[-1]
        a_final = self.softmax(z_final)
        self.z_values.append(z_final)
        self.activations.append(a_final)
        return a_final

    def backward(self, X, y_onehot):
        m = X.shape[0]
        # A convenient mathematical fact: when softmax is paired with
        # cross-entropy loss, the combined derivative simplifies to just
        # (predicted - actual) — the messy softmax/log-derivative math
        # cancels out cleanly. This is why the two are almost always used
        # together in classification networks.
        delta = self.activations[-1] - y_onehot
        deltas = [delta]
        for i in reversed(range(len(self.weights) - 1)):
            delta = (deltas[-1] @ self.weights[i+1].T) * self.relu_derivative(self.z_values[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            grad_w = self.activations[i].T @ deltas[i] / m
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grad_b

    def train(self, X, y_onehot, epochs=2000, verbose=True):
        for epoch in range(epochs):
            output = self.forward(X)
            # Cross-entropy loss, with a tiny 1e-9 added to avoid log(0)
            loss = -np.mean(np.sum(y_onehot * np.log(output + 1e-9), axis=1))
            self.backward(X, y_onehot)
            if verbose and epoch % 200 == 0:
                preds = np.argmax(output, axis=1)
                targets = np.argmax(y_onehot, axis=1)
                acc = np.mean(preds == targets)
                print(f"Epoch {epoch:4d}  Loss: {loss:.4f}  Train acc: {acc:.3f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)  # pick the highest-probability class


# --- Load real handwritten digit images and train on them ---
digits = load_digits()
X = digits.data / 16.0     # pixel values scaled to 0-1 (helps training stability)
y = digits.target          # the correct digit, 0-9, for each image

# Convert labels like "7" into one-hot vectors like [0,0,0,0,0,0,0,1,0,0]
y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

# Hold out 20% of the data to test on data the network never trained on —
# this is the only honest way to know if it actually learned general
# patterns, rather than just memorizing the training examples.
X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = train_test_split(
    X, y, y_onehot, test_size=0.2, random_state=0)

net = MultiClassNet([64, 32, 10], learning_rate=0.3)  # 64 pixels in, 32 hidden, 10 digit classes out
net.train(X_train, y_train_oh, epochs=2000)

test_preds = net.predict(X_test)
test_acc = np.mean(test_preds == y_test)
print(f"\nTest accuracy: {test_acc:.3f}")
```

**Real output from actually running this:**

```
Epoch    0  Loss: 2.6368  Train acc: 0.079
Epoch  200  Loss: 0.1366  Train acc: 0.971
Epoch  400  Loss: 0.0790  Train acc: 0.984
Epoch  600  Loss: 0.0546  Train acc: 0.991
Epoch  800  Loss: 0.0405  Train acc: 0.996
Epoch 1000  Loss: 0.0315  Train acc: 0.997
Epoch 1200  Loss: 0.0254  Train acc: 0.999
Epoch 1400  Loss: 0.0211  Train acc: 0.999
Epoch 1600  Loss: 0.0179  Train acc: 0.999
Epoch 1800  Loss: 0.0155  Train acc: 0.999

Test accuracy: 0.964
```

**96.4% accuracy on images it never saw during training** — recognizing someone's handwriting, built from raw arithmetic, in about 90 lines of code. Notice the loss starts around 2.6 (essentially random guessing among 10 classes) and drops sharply — that's the network genuinely discovering which pixel patterns correspond to which digits, purely through the gradient descent process described in Part 5.

---

## Part 7 — Understanding What Just Happened

A few things worth sitting with:

- **The network never saw code that says "a 7 has a horizontal line at the top."** It discovered pixel patterns correlated with each digit entirely through repeated small corrections to ~2,500 numbers (the weights). That's the whole trick of machine learning: instead of programming the rules by hand, you program a way to *discover* the rules from examples.
- **The gap between train accuracy (99.9%) and test accuracy (96.4%)** is called **overfitting** — the network memorizing quirks of the training images slightly more than learning the general pattern. This gap is one of the central challenges in real-world machine learning, addressed with techniques like regularization, dropout, and more training data — worth researching once you're comfortable with the basics here.
- **Every "AI" headline you read** — chatbots, image generators, self-driving perception systems — is this same loop (forward pass, loss, backward pass, gradient descent update) run at a vastly larger scale: billions of weights instead of ~2,500, specialized layer types instead of plain matrix multiplications, and datasets of billions of examples instead of 1,797. The core learning algorithm is the one you just built.

---

## Part 8 — Going Beyond: Why and When to Use Frameworks

Hand-writing backpropagation, as you just did, is the right way to *learn* how neural networks work. It is not how you'd build anything serious, for a few concrete reasons:

- **Manually deriving the chain-rule math for every new architecture is slow and error-prone.** Frameworks like **PyTorch** and **TensorFlow** provide "automatic differentiation" — you just describe the forward pass, and the framework computes all the gradients for you, correctly, every time.
- **GPU acceleration.** Real networks train on graphics cards, which can do thousands of matrix multiplications in parallel. Frameworks handle shipping computation to the GPU; raw NumPy does not.
- **Battle-tested layer types.** Convolutional layers (for images), recurrent/attention layers (for sequences and language — the basis of transformers, which power tools like Claude and ChatGPT) are complex to implement correctly and are provided as tested building blocks.

The same digit-classifier network in PyTorch looks like this — notice it's the exact same concepts (layers, activation functions, a loss function, gradient descent) with the manual math handled for you:

```python
import torch
import torch.nn as nn

class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 32)  # same shape as our from-scratch version
        self.layer2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))   # same ReLU we built by hand
        x = self.layer2(x)               # raw scores; softmax is applied inside the loss below
        return x

model = DigitNet()
loss_fn = nn.CrossEntropyLoss()          # the same cross-entropy from Part 5.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # the same gradient descent from Part 2.3

# One training step:
# predictions = model(X_batch)
# loss = loss_fn(predictions, y_batch)
# loss.backward()        # <-- automatic differentiation replaces our hand-written backward()
# optimizer.step()       # <-- applies the gradient descent update automatically
```

You should recognize every single concept in that PyTorch code because you built it by hand first. That's the entire point of this exercise: frameworks stop being magic once you know what they're automating.

---

## Part 9 — Where to Go From Here

Now that the fundamentals are solid, natural next steps, roughly in order of difficulty:

1. **Install PyTorch** and reimplement the digit classifier there, to see the same ideas expressed with a framework
2. **Convolutional Neural Networks (CNNs)** — a specialized layer type that dramatically improves image tasks by exploiting the fact that nearby pixels are related
3. **Try a bigger, messier image dataset** (like MNIST at full size, or CIFAR-10) to see overfitting and regularization become real problems worth solving
4. **Recurrent Neural Networks and Transformers** — the architectures behind language models, which handle sequences (text, audio) rather than fixed-size inputs
5. **Read the original backpropagation paper** (Rumelhart, Hinton & Williams, 1986) now that the math is no longer intimidating

You now understand, from raw arithmetic upward, what every "AI" system in the news is fundamentally doing under the hood.
