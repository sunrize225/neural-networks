"""
This is a neural network with one input and one output neuron. 
No hidden layer. One weight, one bias, no activation function
Essentially, it will try to estimate the slope and y-intercept of a line
"""
import random, matplotlib.pyplot as plt

def testCaseGen(slope):
    used_ints = set()
    def case():
        nonlocal used_ints
        if len(used_ints) == 1400:
            used_ints = set()
        while True:
            x = random.randint(-700,700)
            if x not in used_ints: break
        used_ints.add(x)
        return x, (slope * x) + intercept
    return case

def feedFoward(a_0):
    a_1 = (a_0 * w) + b
    return a_1

def error(a_1, y):
    return (a_1 - y)**2
def error_dx(a_1, y):
    return 2 * (a_1 - y)

def adjust_w(a_0, a_1, y):
    """
    this is one-half of the 'backpropagation', but there is only one neuron.
    Returns pCpw : partial derivative of cost function with respect to w
    """
    papw = a_0
    pCpa = error_dx(a_1, y)
    pCpw = papw * pCpa

    return pCpw

def adjust_b(a_1, y):
    """
    This is the other half of the 'backpropagation'
    returns pCpb : partial derivative of cost function with respect to b
    """
    papb = 1
    pCpa = error_dx(a_1, y)
    pCpb = papb * pCpa

    return pCpb

def train(trials = 1000, step_size = 0.0000001):
    global w
    global b
    y_points = []
    testCases = testCaseGen(slope)
    for _ in range(trials):
        a_0, y = testCases()
        print("weight:", w, "bias:", b, "input:", a_0, "output:", a_1 := feedFoward(a_0))
        y_points.append(error(a_1, y))
        w -= adjust_w(a_0, a_1, y) * step_size
        b -= adjust_b(a_1, y) * step_size * 10000
    print("final error:", y_points[-1] )
    plt.plot(list(range(trials)), y_points)
    plt.show()

slope = float(input("What do you want the slope of the line to be?"))

intercept = float(input("What do you want the y-intercept of the line to be?"))

w = random.randint(-200, 200) # initalize single and only weight to random value

b = random.randint(-200, 200) # initalize single and only bias to random value

train(trials=5000)