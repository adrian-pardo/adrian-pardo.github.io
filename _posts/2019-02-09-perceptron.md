---
title: "Machine Learning Project: Perceptron Logic Gates"
date: 2019-02-09
tags: [machine learning, data science, neural network, data visualization, neuroscience]
#header:
  # image: "images/perceptron/neuron.png"
excerpt: "Visualization of Perceptron Logic Gates using sklearn and matplotlib"
mathjax: "true"
classes: wide
---
# Visualization of Perceptron Logic Gates using sklearn and matplotlib

In this project, I used perceptrons to model the fundamental building blocks of computers — logic gates.

Comparison of AND, OR, XOR Logic Gates: <img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/logicgate.png" alt="logic gates ">

This project helped me conceptualize the inner workings of an artificial neural network by examining the limitations of its basic building block, the perceptron. Considering my neuroscience background, I found it particularly helpful to compare the perceptron to its biological equivalent, the neuron.

In the human brain, neural nets are composed of multiple *neurons*. Conversely, artificial neural nets are composed of multiple *perceptrons*.

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/neuron.png" alt="neuron vs perceptron">

A *biological neuron* has:
1. **Dendrites** to receive signals
2. A **cell body** to process these signals
3. An **axon** to send signals out to other neurons

A *perceptron* has:
1. A number of **input channels**
2. A **processing stage**
3. **One output** that can fan out to multiple other artificial neurons

*One defining, limiting characteristic of the perceptron machine learning model is that is its activation function is a simple binary function.* This means that a single perceptron can only provide solutions for problems that are linearly separable.

First, let's visualize AND, OR, and XOR Logic Gates using coordinate points:
```python
    from sklearn.linear_model import Perceptron
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product

    #list of four possible inputs to gate
    data = [[0,0], [0,1], [1,0], [1,1]]
    #AND, OR, and XOR gate labels
    ANDlabels = [0, 0, 0, 1]
    ORlabels = [0, 1, 1, 1]
    XORlabels = [0, 1, 1, 0]

    #generate x and y values from data
    x_values = [point[0] for point in data]
    y_values = [point[1] for point in data]

    #plot AND gate
    fig = plt.figure(figsize = (12,3))
    plt.subplot(1,3,1)
    plt.scatter(x_values, y_values, c=ANDlabels)
    plt.title("AND Logic Gate")

    #plot OR gate
    plt.subplot(1,3,2)
    plt.scatter(x_values, y_values, c=ORlabels)
    plt.title("OR Logic Gate")

    #plot XOR gate
    plt.subplot(1,3,3)
    plt.scatter(x_values, y_values, c=XORlabels)
    plt.title("XOR Logic Gate")

    plt.show()
```


From this code we obtain this graph: <img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/3_logic_gates.png" alt=" 3 logic gates ">


In this context "linearly separable" means that a line could be drawn that would entirely separate the colored dots. This line can be drawn on the AND, OR graphs indicating that these logic gates are linearly separable. This line cannot be drawn on the XOR graph, indicating that the XOR logic gate is not linearly separable:

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/3_logic_gates_edit.png" alt="3 logic gates">

We can further support these findings by obtaining the accuracy of the perceptron model when fitted to each gate:
```python
    #create perceptron object, train model, and print accuracy of model on the data points
    classifier = Perceptron(max_iter=40, tol=1e-3)

    classifier.fit(data, ANDlabels)
    print(classifier.score(data, ANDlabels))
    #output of 1.0 indicates that 100% of the time, model was able to correctly determine output given data

    classifier.fit(data, ORlabels)
    print(classifier.score(data, ORlabels))
    #output of 1.0 indicates that 100% of the time, model was able to correctly determine output given data

    classifier.fit(data, XORlabels)
    print(classifier.score(data, XORlabels))
    #output of 0.5 indicates that 50% of the time, model was able to correctly determine output given data
```

An output score of 1.0 confirms that the AND, OR logic gates are linearly separable. An output score of 0.5 confirms that the XOR logic gate is NOT linearly separable.

However, we're not done yet. We can use a decision function to determine the distance between a coordinate point and the logic gate's decision boundary.

For example if the data is fitted to the XOR labels:
```Python
    print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))
    #output of [ 0.  -1.  -0.5] tells us that points [0, 0], [1, 1], [0.5, 0.5] are 0, -1.0, and-0.5 units away from XOR decision boundary
```

We can then utilize this decision function to create a heatmap indicating the realtively location of the decision boundary:
```python
    #use decision_function method to create a heatmap containing 100 equidistant, ordered pairs and their respective distances from the decision boundary
    x_values = np.linspace(0, 1, 100)
    y_values = np.linspace(0, 1, 100)
    point_grid = list(product(x_values, y_values))

    #plot heatmap for AND gate
    classifier.fit(data, ANDlabels)
    distances = classifier.decision_function(point_grid)
    abs_distances = [abs(pt) for pt in distances]
    distances_matrix = np.reshape(abs_distances, (100,100))

    heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
    cbar = plt.colorbar(heatmap)
    plt.xlabel("X-Value")
    plt.ylabel("Y-Value")
    plt.title("AND Logic Gate Heatmap")
    cbar.set_label("Distance From Decision Boundary", rotation=270, labelpad=13)
    plt.show()
```
This code gives us this graph:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/AND_gate_heatmap.png" alt="AND gate heatmap">

This process can then be repeated for the OR and XOR gates:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/OR_gate_heatmap.png" alt="OR gate heatmap">
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/XOR_gate_heatmap.png" alt="XOR gate heatmap">


The biggest takeaway from this project is that a single perceptron only has the capability to solve problems that are linearly separable. However, if multiple perceptrons are combined, then a neural net is created which can then solve all kinds of problems! More on neural nets in future posts! :D
