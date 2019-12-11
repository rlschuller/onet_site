---
title: Occupancy Network 
author: Rodrigo Loro Schuller
date: December 2019
---


Introduction
============

Presentation
------------

Click [here](slides_no_copyright/index.html) to see it in your browser - Firefox is recommended.

Click [here](slides_no_copyright/slides.pdf) for the PDF version.


Abstract Definition
-------------------

The problem of representing 3D structures is harder than its 2D counterpart. Good solutions for it are specially important for learning-based algorithms, since bad representations usually yield unreasonable large memory requirements, glaring inconsistencies or other difficulties.

An *Occupancy Network* is a state-of-the-art solution that uses implicit functions (neural networks with parameters $\theta$) to represent 3D objects in a compact and expressive manner. Bellow we have its formal definition.

**Definition (Occupancy Network)** For a given input $x \in X$, we want a binary classification neural network: $f^x_\theta : \mathbb{R}^3 \to [0,1]$. We can just add $x$ to the inputs, ie, 

$$f_\theta : \mathbb{R}^3 \times X \to [0,1].$$

$f_\theta$ is called the *Occupancy Network*.


Other Representations
=====================

A more detailed (and visual) comparison can be found on the presentation. To avoid redundancy, we'll present a brief synthesis with the main takeaways.


1 Voxels
--------

### Pros

* Simple to use;

### Cons

* Doesn't quite work in low resolutions;
* Requires a lot of memory;


2 Point clouds
--------------

### Pros

* Simple to use and behaves well under geometric transformations;
* Doesn't require a lot of memory; 

### Cons

* Hard to extract the underlying geometry;


3 Meshes
--------

### Pros

* Simple to use and behaves well under geometric transformations;
* Doesn't require a lot of memory;

### Cons

* Topology limitations or consistency problems - depending on the approach;


The Devil is in The Details
===========================

Using the abstract definition, the authors presented solutions for 4 different problems. In order to set apart the common details from the specificities of each problem, this section is divided in 5 subsections.

0 Common
--------

### ONet Architecture - The Big Picture

In all experiments, the same base architecture was employed: 

![**Figure 1** Architecture of the *Occupancy Network*.](img/common_arch.svg){width=80%}

**Input** The output of a task-specific encoder $c \in X = \mathbb{R}^C$ and a batch of $T$ points $p_i \in \mathbb{R}^3$.

**Output** To be consistent with our previous notation, the output is given by the numbers $$f_{\theta}(c, p_1),\, \cdots,\, f_{\theta}(c, p_T) \in [0,1].$$ In other words, for each point $p_i$ in the batch, we get a number in $[0, 1]$.

**Evaluation** To show how the architecture works we'll first explain the *Big Picture* - how the components are connected - and afterwards tell exactly what each component does.

For each point $p_i \in \mathbb{R}^3$ in the batch:

1. Use a fully-connected layer to produce a 256-dim feature vector from $p_i$;
2. Do 5 times:
	- Take the output from the previous step and use a **ONet ResNet-block** to produce a new 256-dim feature vector;
3. Take the output from the last **ONet ResNet-block** and pass through a **CBN layer** and a **ReLu activation**;
4. Pass the result through a fully-connected layer to project the features down to 1-dim;
5. Use a **Sigmoid activation** to obtain a number in $[0,1]$;

**Observation** In the ONet article, they originally used the nomenclature *ResNet-blocks*. Since there're different kinds of *ResNet-blocks*, I've added specification tokens to avoid unnecessary confusion.

### Activation Functions 

**ReLU** From the source code (see file `im2mesh/layers.py`), it's clear that the standard PyTorch's ReLU was used. For both the current stable version (1.3.1) and the version used in the project (1.0.0), it is defined as

$$ 
\mathrm{ReLU}(x)
= 
\max\{0, x\}.
$$

**Sigmoid** The sigmoid function is actually implemented in the mesh extraction phase (see `im2mesh/onet/generation.py - line 171`), by applying the inverse function to the threshold:

`threshold = np.log(self.threshold) - np.log(1. - self.threshold)`

By inverting the threshold function again, we can recoup the information:

$$\mathrm{Sigmoid}(x) = \frac{1}{1 + e^{-x}}.$$

This is known as the *logistic sigmoid*.


### Conditional Batch Normalization (CBN) Layer

Let $i$ be the index of our batch members, ie, the index associated with the 3D point $p_i$.

**Input** $c$, the output from the last layer $f_{in}^i$ and the first two moments ($\mu$ and $\sigma^2$) of the corresponding layers $f_{in}$ over all the $T$ members of the batch.

**Output** A 256-dim vector $f_{out}^i$.

**Evaluation** Use two fully-connected layers to obtain 256-dim vectors $\beta(c)$ and $\gamma(c)$. Then  compute 

$$f_{out}^i = \gamma(c) \frac{f_{in}^i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta(c),$$

in which $\epsilon = 10^{-5}$ is a constant added for numerical stability [1]. Since the sum between scalars and vectors is already implicitly defined, it's important to highlight (as in the original article [1]) that the multiplication by $\gamma$ is a *picewise* (not inner) product.

Here follows the article describing this acceleration technique:

[1] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) - Sergey Ioffe, Christian Szegedy (2015)

and a link for the respective PyTorch module (see `im2mesh/layers.py`):

[2] [torch.nn.modules.batchnorm](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html)

**<span style="color:red">TODO: Explain what the running mean of ($\mu$ and $\sigma^2$) means.</span>**


### ONet ResNet-block

We'll now describe a single ONet ResNet-block (see `im2mesh/layers.py - class CResnetBlockConv1d`) as the composition of previously defined components, in the order of application:

1. CBN layer;
2. ReLU activation function in each dimension;
3. Fully-connected layer;
4. CBN layer;
5. ReLU activation function in each dimension;
6. Fully-connected layer;

To get the output of the ONet ResNet-block, we then sum the input of step 1 to the output of step 6.


1 Single View Image Reconstruction
----------------------------------

Before talking about the encoder, we need the definition bellow.

**Definition (ImageNet normalization)** Let $x \in [0,1]^{w \times h \times 3}$ be a colored image, $\mathrm{\mu_{ImN}} :=(	.485,\, .456,\,	.406)$ and $\mathrm{\sigma_{ImN}}:= (.229,\, .224, \,	.225)$. Then the *normalized* image is then given by 

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_{ImN}}{\sigma_{ImN}},
$$
in which the division is defined as piecewise division.


### The Image Encoder

![**Figure 2** Modified ResNet18 - the image encoder.](img/svi_encoder.svg){width=100%}

**Input** $224 \times 224$ image, normalized according to ImageNet standards. 

**Output** A feature vector $c \in \mathbb{R}^C$, for $C=256$.

**Evaluation** The only difference between ResNet18 [3] and the neural network used as the encoder is the last fully connected layer. Instead of producing a 512-dim output, the last layer projects it down to a 256-dim vector $c$. The encoder was pre-trained on the ImageNet dataset.

[3] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), Kaiming He et al (2015)

The reference above is a landmark of sorts for its area - hence the large number of citations. They introduced the idea of using residues to make deep NN viable.

One eye-catching piece of information is that the authors of ONet (see `im2mesh/encoder/conv.py`) did implement several sizes of ResNets: 18, 34, 50 and 101. It raises the question of why ResNet18 worked better than its deeper counterparts.  


2 Point Cloud Completion
------------------------

![**Figure 3** Encoder for point cloud completion.](img/psgn_encoder.svg){width=100%}

**Input** M=300 points generated from a (watertight) mesh taken from ShapeNet in the following manner:

1. Normalize the shape such that its bounding box is centered at the origin and that the biggest side of the bounding box measures exactly 1;
2. Subsample 300 points from the *surface* of the model;
3. Apply noise to the points using a Gaussian distribution with zero mean and standard deviation of 0.05 (see `im2mesh/data/transforms.py`);

**Output** A feature vector $c \in \mathbb{R}^{C_p}$, for $C_p=512$.

**Description** The network consists of 2 fully connected layers (for input and output) and 5 *PointNet ResNet-blocks* intercalated by pooling+expansion layers, as shown in Fig [3].

**Definition (PointNet ResNet-block)** The explanation on the supplementary material was incomplete, but we can take a look at the code (see `im2mesh/layers.py` and `./im2mesh/encoder/pointnet.py`) to understand these layers.

The class used to represent the *PointNet ResNet-blocks* is the following:

```
# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
```

For *PointNet ResNet-blocks*: `size_in=1024`, `size_out=512` and `size_h=512`. One block is defined as the following composition, in order of application:

1. Fully connected NN - 1024-dim to 512-dim;
2. Fully connected NN - 512-dim to 512-dim;
3. ReLU activation layer;

Since the input and output dimensions differ, we have an additional FCNN projecting the input for step 1 $x \in \mathbb{R}^{1024}$ to $x_s \in \mathbb{R}^{512}$. Hence the final output is the sum of the output from step 3 and $x_s$.


3 Voxel Super Resolution
------------------------

![**Figure 4** Encoder for voxel super resolution.](img/voxel_encoder.svg)

**Input** A grid of $32^3$ voxels. More specifically, voxels generated from (watertight) ShapeNet meshes with the algorithm bellow: 

1. Normalize the shape just like in the point cloud completion preprocessor;
2. Mark all voxels that intercept with the surface as occupied;
3. For each of the remaining voxels:
	- Choose a random point inside the voxel;
	- If the point lies inside the mesh mark the corresponding voxel as occupied;

**Output** A feature vector $c \in \mathbb{R}^C$, for $C=256$.

**Evaluation** The input passes trough 5 3D convolution layers, and a fully connected layer to project the output to the space $\mathbb{R}^{256}$. All convolution layers use *zero-padding* with size 1 and $3\times 3\times 3$ filters (see `im2mesh/encoder/voxels.py`). Stride is implicitly defined in Fig [4].


4 Unconditional Mesh Generation
-------------------------------

We'll talk about the encoder before long, but before that I'd like to present a general idea of the process to the reader. Although the authors did use the word *unsupervised* in the article, it's a bit of a stretch - here's why:

Suppose that we want to generate shapes in the category *car*, we would proceed as follows: 

1. Take the annotated meshes from ShapeNet and extract the subset of shapes that correspond to our chosen category;
2. Train a Variational Autoencoder using this subset;
3. Sample the latent space and use the decoder to generate a shape;


### The VAC encoder 

**Input**  
