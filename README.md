# Movement-Representation

## Movement primitives

The term movement primitives refers to an organization of continuous motion signals in the form of a superposition in parallel and in series of simpler signals, which can be viewed as “building blocks” to create more complex movements. This principle, coined in the context of motor control, remains valid for a wide range of continuous time signals (for both analysis and synthesis). 

Radial basis functions (RBFs) are ubiquitous in continuous time series encoding, notably due to their simplicity and ease of implementation. Most algorithms exploiting this representation rely on some form of regression, often related to locally weighted regression (LWR), which was introducedin statistics and popularized in robotics.

In time series encoding, the use of Fourier basis functions provides useful connections between the spatial domain and the frequency domain. In the context of Gaussian mixture models, several Fourier series properties can be exploited, notably regarding zero-centered Gaussians, shift, symmetry, and linear combination. 

## Learning by exploration 

Learning by exploration covers a wide range of techniques and frameworks, from reinforcement learning to evolutionary methods. A crucial aspect for the success of these techniques is to represent the movement or the policy in a compact and adaptive manner, by leveraging prior knowledge about movements and skills. The project proposes to test different representations by using a cross-entropy method (CEM), which can be viewed as one of the most simple form of learning by self-refinement. The movement will be formed as a superposition of basis functions. Two representations will be tested: radial basis functions and Fourier series. 


--

[[1]](http://calinon.ch/papers/Calinon_MMchapter2019.pdf) S. Calinon, “Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series,” in Mixture Models and Applications, N. Bouguila and W. Fan, Eds. Cham: Springer International Publishing, 2020, pp. 39–57.

[[2]](calinon.ch/papers/Calinon-Lee-learningControl.pdf) S. Calinon and D. Lee, “Learning Control,” in Humanoid Robotics: A Reference, A. Goswami and P. Vadakkepat, Eds. Dordrecht: Springer Netherlands, 2017, pp. 1–52.

