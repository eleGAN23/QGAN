# QGAN
Official PyTorch repository for [Quaternion Generative Adversarial Networks](https://arxiv.org/pdf/2104.09630.pdf).

Eleonora Grassucci, Edoardo Cicero, Danilo Comminiello

### Abstract

Latest Generative Adversarial Networks (GANs) are gathering outstanding results through a large-scale training, thus employing models composed of millions of parameters requiring extensive computational capabilities. Building such huge models undermines their replicability and increases the training instability. Moreover, multi-channel data, such as images or audio, are usually processed by real-valued convolutional networks that flatten and concatenate the input, losing any intra-channel spatial relation. To address these issues, here we propose a family of quaternion-valued generative adversarial networks (QGANs). QGANs exploit the properties of quaternion algebra, e.g., the Hamilton product for convolutions. This allows to process channels as a single entity and capture internal latent relations, while reducing by a factor of 4 the overall number of parameters. We show how to design QGANs and to extend the proposed approach even to advanced models. We compare the proposed QGANs with real-valued counterparts on multiple image generation benchmarks. Results show that QGANs are able to generate visually pleasing images and to obtain better FID scores with respect to their real-valued GANs. Furthermore, QGANs save up to 75% of the training parameters. We believe these results may pave the way to novel, more accessible, GANs capable of improving performance and saving computational resources.

### Results

<img src="./samples/CelebAHQ-SNGAN.png" width="500" height="500"/> <img src="./samples/CelebAHQ-QSNGAN_QSN.png" width="500" height="500"/>

<img src="./samples/flowers-SNGAN.png" width="500" height="500"/> <img src="./samples/flowers-QSNGAN_QSN.png" width="500" height="500"/>


### Training

