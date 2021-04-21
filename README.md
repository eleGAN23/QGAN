# QGAN
Official PyTorch repository for:

Eleonora Grassucci, Edoardo Cicero, Danilo Comminiello, "[Quaternion Generative Adversarial Networks](https://arxiv.org/pdf/2104.09630.pdf)", <i>arXiv preprint: arXiv:2104.09630v1</i>, Apr. 2021.

### Abstract

Latest Generative Adversarial Networks (GANs) are gathering outstanding results through a large-scale training, thus employing models composed of millions of parameters requiring extensive computational capabilities. Building such huge models undermines their replicability and increases the training instability. Moreover, multi-channel data, such as images or audio, are usually processed by real-valued convolutional networks that flatten and concatenate the input, losing any intra-channel spatial relation. To address these issues, here we propose a family of quaternion-valued generative adversarial networks (QGANs). QGANs exploit the properties of quaternion algebra, e.g., the Hamilton product for convolutions. This allows to process channels as a single entity and capture internal latent relations, while reducing by a factor of 4 the overall number of parameters. We show how to design QGANs and to extend the proposed approach even to advanced models. We compare the proposed QGANs with real-valued counterparts on multiple image generation benchmarks. Results show that QGANs are able to generate visually pleasing images and to obtain better FID scores with respect to their real-valued GANs. Furthermore, QGANs save up to 75% of the training parameters. We believe these results may pave the way to novel, more accessible, GANs capable of improving performance and saving computational resources.

### Results

Summary parameters and memory results for SNGAN and QSNGAN.

| Model  | # Total parameters | Disk Memory^|
|--------|:------------------:|:---------:|
| SNGAN  |     61,173,000     |   115 GB  |
| QSNGAN |     16,896,105     |   35 GB   |

^ Memory required by the generator checkpoint for inference.

Samples generated from the real-valued SNGAN on the left and from the proposed Quaternion-valued QSNGAN on the right on the CelebA-HQ dataset and 102 Oxford Flowers dataset.

<img src="./samples/CelebAHQ-SNGAN.png" width="350" height="350"/>          <img src="./samples/CelebAHQ-QSNGAN_QSN.png" width="350" height="350"/>

<img src="./samples/flowers-SNGAN.png" width="350" height="350"/>          <img src="./samples/flowers-QSNGAN_QSN.png" width="350" height="350"/>


### Training

Please install the requirements file. Download the dataset and set the name and the path in the `.txt` file. Images are resized by default to 128x128 if not set differently. Default Dataloaders are set up for CelebA-HQ, 102 Oxford Flowers and CIFAR10. For the latter set `image_size=32` and the model either `SNGAN_32` or `QSNGAN_QSN_32`.
The files `SNGAN_128.txt` and `QSNGAN_128.txt` contain the configurations and options for training. Then, training can be performed through:

```python
python Qmain_FromText.py --TextArgs=*choose-the-txt-file*
```



