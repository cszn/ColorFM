# [ColorFM: An Optimization-to-Learning Framework for Color Transfer via Flow Matching](https://arxiv.org/abs/2607.07119)

<p align="center">
  <a href="https://github.com/heyh31">Yuhang He</a><sup>1</sup>,
  <a href="https://github.com/cszn">Kai Zhang</a><sup>1,&#8224;</sup>,
  Xiaoming Li<sup>1</sup>,
  Du Chen<sup>2</sup>,
  Jian Yang<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Nanjing University, China &nbsp;&nbsp;
  <sup>2</sup>VIVO BlueImage Lab, China<br>
  <sup>&#8224;</sup>Corresponding author
</p>

<p align="center"><strong>ECCV 2026</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.07119">
    <img src="https://img.shields.io/badge/arXiv-2607.07119-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://heyh31.github.io/ColorFM_page/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
</p>

<p align="center">
  <img src="static/images/gifs/image-transfer-01.gif" width="44%" alt="ColorFM image color transfer result">
  <img src="static/videos/video-transfer-01.gif" width="51%" alt="ColorFM video color transfer result">
</p>

<p align="center">
  <em>ColorFM image color transfer (left) and video color transfer (right).</em>
</p>

<p align="center">
  <strong>🚀 Explore more application results:</strong>
  <a href="#image-color-transfer"><strong>Image Color Transfer</strong></a>
  &nbsp;|&nbsp;
  <a href="#video-color-transfer"><strong>Video Color Transfer</strong></a>
</p>

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [ColorFM: An Optimization-to-Learning Framework for Color Transfer via Flow Matching](#colorfm-an-optimization-to-learning-framework-for-color-transfer-via-flow-matching)
  - [Overview](#overview)
  - [Online Demos](#online-demos)
  - [Image Color Transfer](#image-color-transfer)
  - [Video Color Transfer](#video-color-transfer)
  - [Method](#method)
  - [Quantitative Results](#quantitative-results)
  - [Installation](#installation)
  - [Testing/Training](#testingtraining)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)
  - [License](#license)

</details>

Overview
----------

ColorFM is an optimization-to-learning framework for accurate and semantically consistent color transfer. It connects instance-specific optimization with efficient feed-forward inference through two complementary variants: ColorFM-O and ColorFM-L.


Online Demos
----------

| Method | Type | Demo |
|:---:|:---:|:---:|
| ColorFM-O | Optimization-based | [Try online](https://huggingface.co/spaces/heyh97791/ColorFM-O) |
| ColorFM-L | Learning-based | [Try online](https://huggingface.co/spaces/heyh97791/ColorFM-L) |

Image Color Transfer
----------

<p align="center">
  <img src="static/images/static/triptychs/4.png" width="90%" alt="Content, style, and color transfer result for example 4"/> 
</p>

<p align="center">
  <img src="static/images/static/triptychs/6.png" width="90%" alt="Content, style, and color transfer result for example 6"/>
</p>

<p align="center">
  <img src="static/images/static/triptychs/5.png" width="90%" alt="Content, style, and color transfer result for example 5"/>
</p>

<p align="center">
  <img src="static/images/static/triptychs/7.png" width="90%" alt="Content, style, and color transfer result for example 7"/>
</p>

<p align="center">
  <img src="static/images/static/triptychs/1.png" width="90%" alt="Content, style, and color transfer result for example 1"/>
</p>

<p align="center">
  <img src="static/images/static/triptychs/8.png" width="90%" alt="Content, style, and color transfer result for example 8"/>
</p>

<p align="center">
  <img src="static/images/static/triptychs/3.png" width="90%" alt="Content, style, and color transfer result for example 3"/>
</p>

<p align="center">
  <img src="static/images/static/triptychs/2.png" width="90%" alt="Content, style, and color transfer result for example 2"/>
</p>


<!-- <p align="center">
  <img src="static/images/gifs/image-transfer-01.gif" height="190px" alt="Image color transfer example 1"/>
  <img src="static/images/gifs/image-transfer-02.gif" height="190px" alt="Image color transfer example 2"/>
  <img src="static/images/gifs/image-transfer-03.gif" height="190px" alt="Image color transfer example 3"/>
</p> -->

Video Color Transfer
----------

<p align="center">
  <img src="static/videos/video-transfer-01.gif" height="190px" alt="Video color transfer example 1"/>
  <img src="static/videos/video-transfer-02.gif" height="190px" alt="Video color transfer example 2"/>
</p>

Method
----------

ColorFM formulates color transfer as transporting pixel distributions along velocity fields via Flow Matching. ColorFM-O optimizes an instance-specific velocity field with semantic guidance, while ColorFM-L learns from the generated pairs to provide efficient feed-forward inference.

<p align="center">
  <img src="static/images/method/colorfm-framework.jpg" width="100%" alt="Overview of the ColorFM framework"/>
</p>

<p align="center"><em>Overview of the ColorFM-O and ColorFM-L frameworks.</em></p>

Quantitative Results
----------

The following table compares ColorFM with existing color transfer methods in terms of similarity, Lipschitz constant, and inference time. All results are evaluated at an image resolution of 512 x 512.

<p align="center">
  <img src="static/images/method/quantitative-results.jpg" width="100%" alt="Quantitative comparison with existing color transfer methods"/>
</p>

Installation
----------

Create an environment and install the dependencies from the repository root:

```bash
conda create -n ColorFM python=3.10 -y
conda activate ColorFM
pip install torch torchvision
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version when GPU acceleration is required. [xFormers](https://github.com/facebookresearch/xformers) can optionally accelerate ColorFM-L on supported CUDA environments.

Download the pretrained ColorFM-L checkpoint from [Hugging Face](https://huggingface.co/heyh97791/ColorFM) and place it under the repository-level `checkpoints` folder. ColorFM-O does not require a pretrained checkpoint.

Testing/Training
----------

| Guide | Description |
|:---|:---|
| [Testing](app/README.md) | Run the ColorFM-O and ColorFM-L image or video WebUIs. |
| [Training](TRAINING.md) | Generate ColorFM-O training pairs, train ColorFM-L, and run evaluation. |

Acknowledgements
----------

This project builds upon the open-source implementations of [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI.

Citation
----------

If you find this work useful, please cite:

```bibtex
@inproceedings{he2026colorfm,
    title={ColorFM: An Optimization-to-Learning Framework for Color Transfer via Flow Matching},
    author={He, Yuhang and Zhang, Kai and Li, Xiaoming and Chen, Du and Yang, Jian},
    booktitle={European Conference on Computer Vision},
    year={2026}
}
```

License
----------

This project is released under the [Apache License 2.0](LICENSE).
