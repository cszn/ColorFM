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

---

- [ColorFM: An Optimization-to-Learning Framework for Color Transfer via Flow Matching](#colorfm-an-optimization-to-learning-framework-for-color-transfer-via-flow-matching)
  - [Overview](#overview)
  - [Online Demos](#online-demos)
  - [Testing](#testing)
  - [Method](#method)
  - [Quantitative Results](#quantitative-results)
  - [Image Color Transfer](#image-color-transfer)
  - [Video Color Transfer](#video-color-transfer)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)
  - [License](#license)

Overview
----------

ColorFM is an optimization-to-learning framework for accurate and semantically consistent color transfer. It connects instance-specific optimization with efficient feed-forward inference through two complementary variants: ColorFM-O and ColorFM-L.

Online Demos
----------

| Method | Type | Demo |
|:---:|:---:|:---:|
| ColorFM-O | Optimization-based | [Try online](https://huggingface.co/spaces/heyh97791/ColorFM-O) |
| ColorFM-L | Learning-based | [Try online](https://huggingface.co/spaces/heyh97791/ColorFM-L) |


Testing
----------
Create an environment and install the dependencies:

```bash
conda create -n ColorFM python=3.10 -y
conda activate ColorFM
pip install torch torchvision
pip install -r requirements.txt

# Optional: accelerate ColorFM-L on supported CUDA environments 
# https://github.com/facebookresearch/xformers
```

Run the **image color transfer** WebUI from the repository root:

```bash
# Optimization-based ColorFM-O
python app_colorfm_o.py

# Learning-based ColorFM-L
python app_colorfm_l.py
```

Run the **video color transfer** WebUI:

```bash
# Optimization-based ColorFM-O video transfer
python app_colorfm_o_video.py

# Learning-based ColorFM-L video transfer
python app_colorfm_l_video.py
```

Both video applications support OpenCV encoding. FFmpeg is recommended for H.264 output and retaining the source audio:

```bash
conda install -n ColorFM -c conda-forge ffmpeg
```

Please download [ckpt](https://huggingface.co/heyh97791/ColorFM) and place it under the ``checkpoints`` folder. The checkpoint is required by both ColorFM-L image and video applications. ColorFM-O performs instance-specific optimization and does not require a pretrained checkpoint.


Method
----------

ColorFM formulates color transfer as transporting pixel distributions along velocity fields via Flow Matching. ColorFM-O optimizes an instance-specific velocity field with semantic guidance, while ColorFM-L learns from the generated pairs to provide efficient feed-forward inference.

<p align="center">
  <img src="static/images/method/colorfm-framework.jpg" width="100%" alt="Overview of the ColorFM framework"/>
</p>

<p align="center"><em>Overview of the ColorFM-O and ColorFM-L frameworks.</em></p>

<!--
Algorithm figure placeholder. Expected file:
static/images/method/colorfm-algorithm.png

<p align="center">
  <img src="static/images/method/colorfm-algorithm.png" width="100%" alt="ColorFM algorithm"/>
</p>
-->

Quantitative Results
----------

The following table compares ColorFM with existing color transfer methods in terms of similarity, Lipschitz constant, and inference time. All results are evaluated at an image resolution of 512 x 512.

<p align="center">
  <img src="static/images/method/quantitative-results.jpg" width="100%" alt="Quantitative comparison with existing color transfer methods"/>
</p>

Image Color Transfer
----------

<p align="center">
  <img src="static/images/gifs/image-transfer-01.gif" height="190px" alt="Image color transfer example 1"/>
  <img src="static/images/gifs/image-transfer-02.gif" height="190px" alt="Image color transfer example 2"/>
  <img src="static/images/gifs/image-transfer-03.gif" height="190px" alt="Image color transfer example 3"/>
</p>

Video Color Transfer
----------

<p align="center">
  <img src="static/videos/video-transfer-01.gif" height="190px" alt="Video color transfer example 1"/>
  <img src="static/videos/video-transfer-02.gif" height="190px" alt="Video color transfer example 2"/>
</p>

Acknowledgements
----------

This project builds upon the open-source implementations of [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI.

Citation
----------

If you find this work useful, please cite:

```bibtex
@misc{he2026ColorFM,
      title={ColorFM: An Optimization-to-Learning Framework for Color Transfer via Flow Matching}, 
      author={Yuhang He and Kai Zhang and Xiaoming Li and Du Chen and Jian Yang},
      year={2026},
      eprint={2607.07119},
      url={https://arxiv.org/abs/2607.07119}, 
}
```

License
----------

This project is released under the [Apache License 2.0](LICENSE).
