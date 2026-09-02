# Training ColorFM

[Back to the main page](README.md) · [Environment setup](README.md#installation) · [Application guide](app/README.md)

Complete the [environment setup](README.md#installation) first and run all commands below from the repository root.

ColorFM training consists of two stages. Stage 1 uses ColorFM-O to generate training pairs. Stage 2 uses these generated pairs to train the offline ColorFM-L model.

## Stage 1: Generate training pairs with ColorFM-O

Place the source images in one folder. Because Stage 1 uses `_` to separate the content and style names in `a_b.png`, it is recommended to rename the source images with numeric names first:

```bash
python rename_images.py \
  --input-dir /data/set_a/original_images \
  --output-path /data/set_a/images
```

This copies the images to the output folder as `001.jpg`, `002.jpg`, and so on while preserving the source images. Then generate the training pairs:

```bash
python main_train_stage1.py \
  --input-dir /data/set_a/images \
  --output-dir /data/set_a/styled
```

Every two different images are paired in both directions. If `a.png` is the content image and `b.png` is the style image, the generated result is saved as:

```text
/data/set_a/styled/a_b.png
```

For large-scale pair generation, multiple tasks can run in parallel. Use `run_train_stage1_parallel.sh` as a template for configuring the dataset paths, CUDA devices, and task ranges:

```bash
bash run_train_stage1_parallel.sh
```

## Stage 2: Train the offline ColorFM-L model

Configure the training datasets and CUDA devices in `configs/colorfm_l.yaml`. The dataset paths are matched by position: `image_dir[0]` corresponds to `styled_dir[0]`, `image_dir[1]` corresponds to `styled_dir[1]`, and so on. All matched datasets are combined for training. Set `eval_path: []` to disable evaluation.

When evaluation is enabled, follow the instructions in `metrics/ckpt_download` to download and place the required metric weights.

Start Stage 2 training:

```bash
python main_train_stage2.py
```

Training automatically resumes from `outputs/<exp_name>/checkpoints/last.ckpt` when that file exists. A specific checkpoint can also be selected explicitly:

```bash
python main_train_stage2.py --resume /path/to/checkpoint.ckpt
```

The experiment outputs are organized as follows:

```text
outputs/<exp_name>/
├── checkpoints/
│   ├── colorfm_l_epoch=004.ckpt
│   ├── last.ckpt
│   └── colorfm_l.pth
└── logs/
    └── version_0/
```

TensorBoard records the training losses and, when evaluation is enabled, Content Similarity, Lipschitz, and Style Similarity:

```bash
tensorboard --logdir outputs/<exp_name>/logs
```

## Evaluate ColorFM-L

Before evaluation, follow `metrics/ckpt_download` to place the LDC and Style Similarity weights in the `metrics` folder. The evaluation script accepts both a plain ColorFM-L `.pth` weight file and a Lightning `.ckpt` checkpoint:

```bash
python main_eval_colorfm_l.py \
  --checkpoint checkpoints/colorfm_l.pth \
  --eval-path /data/eval_images \
  --output-dir outputs/colorfm_l_eval
```

Multiple evaluation folders can be passed after `--eval-path`. Each folder is evaluated using all directed pairs of different images:

```bash
python main_eval_colorfm_l.py \
  --checkpoint checkpoints/colorfm_l.pth \
  --eval-path /data/eval_a /data/eval_b \
  --output-dir outputs/colorfm_l_eval
```

By default, content images are resized to `data.eval_image_size` from `configs/colorfm_l.yaml`. Add `--full-resolution` to keep the original content resolution in the generated results:

```bash
python main_eval_colorfm_l.py \
  --checkpoint checkpoints/colorfm_l.pth \
  --eval-path /data/eval_images \
  --output-dir outputs/colorfm_l_eval \
  --full-resolution
```

Generated images are named `content_style.png`. The number of saved results is controlled by `solver.eval_save_images` in the config: `-1` saves all results, `0` disables saving, and a positive value saves the first N results. If `--output-dir` is omitted, images are saved to `outputs/<exp_name>/eval_images`.
