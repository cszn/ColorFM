# Testing ColorFM Applications

[Back to the main page](../README.md) · [Environment setup](../README.md#installation) · [Training guide](../TRAINING.md)

Complete the [environment setup](../README.md#installation) first. Then enter this directory from the repository root:

```bash
cd app
```

## Image color transfer

Run the optimization-based ColorFM-O application:

```bash
python app_colorfm_o.py
```

Run the learning-based ColorFM-L application:

```bash
python app_colorfm_l.py
```

## Video color transfer

Both video applications support OpenCV encoding. FFmpeg is recommended for H.264 output and retaining the source audio:

```bash
conda install -n ColorFM -c conda-forge ffmpeg
```

Run the optimization-based ColorFM-O video application:

```bash
python app_colorfm_o_video.py
```

Run the learning-based ColorFM-L video application:

```bash
python app_colorfm_l_video.py
```

## Files and outputs

- ColorFM-L loads `checkpoints/colorfm_l.pth` from the repository root by default.
- Saved results are written to the repository-level `outputs` directory by default.
- The applications can be launched from the `app` directory without changing the internal model, config, checkpoint, or output paths.
