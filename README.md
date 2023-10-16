<p align="center">
    <img src="assets/logo.png" width="400">
</p>

## DiffBIR TOMO: Towards Blind Image Restoration with Generative Diffusion Prior for Soil Tomography

[Paper](https://arxiv.org/abs/2308.15070)

forked from: [DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior](https://0x3f3f3f3fun.github.io/projects/diffbir/)

### Model architecture: 

<p align="center">
    <img src="assets/architecture.png" style="border-radius: 15px">
</p>

## :book:Table Of Contents

- [Visual Results On Microtomographic images](#visual_results_micro)
- [Visual Results On Medical images](#visual_results_med)
- [TODO](#todo)
- [Installation](#installation)
- [Pretrained Models](#pretrained_models)
- [Quick Start (gradio demo)](#quick_start)
- [Colab demo](#colab)
- [Inference](#inference)
- [Train](#train)

## <a name="visual_results_micro"></a>Visual Results On Microtomographic images
### x4 Super-resolution

<!-- [<img src="assets/visual_results/general6.png" height="223px"/>](https://imgsli.com/MTk5ODI3) -->

## <a name="visual_results_med"></a>Visual Results On Medical Images
### x4 Super-resolution
<!-- [<img src="assets/visual_results/whole_image3.png" height="268"/>](https://imgsli.com/MjA1OTY2) -->
<!-- [<img src="assets/visual_results/face3.png" height="223px"/>](https://imgsli.com/MTk5ODMy) -->
 <!-- [<img src="assets/visual_results/face5.png" height="223px"/>](https://imgsli.com/MTk5ODM1)  -->

<!-- [<img src="assets/visual_results/whole_image1.png" height="410"/>](https://imgsli.com/MjA1OTU5) -->

<!-- </details> -->

## <a name="update"></a>:new:Update

- **2023.16.10**: Repo created
<!-- - [**History Updates** >]() -->


## <a name="todo"></a>:climbing:TODO

- [ ] Release code 
- [ ] Release pretrained models

## <a name="installation"></a>Installation
<!-- - **Python** >= 3.9
- **CUDA** >= 11.3
- **PyTorch** >= 1.12.1
- **xformers** == 0.0.16 -->

```shell
# clone this repo
git clone https://github.com/pihchikk/DiffBIR_TOMO
cd DiffBIR

# create an environment with python >= 3.9
conda create -n diffbir python=3.9
conda activate diffbir
pip install -r requirements.txt
```

Note the installation is only compatible with **Linux** users. If you are working on different platforms, please check [xOS Installation](assets/docs/installation_xOS.md).

<!-- ```shell
# clone this repo
git clone https://github.com/XPixelGroup/DiffBIR.git
cd DiffBIR

# create a conda environment with python >= 3.9
conda create -n diffbir python=3.9
conda activate diffbir

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda install xformers==0.0.16 -c xformers

# other dependencies
pip install -r requirements.txt
``` -->

## <a name="pretrained_models"></a>:dna:Pretrained Models

| Model Name | Description | HuggingFace | OpenXLab |
| :--------- | :---------- | :---------- | :---------- |
| swinir_tomo.ckpt | Stage1 model (SwinIR) for microtomographic image restoration. | [download](-) | [download](-) |

## <a name="quick_start"></a>:flight_departure:Quick Start

 Download [general_full_v1.ckpt](-) and [general_swinir_v1.ckpt](-) to `weights/`, then run the following command to interact with the gradio website. 

```shell
python gradio_diffbir.py \
--ckpt weights/general_full_v1.ckpt \
--config configs/model/cldm.yaml \
--reload_swinir \
--swinir_ckpt weights/general_swinir_v1.ckpt \
```

<div align="center">
    <kbd><img src="assets/gradio.png"></img></kbd>
</div>

## <a name="colab"></a>:colab demo
[Inference + Train on google colab](https://colab.research.google.com/gist/pihchikk/ea1f01bdd70345dbcaa5d5965e5dfa6a/diffbir-inference-train.ipynb)


## <a name="inference"></a>:Inference

### Full Pipeline (Remove Degradations & Refine Details)

<a name="general_image_inference"></a>
#### General Image

Download [general_full_v1.ckpt](https://huggingface.co/lxq007/DiffBIR/resolve/main/general_full_v1.ckpt) and [general_swinir_v1.ckpt](https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt) to `weights/` and run the following command.

```shell
python inference.py \
--input inputs/demo/general \
--config configs/model/cldm.yaml \
--ckpt weights/general_full_v1.ckpt \
--reload_swinir --swinir_ckpt weights/general_swinir_v1.ckpt \
--steps 50 \
--sr_scale 4 \
--color_fix_type wavelet \
--output results/demo/general \
--device cuda [--tiled --tile_size 512 --tile_stride 256]
```

Remove the brackets to enable tiled sampling. If you are confused about where the `reload_swinir` option came from, please refer to the [degradation details](#degradation-details).

### Only Stage1 Model (Remove Degradations)

Download [general_swinir_v1.ckpt](https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt), [face_swinir_v1.ckpt](https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt) for general, face image respectively, and run the following command.

```shell
python scripts/inference_stage1.py \
--config configs/model/swinir.yaml \
--ckpt [swinir_ckpt_path] \
--input [lq_dir] \
--sr_scale 1 --image_size 512 \
--output [output_dir_path]
```

### Only Stage2 Model (Refine Details)

Since the proposed two-stage pipeline is very flexible, you can utilize other awesome models to remove degradations instead of SwinIR and then leverage the Stable Diffusion to refine details.

```shell
# step1: Use other models to remove degradations and save results in [img_dir_path].

# step2: Refine details of step1 outputs.
python inference.py \
--config configs/model/cldm.yaml \
--ckpt [full_ckpt_path] \
--steps 50 --sr_scale 1 \
--input [img_dir_path] \
--color_fix_type wavelet \
--output [output_dir_path] \
--disable_preprocess_model \
--device cuda
```

## <a name="train"></a>:stars:Train

### Degradation Details

For general image restoration, we first train both the stage1 and stage2 model under codeformer degradation to enhance the generative capacity of the stage2 model. In order to improve the ability for degradation removal, we train another stage1 model under Real-ESRGAN degradation and utilize it during inference.

For face image restoration, we adopt the degradation model used in [DifFace](https://github.com/zsyOAOA/DifFace/blob/master/configs/training/swinir_ffhq512.yaml) for training and directly utilize the SwinIR model released by them as our stage1 model.

### Data Preparation

1. Generate file list of training set and validation set.

    ```shell
    python scripts/make_file_list.py \
    --img_folder [hq_dir_path] \
    --val_size [validation_set_size] \
    --save_folder [save_dir_path] \
    --follow_links
    ```
    
    This script will collect all image files in `img_folder` and split them into training set and validation set automatically. You will get two file lists in `save_folder`, each line in a file list contains an absolute path of an image file:
    
    ```
    save_folder
    ├── train.list # training file list
    └── val.list   # validation file list
    ```

2. Configure training set and validation set.

    For general image restoration, fill in the following configuration files with appropriate values.

    - [training set](configs/dataset/general_deg_codeformer_train.yaml) and [validation set](configs/dataset/general_deg_codeformer_val.yaml) for **CodeFormer** degradation.
    - [training set](configs/dataset/general_deg_realesrgan_train.yaml) and [validation set](configs/dataset/general_deg_realesrgan_val.yaml) for **Real-ESRGAN** degradation.

    For face image restoration, fill in the face [training set](configs/dataset/face_train.yaml) and [validation set](configs/dataset/face_val.yaml) configuration files with appropriate values.

### Train Stage1 Model

1. Configure training-related information.

    Fill in the configuration file of [training](configs/train_swinir.yaml) with appropriate values.

2. Start training.

    ```shell
    python train.py --config [training_config_path]
    ```

    :bulb::Checkpoints of SwinIR will be used in training stage2 model.

### Train Stage2 Model

1. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) to provide generative capabilities.

    ```shell
    wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
    ```

2. Create the initial model weights.

    ```shell
    python scripts/make_stage2_init_weight.py \
    --cldm_config configs/model/cldm.yaml \
    --sd_weight [sd_v2.1_ckpt_path] \
    --swinir_weight [swinir_ckpt_path] \
    --output [init_weight_output_path]
    ```

    You will see some [outputs](assets/init_weight_outputs.txt) which show the weight initialization.

3. Configure training-related information.

    Fill in the configuration file of [training](configs/train_cldm.yaml) with appropriate values.

4. Start training.

    ```shell
    python train.py --config [training_config_path]
    ```



## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This project is based on [DiffBIR](https://github.com/XPixelGroup/DiffBIR/) which is itself based on [ControlNet](https://github.com/lllyasviel/ControlNet) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome work.

## Contact

If you have any questions, please feel free to contact with me at bardashovdr@my.msu.ru
