# Robust Generative Image Steganography based on Semantic Space


## Setup

Our codebase is built on [MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play)
and has shared dependencies and model architecture.

### Creating a Conda Environment

```
conda env create -f environment.yaml
conda activate RS-Stego
```

### Downloading StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```


### Setting Experiment Root Path

The data of all the experiments is stored in a root directory.
The path of this directory is specified in `configs/pnp/setup.yaml`, under the `config.exp_path_root` key.


## Hiding and Extracting

For generating stego image, first set the parameters for the translation in a yaml config file.
An example of generation configs can be found in
 `configs/pnp/Steganography.yaml` for generated images. Once the arguments are set, run:

```
python start.py --config <extraction_config_path>
python start.py --config <config_path> --hiding  --extract
```