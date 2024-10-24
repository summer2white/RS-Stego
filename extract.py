import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
from torchvision import transforms
import logging
from pnp_utils import check_safety

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import cv2

def gass(output,sigma):
    output = np.asarray(output)
    mean = 0
    #设置高斯分布的标准差
    #根据均值和标准差生成符合高斯分布的噪声
    gauss = np.random.normal(mean,sigma,output.shape)
    #给图片添加高斯噪声
    gass_img = output + gauss
    #设置图片添加高斯噪声之后的像素值的范围
    gass_img = np.clip(gass_img,a_min=0,a_max=255)
    gass_img=Image.fromarray(np.uint8(gass_img))
    return gass_img
def salt(output,amount):
    output=np.asarray(output)
    #设置添加椒盐噪声的数目比例
    s_vs_p = 0.5
    #设置添加噪声图像像素的数目
    amount = amount/2
    salt_img = np.copy(output)
    #添加salt噪声
    num_salt = np.ceil(amount * output.size * s_vs_p)
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_salt)) for i in output.shape]
    salt_img[coords[0],coords[1],:] = [255,255,255]
    #添加pepper噪声
    num_pepper = np.ceil(amount * output.size * (1. - s_vs_p))
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in output.shape]
    salt_img[coords[0],coords[1],:] = [0,0,0]
    salt_img=Image.fromarray(np.uint8(salt_img))
    return salt_img

def poisson(output):
    output=np.asarray(output)
    #计算图像像素的分布范围
    vals = len(np.unique(output))
    vals = 2 ** np.ceil(np.log2(vals))
    #给图片添加泊松噪声
    poisson_img = np.random.poisson(output * vals) / float(vals)
    poisson_img=Image.fromarray(np.uint8(poisson_img))
    return poisson_img

def speckle(output):
    output=np.asarray(output)
    #随机生成一个服从分布的噪声
    gauss = np.random.randn(output.shape)
    #给图片添加speckle噪声
    noisy_img = output + output * gauss
    #归一化图像的像素值
    noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
    speckle_img=Image.fromarray(np.uint8(noisy_img))
    return speckle_img

def mean_filter(output,size=(3,3)):
    output=np.asarray(output)
    mean_filtered = cv2.blur(output, size)
    mean_img=Image.fromarray(np.uint8(mean_filtered))
    return mean_img

def median_filter(output,size=(3,3)):
    output=np.asarray(output)
    mean_filtered = cv2.medianBlur(output, size)
    mean_img=Image.fromarray(np.uint8(mean_filtered))
    return mean_img

def gass_filter(output,size=(3,3)):
    output=np.asarray(output)
    mean_filtered = cv2.GaussianBlur(output, size,0)
    mean_img=Image.fromarray(np.uint8(mean_filtered))
    return mean_img
    

def Image2tensor(image):
    w=h=512
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_img(path,add_noise=False):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    #add_noise=True
    #image = transforms.CenterCrop(min(x,y))(image)
    if add_noise:
        image_gass=gass(image,0.1)
        #image_poisson=poisson(image)
        image_salt=salt(image,0.1)
        
        image_mean=mean_filter(image,(7,7))
        image_median=median_filter(image,7)
        image_gassf=gass_filter(image,(7,7))
        
        image_gass.save(os.path.join('/root/autodl-tmp/plug-and-play/plug-and-play/experiments/horse_in_mud/samples', "gass.png"))
        image_salt.save(os.path.join('/root/autodl-tmp/plug-and-play/plug-and-play/experiments/horse_in_mud/samples', "salt.png"))
        image_mean.save(os.path.join('/root/autodl-tmp/plug-and-play/plug-and-play/experiments/horse_in_mud/samples', "mean.png"))
        image_gassf.save(os.path.join('/root/autodl-tmp/plug-and-play/plug-and-play/experiments/horse_in_mud/samples', "gassf.png"))
        image_median.save(os.path.join('/root/autodl-tmp/plug-and-play/plug-and-play/experiments/horse_in_mud/samples', "median.png"))
        
        image_gass=Image2tensor(image_gass)
        #image_poisson=Image2tensor(image_poisson)
        image_salt=Image2tensor(image_salt)
        image_mean=Image2tensor(image_mean)
        image_median=Image2tensor(image_median)
        image_gassf=Image2tensor(image_gassf)
        image=Image2tensor(image)
        
        return image,image_gass,image_salt,image_mean,image_median,image_gassf
    
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/pnp/feature-extraction-generated.yaml",
        help="path to the feature extraction config file"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--save_all_features",
        action="store_true",
        help="if set to true, saves all feature maps, otherwise only saves those necessary for PnP",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--check-safety",
        action='store_true',
    )

    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/pnp/setup.yaml")
    model_config = OmegaConf.load(f"{opt.model_config}")
    exp_config = OmegaConf.load(f"{opt.config}")
    exp_path_root = setup_config.config.exp_path_root

    if exp_config.config.init_img != '':
        exp_config.config.seed = -1
        exp_config.config.prompt = ""
        exp_config.config.scale = 1.0
        
    seed = exp_config.config.seed 
    seed_everything(seed)

    model = load_model_from_config(model_config, f"{opt.ckpt}")#ldm.models.diffusion.ddpm.LatentDiffusion

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    # vae_model=model.first_stage_model
    # print(vae_model)
    sampler = DDIMSampler(model)
    save_feature_timesteps = exp_config.config.ddim_steps if exp_config.config.init_img == '' else exp_config.config.save_feature_timesteps

    outpath = f"{exp_path_root}/{exp_config.config.experiment_name}"

    callback_timesteps_to_save = [save_feature_timesteps]
    if os.path.exists(outpath):
        logging.warning("Experiment directory already exists, previously saved content will be overriden")
        if exp_config.config.init_img != '':
            with open(os.path.join(outpath, "args.json"), "r") as f:
                args = json.load(f)
            callback_timesteps_to_save = args["save_feature_timesteps"] + callback_timesteps_to_save

    predicted_samples_path = os.path.join(outpath, "predicted_samples")
    feature_maps_path = os.path.join(outpath, "feature_maps")
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(predicted_samples_path, exist_ok=True)
    os.makedirs(feature_maps_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    # save parse_args in experiment dir
    with open(os.path.join(outpath, "args.json"), "w") as f:
        args_to_save = OmegaConf.to_container(exp_config.config)
        args_to_save["save_feature_timesteps"] = callback_timesteps_to_save
        json.dump(args_to_save, f)

    def save_sampled_img(x, i, save_path):
        x_samples_ddim = model.decode_first_stage(x)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
        x_sample = x_image_torch[0]
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(save_path, f"{i}.png"))

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_sampled_img(pred_x0, i, predicted_samples_path)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block in tqdm(blocks, desc="Saving input blocks feature maps"):
            if not opt.save_all_features and block_idx < 4:
                block_idx += 1
                continue
            if "ResBlock" in str(type(block[0])):
                if opt.save_all_features or block_idx == 4:
                    save_feature_map(block[0].in_layers_features, f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                save_feature_map(block[1].transformer_blocks[0].attn1.k, f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                save_feature_map(block[1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
            block_idx += 1

    def save_feature_maps_callback(i):
        if opt.save_all_features:
            save_feature_maps(unet_model.input_blocks, i, "input_block")
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename):
        save_path = os.path.join(feature_maps_path, f"{filename}.pt")
        torch.save(feature_map, save_path)

    assert exp_config.config.prompt is not None
    prompts = [exp_config.config.prompt]

    precision_scope = nullcontext#autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = model.get_learned_conditioning([""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                z_enc = None
                if exp_config.config.init_img != '':
                    assert os.path.isfile(exp_config.config.init_img)
                    init_image = load_img(exp_config.config.init_img).to(device)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
                    ddim_inversion_steps = 999
                    z_enc, _ = sampler.encode_ddim(init_latent, num_steps=ddim_inversion_steps, conditioning=c,unconditional_conditioning=uc,unconditional_guidance_scale=exp_config.config.scale)
                else:
                    #z_enc = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                    z_enc=torch.load('experiments/horse_in_mud/z_enc.pt')
                #torch.save(z_enc, f"{outpath}/z_enc.pt")
                print('unconditional_guidance_scale=',exp_config.config.scale)
                samples_ddim, intermediates = sampler.sample(S=exp_config.config.ddim_steps,
                                conditioning=c,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=exp_config.config.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=z_enc,
                                img_callback=None,
                                callback_ddim_timesteps=save_feature_timesteps,
                                outpath=outpath)

                atten_map=intermediates['self-atten-map-11'][-1]
                x_1=intermediates['x_1'][-1]
                torch.save(atten_map,'secret/atten_map_extract.pt')
                
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                print('opt.check_safety=',opt.check_safety)
                if opt.check_safety:
                    x_samples_ddim = check_safety(x_samples_ddim)
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                sample_idx = 0
                for x_sample in x_image_torch:
                    
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join("experiments/horse_in_mud/samples/extract.png"))
                    sample_idx += 1
                    
                receive_image = load_img('experiments/horse_in_mud/samples/extract.png').to(device)                
                distribution=model.encode_first_stage(receive_image)
                
    with torch.enable_grad():
        with precision_scope("cuda"):
            # print('samples_ddim.shape',samples_ddim.shape)
            # samples_ddim.requires_grad_(True)
            # print('samples_ddim.mean=',torch.mean(torch.abs(samples_ddim)))
            # print(samples_ddim)
            # x_samples_ddim = model.differentiable_decode_first_stage(samples_ddim,samples_ddim)
            
            # print('x_samples_ddim.shape=',x_samples_ddim.shape)
            
            receive_image,image_gass, image_salt,image_mean,image_median,image_gassf= load_img('experiments/horse_in_mud/samples/0.png',add_noise=True)#.to(device)     
            
            path='experiments/horse_in_mud/samples/0.png'
            image = Image.open(path).convert("RGB")
            image.save('image_10.jpeg', quality=10)
            image.save('image_30.jpeg', quality=30)
            image.save('image_50.jpeg', quality=50)
            image.save('image_70.jpeg', quality=70)
            image.save('image_90.jpeg', quality=90)
            image_10=load_img("image_10.jpeg")
            image_30=load_img("image_30.jpeg")
            image_50=load_img("image_50.jpeg")
            image_70=load_img("image_70.jpeg")
            image_90=load_img("image_90.jpeg")
            image_10=image_10.to(device)
            image_30=image_30.to(device)
            image_50=image_50.to(device)
            image_70=image_70.to(device)
            image_90=image_90.to(device)
            
            receive_image=receive_image.to(device)   
            image_gass=image_gass.to(device)   
            image_salt=image_salt.to(device)   
            #image_poisson=image_poisson.to(device)  
            image_mean=image_mean.to(device)
            image_median=image_median.to(device)
            image_gassf=image_gassf.to(device)
             
            atten_map.requires_grad_(True) 
            x_1.requires_grad_(True)
            atten_map.retain_grad()
            samples_ddim.requires_grad_(True)
            
            #loss=sampler.loss_mse(x_samples_ddim, receive_image) 
            #loss=torch.sum(torch.abs(x_samples_ddim-receive_image) )

            #grad_cond = torch.autograd.grad(loss.requires_grad_(True), [samples_ddim], retain_graph=True, allow_unused=True)[0]
            #print('samples_ddim.grad_cond=',torch.mean(torch.abs(grad_cond)))
            #loss.backward()
            #print('samples_ddim.grad',torch.mean(torch.abs(samples_ddim.grad)))
            del samples_ddim #atten_map
            torch.cuda.empty_cache()
            secret= torch.load('secret/secret01.pt').to(device)
            x_out=torch.load('secret/x_out.pt').to(device)
            #secret=torch.from_numpy(secret).to(device).float()
            print('x_1.shape',x_1.shape)
            n=1
            #atten_map_random=torch.randn_like(atten_map,requires_grad=True)
            torch.save(receive_image[0],"secret/receive_512_attn_sender.pt")
            change=sampler.gradient_descent_cross(atten_map,receive_image,x_1,secret,n,"loss.txt",device,unconditional_guidance_scale=exp_config.config.scale)
            #change=sampler.gradient_descent_64(distribution,receive_image,secret,device)
            rev_secret=(change>0).float()
            rev_num=torch.sum(torch.abs(rev_secret-secret))
            correct=1-rev_num/(n*4096)
            print('normal correct=',correct)

            test_jpge=False
            if test_jpge:
                change=sampler.gradient_descent_64(distribution,image_10,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_10,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('10 correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_30,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_30,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('30 correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_50,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_50,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('50 correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_70,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_70,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('70 correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_90,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_90,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('90 correct=',correct)
                
            test_noise=False
            if test_noise:
                change=sampler.gradient_descent_64(distribution,image_gass,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_gass,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('gass correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_salt,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_salt,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('salt correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_mean,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_mean,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('mean filter correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_median,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_median,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('median filter correct=',correct)
                
                change=sampler.gradient_descent_64(distribution,image_gassf,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                #change=sampler.gradient_descent_cross(atten_map,image_gassf,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
                rev_secret=(change>0).float()
                rev_num=torch.sum(torch.abs(rev_secret-secret))
                correct=1-rev_num/(n*4096)
                print('gass filter correct=',correct)
                
    #print('rec diff=',diff_num/(16*4096*4096))
    print(f"Sampled images and extracted features saved in: {outpath}")


if __name__ == "__main__":
    main()
