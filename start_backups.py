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

from tool.hiding import hide
from tool.attack import load_img_attack

from RoSteALS.tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid
import lpips
from RoSteALS.tools.sifid import SIFID


def load_img(path,sample_path=None,add_noise=False):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    
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

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose=False) 
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
    
    def load_target_features(change_attn=False):
        target_features = []

        time_range = np.flip(sampler.ddim_timesteps)#T->0
        total_steps = sampler.ddim_timesteps.shape[0]

        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)
        #测试改变attension map对生成图片影响
        
        for i, t in enumerate(iterator):
            current_features = {}            
            if change_attn==True:        
                if i==len(time_range)-1:
                    secret=np.random.randint(0,2,(2,4096,1))
                    secret=torch.from_numpy(secret)
                    torch.save(secret,'secret/secret01.pt')
                    inject=((secret-0.5)*2).to(device)                    
                    change=inject
                    #cross-attention map size=16,4096,77
                    zeros4=torch.zeros(8,4096,1).to(device)
                    change_16=torch.cat((zeros4,change,change,change,change),0)
                    print('inject_size=',change_16.shape)
                    current_features[f'output_block_11_self_attn_k'] = change_16
                    current_features[f'output_block_11_self_attn_q'] = None

            target_features.append(current_features)

        return target_features

    assert exp_config.config.prompt is not None
    prompts = [exp_config.config.prompt]
    
    change_attn=False
    injected_features = load_target_features(change_attn)

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
                                injected_features=injected_features,
                                callback_ddim_timesteps=save_feature_timesteps,
                                outpath=outpath)
                
                atten_map=intermediates['self-atten-map-11'][-1]
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                
                cover_org=255*x_samples_ddim.permute(0,2,3,1).cpu().numpy()
                cover_org=cover_org[0,:,:,:]

                sample_idx = 0
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path,"ori.png"))
                    sample_idx += 1
                
                path=os.path.join(sample_path,"ori.png")    
                receive_image = load_img(path).to(device)                
                distribution=model.encode_first_stage(receive_image)
                #4x64x64 add noise
                add_secret_latent=True
                n=1
                if add_secret_latent==True:
                    method=["D-RM"]
                    q=2000
                    lambda_a=2
                    lambda_l=0.5
                    secret=np.random.randint(0,2,(1,n,64,64))
                    secret=torch.from_numpy(secret).to(device)
                    torch.save(secret,'secret/secret_latent.pt')                    
                    secret_attn=secret.reshape(n,4096,1)
                    torch.save(secret_attn,'secret/secret_attn_start.pt')
                    steg_64,change_attn=hide(method,secret,secret_attn,n,q,lambda_a,lambda_l,samples_ddim,distribution,device)
                    
                    for m in method:
                        if m=="SM":
                            x_1=intermediates['x_1'][-1]
                            steg_64_attn=sampler.hiding_cross_quality(atten_map,x_1,change_attn,device,unconditional_guidance_scale=exp_config.config.scale)
                            if steg_64==None:
                                print(change_attn.shape)
                                steg_64=steg_64_attn
                                torch.save(steg_64_attn,"secret/steg_64_attn_start.pt")
                            else:
                                steg_64=torch.cat((steg_64,steg_64_attn),0)

                    x_samples_ddim = model.differentiable_decode_first_stage(steg_64)
                    #torch.save(x_samples_ddim[2,:,:,:],"secret/x_512_attn_start.pt")
                    torch.save(x_samples_ddim,"secret/x_out_start.pt")    
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
                        image_name="steg_"+str(n)+'_'+method[sample_idx]+".png"
                        img.save(os.path.join(sample_path, image_name))
                        sample_idx += 1
                        
                    stego=255*x_image_torch.permute(0,2,3,1).cpu().numpy()
                    lpips_alex = lpips.LPIPS(net='alex').cuda()
                    sifid_model = SIFID()   
                    sample_idx = 0
                    for stego_uint8 in stego:
                        # quality metrics
                        #stego_uint8=stego_uint[np.newaxis,:]

                        print(method[sample_idx],f'Quality metrics at resolution: 512x512 (HxW)')
                        print(f'MSE: {compute_mse(np.array(cover_org)[None,...], stego_uint8[None,...])}')
                        print(f'PSNR: {compute_psnr(np.array(cover_org)[None,...], stego_uint8[None,...])}')
                        print(f'SSIM: {compute_ssim(np.array(cover_org)[None,...], stego_uint8[None,...])}')
                        cover_org_norm = torch.from_numpy(np.array(cover_org[None,...])/127.5-1.).permute(0,3,1,2).float().cuda()
                        stego_norm = torch.from_numpy(stego_uint8[None,...]/127.5-1.).permute(0,3,1,2).float().cuda()
                        print(f'LPIPS: {compute_lpips(cover_org_norm, stego_norm, lpips_alex)}')
                        print(f'SIFID: {compute_sifid(cover_org_norm, stego_norm, sifid_model)}')
                        diff=np.abs(cover_org-stego_uint8)
                        #residual=(diff-np.min(diff))/(np.max(diff)-np.min(diff))*255
                        img=Image.fromarray(diff.astype(np.uint8))
                        image_name="steg_"+str(n)+'_'+method[sample_idx]+"res.png"
                        img.save(os.path.join(sample_path, image_name))
                        sample_idx=sample_idx+1

    
    extract=False     
    if extract==True:            
        with torch.enable_grad():
            with precision_scope("cuda"):
                num=n*4096
                for m in method:
                    path=os.path.join(sample_path, f"steg_{n}_{m}.png")
                    attack_dict={"gass":0.01,"salt":0.01,"mean_filter":3,"median_filter":3,"gass_filter":3,"JPEG":90}
                    receive_image= load_img_attack(path,sample_path,m,attack_dict,add_noise=False)
            
                    receive_image=receive_image.to(device)   
                    
                    torch.cuda.empty_cache()
                    txt_path="loss_all.txt"
                    if m=="D-RM":
                        change_rev=sampler.gradient_descent_64_direct(samples_ddim,receive_image[0],secret,n,txt_path,device)
                        c_secret=secret
                    elif m=="STD-RM":
                        change_rev=sampler.gradient_descent_64_std(distribution,receive_image[0],secret,n,q,txt_path,device)
                        c_secret=secret
                    elif m=="SM":
                        torch.save(receive_image[0],"secret/receive_512_attn_start.pt")
                        x_out=torch.load("secret/x_out_start.pt")
                        change_rev=sampler.gradient_descent_cross(atten_map,receive_image[0],x_1,secret_attn,n,txt_path,device,unconditional_guidance_scale=exp_config.config.scale)
                        c_secret=secret_attn
                    rev_secret=(change_rev>0).float()
                    rev_num=torch.sum(torch.abs(rev_secret-c_secret))
                    correct=1-rev_num/num
                    print(f'{m} correct=',correct)
                    test_noise=False
                    if test_noise:
                        ii=1
                        for key ,value in attack_dict.items():
                            image_input=receive_image[ii]
                            if m=="D-RM":
                                change=sampler.gradient_descent_64_direct(samples_ddim,image_input,secret,n,txt_path,device)
                        
                            elif m=="STD-RM":
                                change=sampler.gradient_descent_64_std(distribution,image_input,secret,n,q,txt_path,device)
                        
                            elif m=="SM":
                                change=sampler.gradient_descent_cross(atten_map,image_input,x_1,secret_attn,n,txt_path,device,unconditional_guidance_scale=exp_config.config.scale)
                            rev_secret=(change>0).float()
                            rev_num=torch.sum(torch.abs(rev_secret-c_secret))
                            correct=1-rev_num/num
                            print(f'{key}_{value} correct=',correct)
                            ii=ii+1
                  
    print(f"Sampled images and extracted features saved in: {outpath}")


if __name__ == "__main__":
    main()
