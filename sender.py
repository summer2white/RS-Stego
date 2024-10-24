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

from RoSteALS.tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid
import lpips
from RoSteALS.tools.sifid import SIFID


def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    #image = transforms.CenterCrop(min(x,y))(image)
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
    
    def load_target_features():
        target_features = []

        time_range = np.flip(sampler.ddim_timesteps)#T->0
        total_steps = sampler.ddim_timesteps.shape[0]

        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)
        #测试改变attension map对生成图片影响
        change_attn=True
        
        for i, t in enumerate(iterator):
            current_features = {}
            # for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
            #     if i <= int(output_block_self_attn_map_injection_threshold):
            #         output_q = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
            #         output_k = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
            #         current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
            #         current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

            # for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
            #     if i <= int(feature_injection_threshold):
            #         output = torch.load(os.path.join(source_experiment_out_layers_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
            #         current_features[f'output_block_{output_block_idx}_out_layers'] = output
            
            if change_attn==True:        
                if i==len(time_range)-1:
                    #secret=torch.rand(16, 4096, 4096).to(device)
                    secret=np.random.randint(0,2,(1,4096,1))
                    secret=torch.from_numpy(secret)
                    secret=torch.load("secret/secret_attn_start.pt")
                    torch.save(secret,'secret/secret01.pt')
                    inject=((secret-0.5)*2).to(device)
                    #inject=inject
                    zeros_8=torch.zeros(8,4096,1).to(device)
                    change_16=torch.cat((zeros_8,inject,inject,inject,inject,inject,inject,inject,inject),0)
                    #print('inject_size=',change_16.shape)
                    current_features[f'output_block_11_self_attn_k'] =inject
                    current_features[f'output_block_11_self_attn_q'] = None

            target_features.append(current_features)

        return target_features

    assert exp_config.config.prompt is not None
    prompts = [exp_config.config.prompt]
    injected_features = load_target_features()

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
                                injected_features=injected_features,
                                callback_ddim_timesteps=save_feature_timesteps,
                                outpath=outpath)
                
                atten_map=intermediates['self-atten-map-11'][-1]
                torch.save(atten_map,'secret/atten_map_run.pt')
                atten_map_ori=intermediates['self-atten-map-ori-11'][-1]
                torch.save(atten_map_ori,'secret/atten_map_ori_run.pt')
                torch.save(samples_ddim,'secret/steg_64_attn_sender.pt')
                
                #4x64x64 add noise
                # secret=np.random.randint(0,2,(4,64,64))
                # secret=torch.from_numpy(secret)
                # torch.save(secret,'secret/secret01.pt')
                # inject=(secret-0.5).to(device)
                # inject=0.5*inject
                # samples_ddim=samples_ddim+inject
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                #x_out=x_samples_ddim.clone().cpu()
                #torch.save(x_out,'secret/x_out.pt')
                x_sample_1=x_samples_ddim.clone()
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
                    img.save(os.path.join(sample_path, f"{sample_idx}.png"))
                    sample_idx += 1
                    
                receive_image = load_img('experiments/horse_in_mud/samples/0.png').to(device)
                diff=torch.sum(torch.abs(receive_image-x_sample_1[0]))
                print('-1,1 -> 0,255 512 diff=',diff)
                
                distribution=model.encode_first_stage(receive_image)
                torch.save(distribution.mean,'secret/encode.mean.pt')
                torch.save(distribution.std,'secret/encode.std.pt')
                print("mean.shape=",distribution.mean.shape,"std.shape=",distribution.std.shape)
                
                secret=np.random.randint(0,2,(1,4,64,64))
                secret=torch.from_numpy(secret)
                
                torch.save(secret,'secret/secret_64.pt')
                
                inject=((secret-0.5)*1.5*2000).to(device)
                #zeros=torch.zeros(1,1,64,64).to(device)
                #inject=torch.cat((inject,zeros),1)
                
                latent_sample = model.get_first_stage_encoding(distribution)
                latent_mode=0.18215*distribution.mode()
                diff=torch.sum(torch.abs(latent_sample-samples_ddim))
                print('encode.sample decode 64 diff=',diff)
                diff=torch.sum(torch.abs(latent_mode-samples_ddim))
                print('encode.mode decode 64 diff=',diff)
                
                x = 0.18215*(distribution.mean + distribution.std *inject)
                x_samples_ddim = model.decode_first_stage(x)                    
                diff=torch.sum(torch.abs(x_samples_ddim-x_sample_1[0]))
                print('encode.mode decode 512 diff=',diff)
                torch.save(x_samples_ddim,'secret/x_out.pt')
                
                    
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
                    img.save('experiments/horse_in_mud/samples/0_recond.png')
                    sample_idx += 1
                
    # with torch.enable_grad():
    #     with precision_scope("cuda"):
    #         # print('samples_ddim.shape',samples_ddim.shape)
    #         # samples_ddim.requires_grad_(True)
    #         # print('samples_ddim.mean=',torch.mean(torch.abs(samples_ddim)))
    #         # print(samples_ddim)
    #         # x_samples_ddim = model.differentiable_decode_first_stage(samples_ddim,samples_ddim)
            
    #         # print('x_samples_ddim.shape=',x_samples_ddim.shape)
            
    #         receive_image = load_img('experiments/horse_in_mud/translations/5.0_a_photo_of_a_horse_in_mud/INJECTION_T_40_STEPS_50_NP-ALPHA_0.75_SCHEDULE_linear_NP_a_photo_of_a_horse_in_mud_sample_0.png').to(device)     
    #         atten_map.requires_grad_(True) 
    #         x_1.requires_grad_(True)
    #         atten_map.retain_grad()
            
    #         #loss=sampler.loss_mse(x_samples_ddim, receive_image) 
    #         #loss=torch.sum(torch.abs(x_samples_ddim-receive_image) )

    #         #grad_cond = torch.autograd.grad(loss.requires_grad_(True), [samples_ddim], retain_graph=True, allow_unused=True)[0]
    #         #print('samples_ddim.grad_cond=',torch.mean(torch.abs(grad_cond)))
    #         #loss.backward()
    #         #print('samples_ddim.grad',torch.mean(torch.abs(samples_ddim.grad)))
    #         del samples_ddim
    #         torch.cuda.empty_cache()
    #         secret= torch.load('secret/secret.pt')
    #         print('x_1.shape',x_1.shape)
    #         atten_map_random=torch.randn_like(atten_map,requires_grad=True)
    #         rec_atten_map=sampler.gradient_descent(atten_map_random,receive_image,x_1,secret,device,unconditional_guidance_scale=exp_config.config.scale)
    #         diff=torch.mean(torch.abs(rec_atten_map-atten_map-secret))
    # print('rec diff=',diff)
    print(f"Sampled images and extracted features saved in: {outpath}")


if __name__ == "__main__":
    main()
