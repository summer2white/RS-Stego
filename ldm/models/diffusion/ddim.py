"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
    
from torch import nn, einsum
from einops import rearrange

from torch.optim import Adam


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.loss_L1 = torch.nn.L1Loss()
        self.loss_mse=torch.nn.MSELoss()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, strength = 1.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                    num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose, strength=strength)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def make_negative_prompt_schedule(self, negative_prompt_schedule, negative_prompt_alpha, total_steps):
        if negative_prompt_schedule == 'linear':
            negative_prompt_schedule = np.flip(np.linspace(0, 1, total_steps))
        elif negative_prompt_schedule == 'constant':
            negative_prompt_schedule = np.flip(np.ones(total_steps))
        elif negative_prompt_schedule == 'exp':
            negative_prompt_schedule = np.exp(-6 * np.linspace(0, 1, total_steps))
        else:
            raise NotImplementedError

        negative_prompt_schedule = negative_prompt_schedule * negative_prompt_alpha

        return negative_prompt_schedule
    
    def gradient_descent(self,A,receive_image,x_1,secret,device,unconditional_guidance_scale=1):
        print('A.size=',A.shape)
        unet_model=self.model.model.diffusion_model
        blocks=unet_model.output_blocks
        block=blocks[11]
        atten_map=A.clone().detach()
        # for k,v in block.named_parameters():
        #     v.requires_grad=True
        # for k,v in unet_model.out.named_parameters():
        #     v.requires_grad=True
        # #self.model.decode_first_stage中网络
        # for param in self.model.first_stage_model.parameters():
        #     param.requires_grad = True
        #     #print('self.model.first_stage_model.parameters()')
        # #autoencoder.py self.model.first_stage_model=AutoencoderKL
        # for param in self.model.first_stage_model.post_quant_conv.parameters():
        #     param.requires_grad=True
        # for param in self.model.first_stage_model.decoder.parameters():
        #     param.requires_grad=True

        #block[1]==spatial transformer block
        #block[1].transformer_blocks[0].attn1==self-attention
        #modules/attention.py
        #A=block[1].transformer_blocks[0].attn1.attn
        v=block[1].transformer_blocks[0].attn1.v#不变
        basictran_in_features=block[1].transformer_blocks[0].in_layers_features#不变
        change=torch.zeros(1,4096,1,requires_grad=True).to(device)+0.25
        loss=9999
        old_loss=99999
        lr=1
        num=4096
        i=0
        #A=torch.nn.Parameter(A)
        # grad_model=[A]
        # opt=Adam(grad_model,lr=1)
        
        while i<100:
            #opt.zero_grad()
            # inject=(secret-0.5).to(device)
            # inject=inject*0.0001
            attn=A+change
            #attn=A.softmax(dim=-1)
            # z=torch.mean(attn)
            # grad_cond_attn = torch.autograd.grad(z, [A], retain_graph=True, allow_unused=True)[0]
            # print('softmax/A',torch.mean(torch.abs((grad_cond_attn))))
            # del z,grad_cond_attn
            
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=8)
            out=block[1].transformer_blocks[0].attn1.to_out(out)
            #出self-attention
            #print('out.size=',out.shape,'basic_in_feature.size=',basictran_in_features.shape)
            
            x = out+ basictran_in_features
            #cross-attention
            x = block[1].transformer_blocks[0].attn2(block[1].transformer_blocks[0].norm2(x), context=block[1].transformer_blocks[0].context) + x
            #ff
            x = block[1].transformer_blocks[0].ff(block[1].transformer_blocks[0].norm3(x)) + x
            #print('ff out.shape=',x.shape)
            #出basic transformer block
            sptran_in_feature=block[1].in_feature#不变
            b, c, h, w = sptran_in_feature.shape
            #print('sptran_in_feature.shape=',sptran_in_feature.shape)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = block[1].proj_out(x)
            x=x + sptran_in_feature
            #出spatial transformer
            #block[2]==upsample
            #openaimodel.py
            if len(block)==3:
                x=block[2](x)
            #进入GSC
            x = x.type(unet_model.dtype)
            x=unet_model.out(x)
            #输出
            #ddim.py DDIMSampler
            e_t_uncond, e_t = x.chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) if unconditional_guidance_scale!=1 else e_t
            #alphas = model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
            #use_original_steps 在ddim.py self.ddim_sampling中=False
            alphas =  self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            b=x_1.shape[0]
            a_t = torch.full((b, 1, 1, 1), alphas[0], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[0], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[0], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[0],device=device)

            # current prediction for x_0
            pred_x0 = (x_1 - sqrt_one_minus_at * e_t) / a_t.sqrt()
            #sample中quantize_x0=False,略过此步骤：
            # if quantize_denoised:
            #     pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            #repeat_noise=False
            #temperature=1
            #noise = sigma_t * noise_like(x_1.shape, device, repeat_noise) * temperature
            #sample中noise_dropout=0,略过此步骤：
            # if noise_dropout > 0.:
            #     noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt #+ noise
            
            #z=torch.mean(x_prev)
            #grad_cond = torch.autograd.grad(z, [A], retain_graph=True, allow_unused=True)[0]
            # print('x_prev/A',torch.sum(grad_cond))
            # print('x_prev',x_prev.requires_grad)
            
            #return x_prev, pred_x0
            #print('x_prev.shape',x_prev.shape)
            
            x_samples_ddim = self.model.differentiable_decode_first_stage(x_prev)
            
            #z=torch.mean(x_samples_ddim)
            #grad_cond = torch.autograd.grad(z, [A], retain_graph=True, allow_unused=True)[0]
            # print('x_sample_ddim/A',torch.sum(grad_cond))
            # print('x_samples_ddim',x_samples_ddim.requires_grad)
            print('x_sample_ddim.size=',x_samples_ddim.shape)
            print('receive_image.size=',receive_image.shape)
            
            loss=torch.sum(torch.abs(x_samples_ddim-receive_image) )
            
            print('loss=',loss,loss.requires_grad)
            #loss.backward()
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [change], retain_graph=False, allow_unused=False)[0]
            #print('A.grad=',torch.mean(grad_cond))

            rev_secret=(change>0).float()
            # print('rev_secret.size=',rev_secret.shape)
            # print('secret.size=',secret.shape)
            rev_num=torch.sum(torch.abs(rev_secret-secret))
            correct=1-rev_num/num
            #diff=torch.mean(torch.abs(A-secret))
            #test=torch.mean(torch.abs(atten_map+secret))
            #print('rec diff=',torch.sum(torch.abs(diff)))
            with open("loss.txt","a") as f:
                f.writelines(f'第{i}次,loss_L1={loss},A.grad.mean={torch.mean(torch.abs(grad_cond))},corret={correct},lr={lr}')
                f.writelines("\n")
            i=i+1
            if loss-old_loss>10:
                lr=0.5*lr
            old_loss=loss
            change = change -lr* grad_cond
            #opt.step()
            self.model.zero_grad()
            del out, x,x_prev,x_samples_ddim,pred_x0 , dir_xt,rev_secret,attn,e_t_uncond, e_t 
            torch.cuda.empty_cache()
        
        # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        # if opt.check_safety:
        #     x_samples_ddim = check_safety(x_samples_ddim)
        # x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

        # sample_idx = 0
        # for k, x_sample in enumerate(x_image_torch):
        #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        #     img = Image.fromarray(x_sample.astype(np.uint8))
        #     img.save(os.path.join(outpaths[k], f"{out_label}_sample_{sample_idx}.png"))
        #     sample_idx += 1
        return A
    
    def gradient_descent_cross(self,A,receive_image,x_1,secret,n=1,txt_path="loss.txt",device="cpu",unconditional_guidance_scale=1):
        unconditional_guidance_scale=10
        unet_model=self.model.model.diffusion_model
        blocks=unet_model.output_blocks
        block=blocks[11]

        v=block[1].transformer_blocks[0].attn2.v#不变
        basictran_in_features=block[1].transformer_blocks[0].in_layers_features#不变
        change=torch.zeros(n,4096,1,requires_grad=True).to(device)+0.25
        
        loss=9999
        old_loss=99999
        lr=0.5
        num=4096
        i=0
        #A=torch.nn.Parameter(A)
        # grad_model=[A]
        # opt=Adam(grad_model,lr=1)

        while i<80:
            #opt.zero_grad()
            # inject=(secret-0.5).to(device)
            # inject=inject*0.0001
            zeros_8=torch.zeros(8,4096,1).to(device)
            change_16=torch.cat((zeros_8,change,change,change,change,change,change,change,change),0)
            attn=A+change
            #attn=A.softmax(dim=-1)
            # z=torch.mean(attn)
            # grad_cond_attn = torch.autograd.grad(z, [A], retain_graph=True, allow_unused=True)[0]
            # print('softmax/A',torch.mean(torch.abs((grad_cond_attn))))
            # del z,grad_cond_attn
            
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=8)
            out=block[1].transformer_blocks[0].attn2.to_out(out)
            #出self-attention
            #print('out.size=',out.shape,'basic_in_feature.size=',basictran_in_features.shape)
            
            x = out+ basictran_in_features
            #cross-attention
            #x = block[1].transformer_blocks[0].attn2(block[1].transformer_blocks[0].norm2(x), context=block[1].transformer_blocks[0].context) + x
            #ff
            x = block[1].transformer_blocks[0].ff(block[1].transformer_blocks[0].norm3(x)) + x
            #print('ff out.shape=',x.shape)
            #出basic transformer block
            sptran_in_feature=block[1].in_feature#不变
            b, c, h, w = sptran_in_feature.shape
            #print('sptran_in_feature.shape=',sptran_in_feature.shape)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = block[1].proj_out(x)
            x=x + sptran_in_feature
            #出spatial transformer
            #block[2]==upsample
            #openaimodel.py
            if len(block)==3:
                x=block[2](x)
            #进入GSC
            x = x.type(unet_model.dtype)
            x=unet_model.out(x)
            #输出
            #ddim.py DDIMSampler
            e_t_uncond, e_t = x.chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) if unconditional_guidance_scale!=1 else e_t
            #alphas = model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
            #use_original_steps 在ddim.py self.ddim_sampling中=False
            alphas =  self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            b=x_1.shape[0]
            a_t = torch.full((b, 1, 1, 1), alphas[0], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[0], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[0], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[0],device=device)

            # current prediction for x_0
            pred_x0 = (x_1 - sqrt_one_minus_at * e_t) / a_t.sqrt()
            #sample中quantize_x0=False,略过此步骤：
            # if quantize_denoised:
            #     pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            #repeat_noise=False
            #temperature=1
            #noise = sigma_t * noise_like(x_1.shape, device, repeat_noise) * temperature
            #sample中noise_dropout=0,略过此步骤：
            # if noise_dropout > 0.:
            #     noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt #+ noise
            
            #z=torch.mean(x_prev)
            #grad_cond = torch.autograd.grad(z, [A], retain_graph=True, allow_unused=True)[0]
            # print('x_prev/A',torch.sum(grad_cond))
            # print('x_prev',x_prev.requires_grad)
            
            #return x_prev, pred_x0
            #print('x_prev.shape',x_prev.shape)
            
            x_samples_ddim = self.model.differentiable_decode_first_stage(x_prev)
            
            loss=torch.sum(torch.abs(x_samples_ddim-receive_image))

            #x_samples_ddim_old=x_samples_ddim
            

            #loss.backward()
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [change], retain_graph=False, allow_unused=False)[0]
            #print('A.grad=',torch.mean(grad_cond))

            rev_secret=(change>0).float()
            rev_num=torch.sum(torch.abs(rev_secret-secret))
            correct=1-rev_num/num
            grad_mean=torch.mean(torch.abs(grad_cond))

            # with open(txt_path,"a") as f:
            #     f.writelines(f'第{i}次,loss_L1={loss},A.grad.mean={torch.mean(torch.abs(grad_cond))},corret={correct},lr={lr}')
            #     #f.writelines(f'第{i}次,loss_L1={loss},A.grad.mean={grad_mean},diff={torch.mean(torch.abs(change))},lr={lr}')
            #     f.writelines("\n")
            i=i+1
            if loss-old_loss>10:
                lr=0.5*lr

            old_loss=loss
            change = change -lr* grad_cond
            #opt.step()
            self.model.zero_grad()
            del out, x,x_prev,x_samples_ddim,pred_x0 , dir_xt,attn,e_t_uncond, e_t 
            torch.cuda.empty_cache()
        
        return change



    def gradient_descent_64(self,distribution,receive_image,secret,device):
        x_mean=distribution.mean
        x_std=2000*distribution.std
        loss=10000
        loss_last=200000
        i=0
        lr=0.04
        n=4
        change=torch.zeros(1,n,64,64,requires_grad=True).to(device)+0.25
        #A=torch.nn.Parameter(A)
        # grad_model=[A]
        # opt=Adam(grad_model,lr=1)
        # secret=secret*2+2
        # secret=torch.reshape(secret,(-1,))
        #zeros=torch.zeros(1,1,64,64).to(device)
        while  i<100:
            
            #inject=torch.cat((change,zeros),1)            
            x_prev=(x_mean+x_std*change)*0.18215
            x_samples_ddim = self.model.differentiable_decode_first_stage(x_prev)

            
            loss=torch.sum(torch.abs(x_samples_ddim-receive_image) )
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [change], retain_graph=False, allow_unused=False)[0]
            #print('A.grad=',torch.mean(grad_cond))
            #diff=x_prev-old_x_prev
            #rev_secret=(change>0).float()
            
            # re_change=torch.reshape(change,(1,-1))
            # dis=torch.cat((re_change+1,re_change+0.5,re_change,re_change-0.5,re_change-1),0)
            # _,indices=torch.min(torch.abs(dis),dim=0)
            # diff=indices-secret
            rev_secret=(change>0).float()
            diff=torch.abs(rev_secret-secret)
            rev_num=torch.sum(diff)
            correct=1-rev_num/n*(64*64)
            #diff=torch.mean(torch.abs(A-secret))
            #test=torch.mean(torch.abs(atten_map+secret))
            #print('rec diff=',torch.sum(torch.abs(diff)))
            # with open("loss.txt","a") as f:
            #     f.writelines(f'第{i}次,loss_L1={loss},A.grad.mean={torch.mean(torch.abs(grad_cond))},corret={correct},lr={lr}')
            #     f.writelines("\n")
            i=i+1
            if loss-loss_last>10:
                lr=lr*0.5
            change = change -lr* grad_cond
            loss_last=loss
            #opt.step()
            self.model.zero_grad()
            del x_samples_ddim,rev_secret 
            torch.cuda.empty_cache()
        
        return change
    
    def gradient_descent_64_std(self,distribution,receive_image,secret,n,q,txt_path,device):
        x_mean=distribution.mean
        x_std=q*distribution.std
        loss=10000
        loss_last=200000
        i=0
        lr=0.04
        change=torch.zeros(1,n,64,64,requires_grad=True).to(device)+0.25
        #A=torch.nn.Parameter(A)
        # grad_model=[A]
        # opt=Adam(grad_model,lr=1)
        # secret=secret*2+2
        # secret=torch.reshape(secret,(-1,))
        #zeros=torch.zeros(1,1,64,64).to(device)
        while  i<80:
            
            #inject=torch.cat((change,zeros),1)            
            x_prev=(x_mean+x_std*change)*0.18215
            x_samples_ddim = self.model.differentiable_decode_first_stage(x_prev)

            
            loss=torch.sum(torch.abs(x_samples_ddim-receive_image) )
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [change], retain_graph=False, allow_unused=False)[0]
            #print('A.grad=',torch.mean(grad_cond))
            #diff=x_prev-old_x_prev
            #rev_secret=(change>0).float()
            
            # re_change=torch.reshape(change,(1,-1))
            # dis=torch.cat((re_change+1,re_change+0.5,re_change,re_change-0.5,re_change-1),0)
            # _,indices=torch.min(torch.abs(dis),dim=0)
            # diff=indices-secret
            rev_secret=(change>0).float()
            diff=torch.abs(rev_secret-secret)
            rev_num=torch.sum(diff)
            correct=1-rev_num/n*(64*64)
            #diff=torch.mean(torch.abs(A-secret))
            #test=torch.mean(torch.abs(atten_map+secret))
            #print('rec diff=',torch.sum(torch.abs(diff)))
            # with open(txt_path,"a") as f:
            #     f.writelines(f'第{i}次,loss_L1={loss},A.grad.mean={torch.mean(torch.abs(grad_cond))},corret={correct},lr={lr}')
            #     f.writelines("\n")
            i=i+1
            if loss-loss_last>10:
                lr=lr*0.5
            change = change -lr* grad_cond
            loss_last=loss
            #opt.step()
            self.model.zero_grad()
            del x_samples_ddim,rev_secret 
            torch.cuda.empty_cache()
        
        return change
    
    def gradient_descent_64_direct(self,samples_ddim,receive_image,secret,n,txt_path,device):

        loss=10000
        loss_last=200000
        i=0
        lr=0.04
        change=torch.zeros(1,n,64,64,requires_grad=True).to(device)+0.25
        #A=torch.nn.Parameter(A)
        # grad_model=[A]
        # opt=Adam(grad_model,lr=1)
        # secret=secret*2+2
        # secret=torch.reshape(secret,(-1,))
        #zeros=torch.zeros(1,1,64,64).to(device)
        while  i<80:
            x_prev=samples_ddim+change
            x_samples_ddim = self.model.differentiable_decode_first_stage(x_prev)
            
            loss=torch.sum(torch.abs(x_samples_ddim-receive_image) )
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [change], retain_graph=False, allow_unused=False)[0]
            #print('A.grad=',torch.mean(grad_cond))
            #diff=x_prev-old_x_prev
            #rev_secret=(change>0).float()
            
            # re_change=torch.reshape(change,(1,-1))
            # dis=torch.cat((re_change+1,re_change+0.5,re_change,re_change-0.5,re_change-1),0)
            # _,indices=torch.min(torch.abs(dis),dim=0)
            # diff=indices-secret
            rev_secret=(change>0).float()
            diff=torch.abs(rev_secret-secret)
            rev_num=torch.sum(diff)
            correct=1-rev_num/n*(64*64)
            #diff=torch.mean(torch.abs(A-secret))
            #test=torch.mean(torch.abs(atten_map+secret))
            #print('rec diff=',torch.sum(torch.abs(diff)))
            # with open(txt_path,"a") as f:
            #     f.writelines(f'第{i}次,loss_L1={loss},A.grad.mean={torch.mean(torch.abs(grad_cond))},corret={correct},lr={lr}')
            #     f.writelines("\n")
            i=i+1
            if loss-loss_last>10:
                lr=lr*0.5
            change = change -lr* grad_cond
            loss_last=loss
            #opt.step()
            self.model.zero_grad()
            del x_samples_ddim,rev_secret 
            torch.cuda.empty_cache()
        
        return change
    
    def hiding_cross(self,A,x_1,secret,device,unconditional_guidance_scale=1):
        unconditional_guidance_scale=10
        unet_model=self.model.model.diffusion_model
        blocks=unet_model.output_blocks
        block=blocks[11]

        v=block[1].transformer_blocks[0].attn2.v#不变
        basictran_in_features=block[1].transformer_blocks[0].in_layers_features#不变
        #change=torch.zeros(n,4096,1,requires_grad=True).to(device)+0.25
        change=torch.load("secret/inject_attn_start.pt").to(device)
        change.requires_grad_(True)
        loss=9999
        old_loss=99999
        lr=0.5
        num=4096
        i=0
        #A=torch.nn.Parameter(A)
        # grad_model=[A]
        # opt=Adam(grad_model,lr=1)


        #opt.zero_grad()
        # inject=(secret-0.5).to(device)
        # inject=inject*0.0001
        zeros_8=torch.zeros(8,4096,1).to(device)
        change_16=torch.cat((zeros_8,change,change,change,change,change,change,change,change),0)
        attn=A+change_16
        #attn=A.softmax(dim=-1)
        # z=torch.mean(attn)
        # grad_cond_attn = torch.autograd.grad(z, [A], retain_graph=True, allow_unused=True)[0]
        # print('softmax/A',torch.mean(torch.abs((grad_cond_attn))))
        # del z,grad_cond_attn
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=8)
        out=block[1].transformer_blocks[0].attn2.to_out(out)
        #出self-attention
        #print('out.size=',out.shape,'basic_in_feature.size=',basictran_in_features.shape)
        
        x = out+ basictran_in_features
        #cross-attention
        #x = block[1].transformer_blocks[0].attn2(block[1].transformer_blocks[0].norm2(x), context=block[1].transformer_blocks[0].context) + x
        #ff
        x = block[1].transformer_blocks[0].ff(block[1].transformer_blocks[0].norm3(x)) + x
        #print('ff out.shape=',x.shape)
        #出basic transformer block
        sptran_in_feature=block[1].in_feature#不变
        b, c, h, w = sptran_in_feature.shape
        #print('sptran_in_feature.shape=',sptran_in_feature.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = block[1].proj_out(x)
        x=x + sptran_in_feature
        #出spatial transformer
        #block[2]==upsample
        #openaimodel.py
        if len(block)==3:
            x=block[2](x)
        #进入GSC
        x = x.type(unet_model.dtype)
        x=unet_model.out(x)
        #输出
        #ddim.py DDIMSampler
        e_t_uncond, e_t = x.chunk(2)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) if unconditional_guidance_scale!=1 else e_t
        #alphas = model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
        #use_original_steps 在ddim.py self.ddim_sampling中=False
        alphas =  self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        b=x_1.shape[0]
        a_t = torch.full((b, 1, 1, 1), alphas[0], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[0], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[0], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[0],device=device)

        # current prediction for x_0
        pred_x0 = (x_1 - sqrt_one_minus_at * e_t) / a_t.sqrt()
        #sample中quantize_x0=False,略过此步骤：
        # if quantize_denoised:
        #     pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        #repeat_noise=False
        #temperature=1
        #noise = sigma_t * noise_like(x_1.shape, device, repeat_noise) * temperature
        #sample中noise_dropout=0,略过此步骤：
        # if noise_dropout > 0.:
        #     noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt #+ noise
        
        
        
        return x_prev
    
    def hiding_cross_quality(self,A,x_1,secret,device,unconditional_guidance_scale=1):
        unconditional_guidance_scale=10
        unet_model=self.model.model.diffusion_model
        blocks=unet_model.output_blocks
        block=blocks[11]

        v=block[1].transformer_blocks[0].attn2.v#不变
        basictran_in_features=block[1].transformer_blocks[0].in_layers_features#不变
        change=torch.load("secret/inject_attn_quality.pt").to(device)
        change.requires_grad_(True)
        
        attn=A+secret        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=8)
        out=block[1].transformer_blocks[0].attn2.to_out(out)    
        x = out+ basictran_in_features
        #cross-attention
        #x = block[1].transformer_blocks[0].attn2(block[1].transformer_blocks[0].norm2(x), context=block[1].transformer_blocks[0].context) + x
        #ff
        x = block[1].transformer_blocks[0].ff(block[1].transformer_blocks[0].norm3(x)) + x
        #出basic transformer block
        sptran_in_feature=block[1].in_feature#不变
        b, c, h, w = sptran_in_feature.shape
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = block[1].proj_out(x)
        x=x + sptran_in_feature
        #出spatial transformer
        if len(block)==3:
            x=block[2](x)
        #进入GSC
        x = x.type(unet_model.dtype)
        x=unet_model.out(x)
        #输出
        #ddim.py DDIMSampler
        e_t_uncond, e_t = x.chunk(2)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) if unconditional_guidance_scale!=1 else e_t
        #alphas = model.alphas_cumprod if use_original_steps else sampler.ddim_alphas
        #use_original_steps 在ddim.py self.ddim_sampling中=False
        alphas =  self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        b=x_1.shape[0]
        a_t = torch.full((b, 1, 1, 1), alphas[0], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[0], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[0], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[0],device=device)

        # current prediction for x_0
        pred_x0 = (x_1 - sqrt_one_minus_at * e_t) / a_t.sqrt()
        #sample中quantize_x0=False,略过此步骤：
        # if quantize_denoised:
        #     pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        #repeat_noise=False
        #temperature=1
        #noise = sigma_t * noise_like(x_1.shape, device, repeat_noise) * temperature
        #sample中noise_dropout=0,略过此步骤：
        # if noise_dropout > 0.:
        #     noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt #+ noise      
        return x_prev


    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               negative_conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               injected_features=None,
               strength=1.,
               callback_ddim_timesteps=None,
               negative_prompt_alpha=1.0,
               negative_prompt_schedule='constant',
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose,strength=strength)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    negative_conditioning=negative_conditioning,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    injected_features=injected_features,
                                                    callback_ddim_timesteps=callback_ddim_timesteps,
                                                    negative_prompt_alpha=negative_prompt_alpha,
                                                    negative_prompt_schedule=negative_prompt_schedule,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, negative_conditioning=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      injected_features=None, callback_ddim_timesteps=None,
                      negative_prompt_alpha=1.0, negative_prompt_schedule='constant'):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img],'self-atten-map-11':[img],'x_1':[img],'self-atten-map-ori-11':[img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        callback_ddim_timesteps_list = np.flip(make_ddim_timesteps("uniform", callback_ddim_timesteps, self.ddpm_num_timesteps))\
            if callback_ddim_timesteps is not None else np.flip(self.ddim_timesteps)

        negative_prompt_alpha_schedule = self.make_negative_prompt_schedule(negative_prompt_schedule, negative_prompt_alpha, total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            injected_features_i = injected_features[i]\
                if (injected_features is not None and len(injected_features) > 0) else None
            negative_prompt_alpha_i = negative_prompt_alpha_schedule[i]
            if i==len(iterator)-1:
                x_1=img.clone()
                unconditional_guidance_scale=10
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      negative_conditioning=negative_conditioning,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      injected_features=injected_features_i,
                                      negative_prompt_alpha=negative_prompt_alpha_i
                                      )

            img, pred_x0 = outs
            if step in callback_ddim_timesteps_list:
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, img, step)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                
            #返回t=0时，最后一层（11）self-attention map值
            if i==len(iterator)-1:
                unet_model = self.model.model.diffusion_model
                blocks=unet_model.output_blocks
                block=blocks[11]
                atten_map=block[1].transformer_blocks[0].attn2.attn
                atten_map_ori=block[1].transformer_blocks[0].attn2.attn_ori
                intermediates['self-atten-map-11'].append(atten_map)
                intermediates['self-atten-map-ori-11'].append(atten_map_ori)
                intermediates['x_1'].append(x_1)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, negative_conditioning=None,
                      repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      injected_features=None, negative_prompt_alpha=1.0
                      ):
        b, *_, device = *x.shape, x.device

        if negative_conditioning is not None:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            uc = unconditional_conditioning
            nc = negative_conditioning

            c_in = torch.cat([nc, uc])
            e_t_negative, e_t_uncond = self.model.apply_model(x_in,
                                                     t_in,
                                                     c_in,
                                                     injected_features=injected_features
                                                     ).chunk(2)

            c_in = torch.cat([uc, c])
            e_t_uncond, e_t = self.model.apply_model(x_in,
                                                     t_in,
                                                     c_in,
                                                     injected_features=injected_features
                                                     ).chunk(2)

            e_t_tilde = negative_prompt_alpha * e_t_uncond + (1 - negative_prompt_alpha) * e_t_negative
            e_t = e_t_tilde + unconditional_guidance_scale * (e_t - e_t_tilde)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in,
                                                     injected_features=injected_features).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) if unconditional_guidance_scale!=1 else e_t

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        #print(torch.sum(sigma_t),temperature,torch.sum(noise))
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode_ddim(self, img, num_steps,conditioning, unconditional_conditioning=None ,unconditional_guidance_scale=1.):
        
        print(f"Running DDIM inversion with {num_steps} timesteps")
        T = 999
        c = T // num_steps
        iterator = tqdm(range(0,T ,c), desc='DDIM Inversion', total= num_steps)
        steps = list(range(0,T + c,c))

        for i, t in enumerate(iterator):
            img, _ = self.reverse_ddim(img, t, t_next=steps[i+1] ,c=conditioning, unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)

        return img, _

    @torch.no_grad()
    def reverse_ddim(self, x, t,t_next, c=None, quantize_denoised=False, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        if c is None:
            e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t_tensor, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_tensor] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod #.flip(0)
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod #.flip(0)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[t], device=device)
        a_next = torch.full((b, 1, 1, 1), alphas[t_next], device=device) #a_next = torch.full((b, 1, 1, 1), alphas[t + 1], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * pred_x0 + dir_xt
        return x_next, pred_x0   

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec