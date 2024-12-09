import torch
import numpy as np


def hide(method,secret,secret_attn,n=1,q=2000,lamda_a=2,lamda_l=0.5,samples_ddim=None,distribution=None,device='cpu'):
    inject_direct=lamda_l*(secret-0.5)
    inject_std=((secret-0.5)*q).to(device)

    inject_attn=((secret_attn-0.5)*lamda_a).to(device)
    torch.save(inject_attn,"secret/inject_attn_start.pt")

    if n==1:
        zeros_8=torch.zeros(8,4096,1).to(device)
        change_attn=torch.cat((zeros_8,inject_attn,inject_attn,inject_attn,inject_attn,inject_attn,inject_attn,inject_attn,inject_attn),0)
        change_attn=inject_attn

        change_direct=inject_direct
        change_std=inject_std
    elif n==2:
        zeros4=torch.zeros(8,4096,1).to(device)
        change_attn=torch.cat((zeros4,inject_attn,inject_attn,inject_attn,inject_attn),0)
        change_direct=torch.cat((inject_direct,inject_direct),0)
        change_std=torch.cat((inject_std,inject_std),0)
    elif n==3:
        zeros4=torch.zeros(8,4096,1).to(device)
        change_attn=torch.cat((zeros4,inject_attn,inject_attn,inject_attn[0:2,:,:]),0)
        sep0,_,_=inject_direct.chunk(3)
        change_direct=torch.cat((inject_direct,sep0),0)
        sep0,_,_=inject_std.chunk(3)
        change_std=torch.cat((inject_std,sep0),0)
    else:
        zeros4=torch.zeros(8,4096,1).to(device)
        change_attn=torch.cat((zeros4,inject_attn,inject_attn),0)
        change_direct=inject_direct
        change_std=inject_std

    steg_64=torch.zeros(1,4,64,64).to(device)
    for m in method:
        if m=="D-RM":    
            steg=samples_ddim+change_direct
            steg_64=torch.cat((steg_64,steg),0)
        elif m=="STD-RM":
            mean=distribution.mean
            std=distribution.std                   
            steg = 0.18215*(mean + std *change_std)
            steg_64=torch.cat((steg_64,steg),0)
        elif m=="SM":
            if len(method)==1:
                return None,change_attn

        
    return steg_64[1:,:,:,:],change_attn
