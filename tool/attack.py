import  os
import torch
import numpy as np
from PIL import Image
import cv2

def gass(output,sigma):
    output = np.asarray(output)
    mean = 0
    #设置高斯分布的标准差
    #sigma = 25
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

def load_img_attack(path,sample_path=None,m=None,attack_dict=None,add_noise=False):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    #add_noise=True
    #image = transforms.CenterCrop(min(x,y))(image)
    image_all=Image2tensor(image)
    if add_noise:
        for name,value in attack_dict.items():
            if name=="gass":
                image_=gass(image,value)
            elif name=="salt":
                image_=salt(image,value)
            elif name=="mean_filter":
                image_=mean_filter(image,(value,value))
            elif name=="median_filter":
                image_=median_filter(image,value)
            elif name=="gass_filter":
                image_=gass_filter(image,(value,value))
            elif name=="JPEG":
                image = Image.open(path).convert("RGB")
                image.save(os.path.join(sample_path,f'{m}_JPEG_{value}.jpeg'), quality=value)
                image_=Image.open(os.path.join(sample_path,f"{m}_JPEG_{value}.jpeg")).convert("RGB")

            save_name=m+'_'+name+"_"+str(value)+".png"
            image_.save(os.path.join(sample_path, save_name))
            image_=Image2tensor(image_)
            image_all=torch.cat((image_all,image_),0)

        return image_all
    
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
