#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter import IPAdapter


# In[2]:


base_model_path = "/mnt/Datasets/Opensource_pretrained_models/Stable_Diffusion/stable-diffusion-v1-5"
vae_model_path = "/mnt/Datasets/Opensource_pretrained_models/Stable_Diffusion/stable-diffusion-v1-5/vae"
image_encoder_path = "/home/zhuyixing/model/IP-Adapter/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "/home/zhuyixing/model/IP-Adapter/ip-adapter_sd15.bin"
device = "cuda"


# In[3]:


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)


# ## Image Variations

# In[4]:


# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)


# In[5]:


# read image prompt
image = Image.open("assets/images/woman.png")
image.resize((256, 256))
image.save('/home/zhuyixing/show/test-1.jpg')

# In[6]:


# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)


# In[7]:


# generate image variations
images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42)
grid = image_grid(images, 1, 4)

grid.save('/home/zhuyixing/show/test0.jpg')


# ## Image-to-Image

# In[8]:


# load SD Img2Img pipe
del pipe, ip_model
torch.cuda.empty_cache()
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)


# In[9]:


# read image prompt
image = Image.open("assets/images/river.png")
g_image = Image.open("assets/images/vermeer.jpg")
image_show = image_grid([image.resize((256, 256)), g_image.resize((256, 256))], 1, 2)

image_show.save('/home/zhuyixing/show/test1.jpg')

# In[10]:


# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)


# In[11]:


# generate
images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=0.6)
grid = image_grid(images, 1, 4)
grid.save('/home/zhuyixing/show/test2.jpg')


# ## Inpainting

# In[12]:


# load SD Inpainting pipe
del pipe, ip_model
torch.cuda.empty_cache()
pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)


# In[13]:


# read image prompt
image = Image.open("assets/images/girl.png")
image.resize((256, 256))


# In[14]:


masked_image = Image.open("assets/inpainting/image.png").resize((512, 768))
mask = Image.open("assets/inpainting/mask.png").resize((512, 768))
image_show = image_grid([masked_image.resize((256, 384)), mask.resize((256, 384))], 1, 2)
image_show.save('/home/zhuyixing/show/test3.jpg')
breakpoint()


# In[15]:


# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)


# In[16]:


# generate
images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50,
                           seed=42, image=masked_image, mask_image=mask, strength=0.7, )
grid = image_grid(images, 1, 4)
breakpoint()
grid.save('/home/zhuyixing/show/test4.jpg')
pass