import torch
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
    
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

import numpy as np
import inspect
import gc
from accelerate import Accelerator

# from transformer_flux_masked import MaskedFluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
from tqdm import tqdm

# from diffusers import FluxTransformer2DModel
# from transformer_flux_mspace import MSpaceFluxTransformer2DModel
# from diffusers import FluxFillPipeline
# from pipeline_flux_mspace import MSpacePlusFluxPipeline
import os

# from diffusers import FluxPriorReduxPipeline
# from diffusers.utils import load_image

import torch
import argparse
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# image = load_image("F:/ImageSet/ObjectRemoval/test.jpg")
# mask = load_image("F:/ImageSet/ObjectRemoval/test_mask.png")

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

@torch.no_grad()
def main():

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Parse resolution
    # if "1024 × 1024" in resolution_str:
    #     return 1024, 1024
    # elif "768 × 1360" in resolution_str:
    #     return 768, 1360
    # elif "1360 × 768" in resolution_str:
    #     return 1360, 768
    # elif "880 × 1168" in resolution_str:
    #     return 880, 1168
    # elif "1168 × 880" in resolution_str:
    #     return 1168, 880
    # elif "1248 × 832" in resolution_str:
    #     return 1248, 832
    # elif "832 × 1248" in resolution_str:
    #     return 832, 1248
    # else:
    #     return 1024, 1024  # Default fallback
    height, width = 832, 1248
    # prompt = "Remove the selected objects from the image."
    # hidream_dir = "F:/HiDream-I1/hidream_models/full"
    hidream_dir = "F:/HiDream-I1/hidream_models/fast"
    seed = 42
    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    # num_inference_steps = 50
    # guidance_scale = 5.0
    num_inference_steps = 16
    guidance_scale = 0
    save_image_name = "test_fast"
    
    
    llama31_dir = "F:/HiDream-I1/hidream_models/llama31"

    embedding_path = "0_embedding.pt"
    # prompt = "A stunning 4K watercolor painting of a European beauty, Beke Jacoba, a sacred yet unholy fallen angel with huge black feather wings, freckled skin, and beautiful glowing red eyes, portrayed as a piece of art by Olivier Ledroit & Calm. She has long black ponytail hair, a black & gold shoulder armor with a glowing red jewel on her pauldrons, a black choker with an upside-down cross, and a multiple-chain belt around her hips while holding a black rosary around her arm. With an hourglass figure, huge veiny breasts, long legs, large hips, trimmed pubic hair, and scars on her body, she strikes a sensual erotic pose—leg raised, swollen labia visible in a 3/4 side view from below, standing with one foot just over the viewer in a dark Gothic cathedral illuminated by blue and purple stained glass. The image, in an absurdly high resolution, captures sharp focus on her body and crotch, emphasizing her dominance in a Femdom POV as she gazes at the viewer with perfect, detailed features—full lips, long eyelashes, and eyeliner enhancing her gothic, black-and-red aesthetic."
    prompt = "Create a surreal, ethereal scene set in a hidden celestial library nestled within the heart of an ancient, overgrown forest. The library is an open-air structure with towering, crumbling marble columns draped in ivy and glowing bioluminescent vines. Shelves of floating books spiral upward into a twilight sky streaked with auroras. At the center, a massive, gilded astrolabe rotates slowly, casting prismatic light onto a pool of liquid stardust below. Surrounding the pool, mythical creatures gather: a phoenix with feathers made of molten amber perches on a fractured clocktower, a fox with a coat resembling shifting constellations sniffs at an open book whose pages flutter into origami birds, and a translucent, jellyfish-like entity floats nearby, its tentacles made of inky calligraphy. Through the library’s arched windows, glimpses of a distant galaxy swirl, while tiny robot sprites with butterfly wings repair cracks in the walls using starlight. The scene is illuminated by a mix of moonlight filtering through stained-glass domes and the soft glow of hovering lanterns shaped like dandelion seeds."
    neg_prompt = "bad anatomy, watermark, logo, anime, comic, drawing, bench, chair, stool, table,"

    # scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)
    scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)
        
    tokenizer = CLIPTokenizer.from_pretrained(hidream_dir, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(hidream_dir, subfolder="tokenizer_2")
    tokenizer_3 = T5Tokenizer.from_pretrained(hidream_dir, subfolder="tokenizer_3")
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama31_dir,use_fast=False)
    
    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    if os.path.exists(embedding_path):
        base_embedding = torch.load(embedding_path)
        
        prompt_embeds = base_embedding["prompt_embeds"]
        pooled_prompt_embeds = base_embedding["pooled_prompt_embeds"]
        negative_prompt_embeds = base_embedding["negative_prompt_embeds"]
        negative_pooled_prompt_embeds = base_embedding["negative_pooled_prompt_embeds"]
    else:
        # text_encoder = CLIPTextModel.from_pretrained(
        #     hidream_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
        # ).to("cuda")
        # text_encoder_2 = CLIPTextModel.from_pretrained(
        #     hidream_dir, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        # ).to("cuda")
        # text_encoder_3 = T5EncoderModel.from_pretrained(
        #     hidream_dir, subfolder="text_encoder_3", torch_dtype=torch.bfloat16
        # ).to("cuda")
        # 第一阶段：加载text_encoder 和 tokenizer处理prompt
        pipeline = HiDreamImagePipeline.from_pretrained(
            hidream_dir,
            scheduler=scheduler,
            # text_encoder=text_encoder,
            # text_encoder_2=text_encoder_2,
            # text_encoder_3=text_encoder_3,
            text_encoder_4=None,
            # tokenizer=tokenizer,
            # tokenizer_2=tokenizer_2,
            # tokenizer_3=tokenizer_3,
            tokenizer_4=tokenizer_4,
            # transformer=None,
            vae=None,
        ).to("cuda")
        
        with torch.no_grad():
            print("Encoding prompts.")
            # prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
            #     prompt=prompt
            # )
            max_sequence_length = 128
            num_images_per_prompt = 1
            def get_first_part_embedding(prompt):
                pooled_prompt_embeds_1 = pipeline._get_clip_prompt_embeds(
                    pipeline.tokenizer,
                    pipeline.text_encoder,
                    prompt = prompt,
                    num_images_per_prompt = num_images_per_prompt,
                    max_sequence_length = max_sequence_length,
                    device = device,
                    dtype = dtype,
                )

                pooled_prompt_embeds_2 = pipeline._get_clip_prompt_embeds(
                    pipeline.tokenizer_2,
                    pipeline.text_encoder_2,
                    prompt = prompt,
                    num_images_per_prompt = num_images_per_prompt,
                    max_sequence_length = max_sequence_length,
                    device = device,
                    dtype = dtype,
                )

                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

                t5_prompt_embeds = pipeline._get_t5_prompt_embeds(
                    prompt = prompt,
                    num_images_per_prompt = num_images_per_prompt,
                    max_sequence_length = max_sequence_length,
                    device = device,
                    dtype = dtype
                )
                return t5_prompt_embeds,pooled_prompt_embeds
            
            pos_t5_prompt_embeds,pos_pooled_prompt_embeds = get_first_part_embedding(prompt)
            neg_t5_prompt_embeds,neg_pooled_prompt_embeds = get_first_part_embedding(neg_prompt)
            
            pos_t5_prompt_embeds = pos_t5_prompt_embeds.to("cpu")
            pos_pooled_prompt_embeds = pos_pooled_prompt_embeds.to("cpu")
            neg_t5_prompt_embeds = neg_t5_prompt_embeds.to("cpu")
            neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to("cpu")
            
            print("Before:", torch.cuda.memory_allocated())  # Memory in use
            print("Before (cached):", torch.cuda.memory_reserved())  # Cached memory
            del pipeline
            # del text_encoder
            # del text_encoder_2
            # del text_encoder_3
            # del tokenizer
            # del tokenizer_2
            # del tokenizer_3
            flush()
            print("After:", torch.cuda.memory_allocated())  # Memory in use
            print("After (cached):", torch.cuda.memory_reserved())  # Cached memory
            
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                llama31_dir,
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=torch.bfloat16).to("cuda")
            pipeline = HiDreamImagePipeline.from_pretrained(
                hidream_dir,
                scheduler=scheduler,
                text_encoder=None,
                text_encoder_2=None,
                text_encoder_3=None,
                text_encoder_4=text_encoder_4,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                tokenizer_3=tokenizer_3,
                tokenizer_4=tokenizer_4,
                transformer=None,
                vae=None,
            ).to("cuda")
            pos_t5_prompt_embeds = pos_t5_prompt_embeds.to(device)
            pos_pooled_prompt_embeds = pos_pooled_prompt_embeds.to(device)
            neg_t5_prompt_embeds = neg_t5_prompt_embeds.to(device)
            neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to(device)
            def get_second_part_embedding(prompt):
                llama3_prompt_embeds = pipeline._get_llama3_prompt_embeds(
                    prompt = prompt,
                    num_images_per_prompt = num_images_per_prompt,
                    max_sequence_length = max_sequence_length,
                    device = device,
                    dtype = dtype
                )
                return llama3_prompt_embeds
            pos_llama3_prompt_embeds = get_second_part_embedding(prompt)
            neg_llama3_prompt_embeds = get_second_part_embedding(neg_prompt)
            prompt_embeds = [pos_t5_prompt_embeds, pos_llama3_prompt_embeds]
            negative_prompt_embeds = [neg_t5_prompt_embeds, neg_llama3_prompt_embeds]
            pooled_prompt_embeds = pos_pooled_prompt_embeds
            negative_pooled_prompt_embeds = neg_pooled_prompt_embeds
            
            base_embedding = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pos_pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds
            }
            # save to base_embedding
            torch.save(base_embedding, embedding_path)
            
        del text_encoder_4
        # del tokenizer_4
        del pipeline
        flush()

    
    generator = torch.Generator("cuda").manual_seed(seed)

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        hidream_dir, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16).to("cuda")

    quantize(transformer, weights=qfloat8) # 对模型进行量化
    freeze(transformer)
    
    transformer.enable_block_swap(25, device)
    

    # 第一阶段：加载text_encoder 和 tokenizer处理prompt
    pipeline = HiDreamImagePipeline.from_pretrained(
        hidream_dir,
        scheduler=scheduler,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        text_encoder_4=None,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        tokenizer_4=tokenizer_4,
        transformer=None
    ).to("cuda",dtype=torch.bfloat16)
    
    pipeline.transformer = transformer
    # pipeline.enable_model_cpu_offload()

    # guidance_scale = 0
    
    image = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator,
        device=device
    ).images[0]
    image.save(f"{save_image_name}.png")


if __name__ == "__main__":
    main()