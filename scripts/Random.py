import math
import os
import sys
import traceback
import random
import modules.scripts as scripts
import gradio as gr
from modules.processing import Processed,process_images

# 개조용 추가 로드
import re,random
from modules.images import FilenameGenerator
from PIL import Image
from modules import processing,shared,generation_parameters_copypaste
from modules.shared import opts,state

def create_infotext(p):

    clip_skip = getattr(p,'clip_skip',opts.CLIP_stop_at_last_layers)

    generation_params = {
        #"Steps": p.steps,
        #"Sampler": processing.get_correct_sampler(p)[p.sampler_index].name,
        #"CFG scale": p.cfg_scale,
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": getattr(p,'sd_model_hash',None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        "Model": (None if not opts.add_model_name_to_info or not shared.sd_model.sd_checkpoint_info.model_name else shared.sd_model.sd_checkpoint_info.model_name.replace(',','').replace(':','')),
        "Hypernet": (None if shared.loaded_hypernetwork is None else shared.loaded_hypernetwork.name),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p,'denoising_strength',None),
        "Eta": (None if p.sampler is None or p.sampler.eta == p.sampler.default_eta else p.sampler.eta),
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta,
    }

    generation_params.update(p.extra_generation_params)

    generation_params_text = ",".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k,v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: \n" + p.negative_prompt if p.negative_prompt else ""

    return f"{p.prompt[0] if type(p.prompt) == list else p.prompt}{negative_prompt_text}\n{generation_params_text}".strip()
    
class Script(scripts.Script):

    def title(self):
        return "Random"

    def ui(self,is_img2img):
        loops = gr.Slider(minimum=1,maximum=10000,step=1,label='Loops',value=10000)
        #denoising_strength_change_factor = gr.Slider(minimum=0.9,maximum=1.1,step=0.01,label='Denoising strength change factor',value=1)

        step1 = gr.Slider(minimum=1,maximum=150,step=1,label='step1 min/max',value=10)
        step2 = gr.Slider(minimum=1,maximum=150,step=1,label='step2 min/max',value=15)
        #stepc = gr.Slider(minimum=1,maximum=100,step=1,label='step cnt',value=10)
        cfg1 = gr.Slider(minimum=1,maximum=30,step=1,label='cfg1 min/max',value=6)
        cfg2 = gr.Slider(minimum=1,maximum=30,step=1,label='cfg2 min/max',value=15)
        #cfgc = gr.Slider(minimum=1,maximum=100,step=1,label='cfg cnt',value=10)

        w1 = gr.Slider(minimum=64,maximum=2048,step=64,label='w1 min/max',value=512)
        w2 = gr.Slider(minimum=64,maximum=2048,step=64,label='w2 min/max',value=768)
        h1 = gr.Slider(minimum=64,maximum=2048,step=64,label='h1 min/max',value=512)
        h2 = gr.Slider(minimum=64,maximum=2048,step=64,label='h2 min/max',value=768)
        
        #whmax = gr.Slider(minimum=4096,maximum=4194304,step=4096,label='w*h max',value=393216)
        
        no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds',value=True)
        
        #return [loops,denoising_strength_change_factor]
        return [loops,step1,step2,cfg1,cfg2,no_fixed_seeds,w1,w2,h1,h2]#,whmax

    #def run(self,p,loops,denoising_strength_change_factor):
    def run(self,p,loops,step1,step2,cfg1,cfg2,no_fixed_seeds,w1,w2,h1,h2):#,whmax
        #print(f"p.all_seeds ; {p.all_seeds}")
        # print(f"{loops};{step1};{step2};{cfg1};{cfg2};{no_fixed_seeds};")
        # print(f"{type(loops)};{type(step1)};{type(step2)};{type(cfg1)};{type(cfg2)};{type(no_fixed_seeds)};")
        
        # 와일드카드 텍스트 저장용 폴더 생성
        #print(f"p.outpath_samples ; {p.outpath_samples}")
        os.makedirs(p.outpath_samples,exist_ok=True)
        
        namegen = FilenameGenerator(p,p.seed,p.prompt,Image.new('RGBA',(p.width,p.height)))
        #print(f"opts.save_to_dirs ; {opts.save_to_dirs}")
        #print(f"opts.directories_filename_pattern ; {opts.directories_filename_pattern}")
        file_decoration = namegen.apply( opts.directories_filename_pattern)
        #print(f"file_decoration ; {file_decoration}")
        fullfn=os.path.join(p.outpath_samples,file_decoration)
        #print(f"fullfn ; {fullfn}")
        os.makedirs(fullfn,exist_ok=True)

        if opts.save_txt is not None:
            txt_fullfn =os.path.join(fullfn,f"{file_decoration}-{self.title()}.txt") 
            #print(f"txt_fullfn ; {txt_fullfn}")
            with open(txt_fullfn,"w",encoding="utf8") as file:
                infotexts=create_infotext(p)
                #print(f"p.info ; {p.info}")
                #print(f"infotexts ; {infotexts}")
                #print(f"p.job_timestamp ; {p.job_timestamp}")
                file.write(infotexts + "\n")

        prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        negative_prompt = p.negative_prompt[0] if type(p.negative_prompt) == list else p.negative_prompt
        #processing.fix_seed(p)
        if not no_fixed_seeds:
            p.seed=-1;
            processing.fix_seed(p)
            
        h1=h1/64
        h2=h2/64
        w1=w1/64
        w2=w2/64
        print(f" width:{w1},{w2} ; height:{h1},{h2}")
            
        print(f"bdfore loops:{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale}")
        for i in range(loops):
            if step1 > step2 :
                p.steps=random.randint(step2,step1)
            else :
                p.steps=random.randint(step1,step2)
                
            if cfg1 > cfg2 :
                p.cfg_scale=random.randint(cfg2,cfg1)
            else :
                p.cfg_scale=random.randint(cfg1,cfg2)
                
            if w1 > w2 :
                p.width=random.randint(w2,w1)
            else :
                p.width=random.randint(w1,w2)
            
            p.width=p.width*64
            
            if h1 > h2 :
                p.height=random.randint(h2,h1)
            else :
                p.height=random.randint(h1,h2)
            
            p.height=p.height*64
            
            print(f"loops: {i+1}/{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale} ; width:{p.width} ; height:{p.height}")
            
            p.prompt = prompt
            p.negative_prompt = negative_prompt
            p.prompt_for_display = p.prompt[0] if type(p.prompt) == list else p.prompt
            
            proc = process_images(p)
            image = proc.images
            
            
            if state.interrupted:
                break
            
        return Processed(p,image,p.seed,proc.info)
