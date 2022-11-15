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
from modules.sd_samplers import samplers

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

def wh_chg_n(p):
    return
def wh_chg_w(p):
    if p.height>p.width:
        (p.width,p.height)=(p.height,p.width)
def wh_chg_h(p):
    if p.width>p.height:
        (p.width,p.height)=(p.height,p.width)
    
class Script(scripts.Script):
    fix_whs=['none','width long','height long']
    
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

        w1 = gr.Slider(minimum=64,maximum=2048,step=64,label='width 1 min/max', elem_id="w1",value=512)
        w2 = gr.Slider(minimum=64,maximum=2048,step=64,label='width 2 min/max', elem_id="w2",value=768)
        h1 = gr.Slider(minimum=64,maximum=2048,step=64,label='height 1 min/max', elem_id="h1",value=512)
        h2 = gr.Slider(minimum=64,maximum=2048,step=64,label='height 2 min/max', elem_id="h2",value=768)
        
        #whmax = gr.Slider(minimum=4096,maximum=4194304,step=4096,label='w*h max',value=393216)
        
        fix_wh = gr.Radio(label='fix width height direction', elem_id="fix_wh", choices=[x for x in self.fix_whs], value=self.fix_whs[0], type="index")
        
        rnd_sampler = gr.CheckboxGroup(label='Sampling Random', elem_id="rnd_sampler", choices=[x.name for x in samplers], type="index")
        
        # samplers
        
        fixed_seeds = gr.Checkbox(label='Keep -1 for seeds',value=True)
        
        #return [loops,denoising_strength_change_factor]
        return [loops,step1,step2,cfg1,cfg2,fixed_seeds,w1,w2,h1,h2,fix_wh,rnd_sampler]#,whmax

    #def run(self,p,loops,denoising_strength_change_factor):
    def run(self,p,loops,step1,step2,cfg1,cfg2,fixed_seeds,w1,w2,h1,h2,fix_wh,rnd_sampler):#,whmax
        #print(f"p.all_seeds ; {p.all_seeds}")
        print(f"{loops};{step1};{step2};{cfg1};{cfg2};{fixed_seeds};{fix_wh};{p.sampler_index};{rnd_sampler};")
        print(f"{type(loops)};{type(step1)};{type(step2)};{type(cfg1)};{type(cfg2)};{type(fixed_seeds)};{type(fix_wh)};{type(p.sampler_index)};{type(rnd_sampler)};")
        
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

        h1=h1/64
        h2=h2/64
        w1=w1/64
        w2=w2/64
        
        (stepmin,stepmax)= (min(step2,step1),max(step2,step1))
        (cfgmin,cfgmax)= (min(cfg1,cfg2),max(cfg1,cfg2))
        (wmin,wmax)= (min(w1,w2),max(w1,w2))
        (hmin,hmax)= (min(h1,h2),max(h1,h2))
        
        #print(f" width:{w1},{w2} ; height:{h1},{h2}")
        
        wh_chg = {0 : wh_chg_n, 1: wh_chg_w, 2 : wh_chg_h}.get(fix_wh, wh_chg_n)
        
        rnd_sampler_chg=False
        if len(rnd_sampler) == 1:
            p.sampler_index=rnd_sampler[0]
        elif len(rnd_sampler) > 1:
            rnd_sampler_chg=True
        
        print(f"bdfore loops:{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale}")
        for i in range(loops):
            
            p.steps=random.randint(stepmin,stepmax)
            p.cfg_scale=random.randint(cfgmin,cfgmax)
            p.width=random.randint(wmin,wmax)
            p.height=random.randint(hmin,hmax)
            if rnd_sampler_chg:
                p.sampler_index=random.choice(rnd_sampler)
            
            wh_chg(p)
            
            p.width=p.width*64
            p.height=p.height*64
            
            print(f"loops: {i+1}/{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale} ; width:{p.width} ; height:{p.height}")
            
            p.prompt = prompt
            p.negative_prompt = negative_prompt
            p.prompt_for_display = p.prompt[0] if type(p.prompt) == list else p.prompt

            if fixed_seeds:
                p.seed=-1;
                processing.fix_seed(p)
            
            try:
                proc = process_images(p)
                image = proc.images
            except Exception as e :
                print(f"process_images err ;\r\n",e)
            
            if state.interrupted:
                break
            
        return Processed(p,image,p.seed,proc.info)
