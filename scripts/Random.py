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
from modules.sd_samplers import samplers,samplers_for_img2img
import logging
from my import *

logger = logging.getLogger(__name__)
logger.handlers.clear()
#logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.WARNING)

# 일반 핸들러. 할 필요 업음. 이미 메인에서 출력해줌
#streamFormatter = logging.Formatter("Random Stream %(asctime)s %(levelname)s %(message)s")
#streamHandler = logging.StreamHandler()
#streamHandler.setLevel(logging.INFO)
##streamHandler.setLevel(logging.WARNING)
#streamHandler.setFormatter(streamFormatter)
#logger.addHandler(streamHandler)

# 파일 핸들러
fileFormatter = logging.Formatter("Random File %(asctime)s %(levelname)s %(message)s")
fileHandler = logging.FileHandler("Random.py.log")
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(fileFormatter)
logger.addHandler(fileHandler)

if logger.getEffectiveLevel() == logging.DEBUG :
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

logger.debug('==== DEBUG ====')
logger.info(' Load ')

class Script(scripts.Script):
    fix_whs=['none','width long','height long','random']
    #fix_whs_d== dict(enumerate(fix_whs))
    fix_whs_d={0 : wh_chg_n, 1: wh_chg_w, 2 : wh_chg_h, 3 : wh_chg_r}
    
    def title(self):
        return "Random"

    def ui(self,is_img2img):
        logger.debug(f"is_img2img : {(is_img2img)};")
        with gr.Blocks():
            gr.Markdown(" ", elem_id="rnd-loop")#    padding-bottom: 100px;
            if is_img2img:
                loops = gr.Slider(minimum=1,maximum=10000,step=1,label='Loops',value=1, elem_id="rnd-loops")
            else:
                loops = gr.Slider(minimum=1,maximum=10000,step=1,label='Loops',value=10000, elem_id="rnd-loops")
        #denoising_strength_change_factor = gr.Slider(minimum=0.9,maximum=1.1,step=0.01,label='Denoising strength change factor',value=1)
        gr.Markdown("how it works step,cfg,denoising,width,height  \r\nif step1 > step2 then random(min=step2,max=step1)  \r\nif step2 > step1 then random(min=step1,max=step2)", elem_id="rnd-step")
        with gr.Blocks():
            gr.Markdown(" ", elem_id="rnd-step")
            step1 = gr.Slider(minimum=1,maximum=150,step=1,label='step1 min/max',value=10, elem_id="rnd-step1")
            step2 = gr.Slider(minimum=1,maximum=150,step=1,label='step2 min/max',value=15, elem_id="rnd-step2")
        #stepc = gr.Slider(minimum=1,maximum=100,step=1,label='step cnt',value=10)
        with gr.Blocks():
            gr.Markdown(" ", elem_id="rnd-cfg")
            cfg1 = gr.Slider(minimum=1,maximum=30,step=0.5,label='cfg1 min/max',value=6 , elem_id="rnd-cfg1")
            cfg2 = gr.Slider(minimum=1,maximum=30,step=0.5,label='cfg2 min/max',value=15, elem_id="rnd-cfg2")
        #cfgc = gr.Slider(minimum=1,maximum=100,step=1,label='cfg cnt',value=10)
        #if is_img2img:
        with gr.Blocks():
            gr.Markdown("only i2i option", elem_id="rnd-denoising",css=".gradio-container {min-height: 6rem;}")
            denoising1 = gr.Slider(minimum=0,maximum=1,step=0.01,label='denoising1 min/max',value=0.5, elem_id="rnd-denoising1")
            denoising2 = gr.Slider(minimum=0,maximum=1,step=0.01,label='denoising2 min/max',value=1.0, elem_id="rnd-denoising2")
            #else :
            #    denoising1=None
            #    denoising2=None

        with gr.Blocks():
            gr.Markdown("size", elem_id="rnd-size")
            if is_img2img:
                no_resize = gr.Checkbox(label='no resize',value=True , elem_id="rnd-no-resize")
            else:
                no_resize = gr.Checkbox(label='no resize',value=False, elem_id="rnd-no-resize")
            w1 = gr.Slider(minimum=64,maximum=2048,step=64,label='width 1 min/max' ,value=512 , elem_id="rnd-w1")
            w2 = gr.Slider(minimum=64,maximum=2048,step=64,label='width 2 min/max' ,value=768 , elem_id="rnd-w2")
            h1 = gr.Slider(minimum=64,maximum=2048,step=64,label='height 1 min/max',value=512 , elem_id="rnd-h1")
            h2 = gr.Slider(minimum=64,maximum=2048,step=64,label='height 2 min/max',value=768 , elem_id="rnd-h2")
        
        #whmax = gr.Slider(minimum=4096,maximum=4194304,step=4096,label='w*h max',value=393216)
        
            fix_wh = gr.Radio(label='fix width height direction', choices=[x for x in self.fix_whs], value=self.fix_whs[0], type="index", elem_id="rnd-fix_wh")


        with gr.Blocks():
            gr.Markdown(" ", elem_id="sampler")
            if is_img2img:
                rnd_sampler = gr.CheckboxGroup(label='Sampling Random', elem_id="rnd_sampler", choices=[x.name for x in samplers_for_img2img],value=[x.name for x in samplers_for_img2img])#, type="index"
            else :
                rnd_sampler = gr.CheckboxGroup(label='Sampling Random', elem_id="rnd_sampler", choices=[x.name for x in samplers],value=[x.name for x in samplers])#, type="index"
        
        # samplers
        with gr.Blocks():
            gr.Markdown(" ", elem_id="Keep")
            fixed_seeds = gr.Checkbox(label='Keep -1 for seeds',value=True)
        
        #return [loops,denoising_strength_change_factor]
        return [loops,step1,step2,cfg1,cfg2,fixed_seeds,w1,w2,h1,h2,fix_wh,rnd_sampler,no_resize,denoising2,denoising1]#,whmax

    #def run(self,p,loops,denoising_strength_change_factor):
    def run(self,p,loops,step1,step2,cfg1,cfg2,fixed_seeds,w1,w2,h1,h2,fix_wh,rnd_sampler,no_resize,denoising2,denoising1):#,whmax
        #print(f"p.all_seeds ; {p.all_seeds}")
        logger.debug(f"{loops};{step1};{step2};{cfg1};{cfg2};{fixed_seeds};{fix_wh};{p.sampler_name};{p.denoising_strength};{rnd_sampler};")
        logger.debug(f"{type(loops)};{type(step1)};{type(step2)};{type(cfg1)};{type(cfg2)};{type(fixed_seeds)};{type(fix_wh)};{type(p.sampler_name)};{type(p.denoising_strength)};{type(rnd_sampler)};")
        
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
        if not no_resize:
            h1=h1/64
            h2=h2/64
            w1=w1/64
            w2=w2/64
            (wmin,wmax)= (min(w1,w2),max(w1,w2))
            (hmin,hmax)= (min(h1,h2),max(h1,h2))
            wh_chg = self.fix_whs_d.get(fix_wh, wh_chg_n)
        
        (stepmin,stepmax)= (min(step2,step1),max(step2,step1))
        (cfgmin,cfgmax)= (min(cfg1,cfg2),max(cfg1,cfg2))
        if denoising1 is not None and denoising2 is not None :
            is_img2img=True
        else :
            is_img2img=False
        
        
        if is_img2img:
            (dmin,dmax)= (min(denoising1,denoising2),max(denoising1,denoising2))
        #print(f" width:{w1},{w2} ; height:{h1},{h2}")
        
        
        rnd_sampler_chg=False
        if len(rnd_sampler) == 1:
            p.sampler_name=rnd_sampler[0]
        elif len(rnd_sampler) > 1:
            rnd_sampler_chg=True
        
        logger.debug(f"bdfore loops:{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale}")
        for i in range(loops):
            
            p.steps=random.randint(stepmin,stepmax)
            #p.cfg_scale=random.randint(cfgmin,cfgmax)
            p.cfg_scale=random.randint(0, int((cfgmax - cfgmin) / 0.5)) * 0.5 + cfgmin
            if is_img2img:
                p.denoising_strength=random.uniform(dmin,dmax)
            if not no_resize:
                p.width=random.randint(wmin,wmax)*64
                p.height=random.randint(hmin,hmax)*64
                wh_chg(p)
            
            if rnd_sampler_chg:
                p.sampler_name=random.choice(rnd_sampler)
            
            if is_img2img:
                logger.info(f"loops: {i+1}/{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale} ;  denoising_strength:{p.denoising_strength} ; width:{p.width} ; height:{p.height}")
            else :
                logger.info(f"loops: {i+1}/{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale} ; width:{p.width} ; height:{p.height}")
            
            p.prompt = prompt
            p.negative_prompt = negative_prompt
            p.prompt_for_display = p.prompt[0] if type(p.prompt) == list else p.prompt

            logger.debug(f"--prompt--")
            logger.debug(f"{p.prompt}")
            logger.debug(f"--negative_prompt--")
            logger.debug(f"{p.negative_prompt}")
            logger.debug(f"-------------------")
            
            if fixed_seeds:
                p.seed=-1;
                processing.fix_seed(p)
            
            try:
                processed = process_images(p)
            except Exception as e :
                logger.error(f"process_images err ;",e)
                
            logger.debug(f"--prompt--")
            logger.debug(f"{p.prompt}")
            logger.debug(f"--negative_prompt--")
            logger.debug(f"{p.negative_prompt}")
            logger.debug(f"-------------------")
            
            if state.interrupted:
                break
            
        return processed

