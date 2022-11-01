import math
import os
import sys
import traceback
import random
import modules.scripts as scripts
import gradio as gr
from modules.processing import Processed, process_images

class Script(scripts.Script):
    def title(self):
        return "Random"

    def ui(self, is_img2img):
        loops = gr.Slider(minimum=1, maximum=1000, step=1, label='Loops', value=100)
        #denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor', value=1)

        step1 = gr.Slider(minimum=1, maximum=150, step=1, label='step1 min/max', value=10)
        step2 = gr.Slider(minimum=1, maximum=150, step=1, label='step2 min/max', value=30)
        #stepc = gr.Slider(minimum=1, maximum=100, step=1, label='step cnt', value=10)
        cfg1 = gr.Slider(minimum=1, maximum=30, step=1, label='cfg1 min/max', value=6)
        cfg2 = gr.Slider(minimum=1, maximum=30, step=1, label='cfg2 min/max', value=15)
        #cfgc = gr.Slider(minimum=1, maximum=100, step=1, label='cfg cnt', value=10)
        
        no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=True)
        
        #return [loops, denoising_strength_change_factor]
        return [loops,step1, step2,  cfg1, cfg2,  no_fixed_seeds]

    #def run(self, p, loops, denoising_strength_change_factor):
    def run(self, p, loops, step1, step2, cfg1, cfg2, no_fixed_seeds):
        print(f"{loops};{step1};{step2};{cfg1};{cfg2};{no_fixed_seeds};")
        print(f"{type(loops)};{type(step1)};{type(step2)};{type(cfg1)};{type(cfg2)};{type(no_fixed_seeds)};")
        
        #processing.fix_seed(p)
        if not no_fixed_seeds:
            processing.fix_seed(p)
            

        print(f"bdfore loops:{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale}\r\n")
        for i in range(loops):
            if step1 > step2 :
                p.steps=random.randint(step2,step1)
            else :
                p.steps=random.randint(step1,step2)
                
            if cfg1 > cfg2 :
                p.cfg_scale=random.randint(cfg2,cfg1)
            else :
                p.cfg_scale=random.randint(cfg1,cfg2)
            
            print(f"loops: {i+1}/{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale}\r\n")
            proc = process_images(p)
            image = proc.images
            
        return Processed(p, image, p.seed, proc.info)
