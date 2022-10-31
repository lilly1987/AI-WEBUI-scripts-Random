import os
from pathlib import Path
import logging
import math
import re, random
import pathlib
from typing import Set

import gradio as gr
import modules.scripts as scripts

from modules.processing import process_images, fix_seed, Processed
from modules.shared import opts
from modules import processing,shared,generation_parameters_copypaste
from modules.images import FilenameGenerator
from PIL import Image

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

WILDCARD_DIR = getattr(opts, "wildcard_dir", "scripts/wildcards")
MAX_RECURSIONS = 20
VERSION = "0.6.2"
WILDCARD_SUFFIX = "txt"

re_wildcard = re.compile(r"__(.*?)__")
re_combinations = re.compile(r"\{([^{}]*)}")

DEFAULT_NUM_COMBINATIONS = 1

class WildcardFile:
    def __init__(self, path: Path, encoding="utf8"):
        self._path = path
        self._encoding = encoding

    def get_wildcards(self) -> Set[str]:
        is_empty_line = lambda line: line is None or line.strip() == "" or line.strip().startswith("#")

        with self._path.open(encoding=self._encoding, errors="ignore") as f:
            lines = [line.strip() for line in f if not is_empty_line(line)]
            return set(lines)


class WildcardManager:
    def __init__(self, path:str=WILDCARD_DIR):
        self._path = Path(path)

    def _directory_exists(self) -> bool:
        return self._path.exists() and self._path.is_dir()

    def ensure_directory(self) -> bool:
        try:
            self._path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.exception(f"Failed to create directory {self._path}")

    def get_files(self, relative:bool=False) -> list[Path]:
        if not self._directory_exists():
            return []


        files = self._path.rglob(f"*.{WILDCARD_SUFFIX}")
        if relative:
            files = [f.relative_to(self._path) for f in files]

        return files

    def match_files(self, wildcard:str) -> list[WildcardFile]:
        return [
            WildcardFile(path) for path in self._path.rglob(f"{wildcard}.{WILDCARD_SUFFIX}")
        ]

    def get_wildcards(self) -> list[str]:
        files = self.get_files(relative=True)
        wildcards = [f"__{path.with_suffix('')}__" for path in files]
        return wildcards

wildcard_manager = WildcardManager()

def replace_combinations(match):
    if match is None or len(match.groups()) == 0:
        logger.warning("Unexpected missing combination")
        return ""

    variants = [s.strip() for s in match.groups()[0].split("|")]
    if len(variants) > 0:
        first = variants[0].split("$$")
        quantity = DEFAULT_NUM_COMBINATIONS
        if len(first) == 2: # there is a $$
            prefix_num, first_variant = first
            variants[0] = first_variant
            
            try:
                prefix_ints = [int(i) for i in prefix_num.split("-")]
                if len(prefix_ints) == 1:
                    quantity = prefix_ints[0]
                elif len(prefix_ints) == 2:
                    prefix_low = min(prefix_ints)
                    prefix_high = max(prefix_ints)
                    quantity = random.randint(prefix_low, prefix_high)
                else:
                    raise 
            except Exception:
                logger.warning(f"Unexpected combination formatting, expected $$ prefix to be a number or interval. Defaulting to {DEFAULT_NUM_COMBINATIONS}")
        
        try:
            picked = random.sample(variants, quantity)
            return ", ".join(picked)
        except ValueError as e:
            logger.exception(e)
            return ""

    return ""

def replace_wildcard(match):
    is_empty_line = lambda line: line is None or line.strip() == "" or line.strip().startswith("#")
    if match is None or len(match.groups()) == 0:
        logger.warning("Expected match to contain a filename")
        return ""

    wildcard = match.groups()[0]
    wildcard_files = wildcard_manager.match_files(wildcard)

    if len(wildcard_files) == 0:
        logging.warning(f"Could not find any wildcard files matching {wildcard}")
        return ""

    wildcards = set().union(*[f.get_wildcards() for f in wildcard_files])

    if len(wildcards) > 0:
        return random.choice(list(wildcards))
    else:
        logging.warning(f"Could not find any wildcards in {wildcard}")
        return ""
    
def pick_wildcards(template):
    return re_wildcard.sub(replace_wildcard, template)


def pick_variant(template):
    if template is None:
        return None

    return re_combinations.sub(replace_combinations, template)

def generate_prompt(template):
    old_prompt = template
    counter = 0
    while True:
        counter += 1
        if counter > MAX_RECURSIONS:
            raise Exception("Too many recursions, something went wrong with generating the prompt")

        prompt = pick_variant(old_prompt)
        prompt = pick_wildcards(prompt)

        if prompt == old_prompt:
            logger.info(f"Prompt: {prompt}")
            return prompt
        old_prompt = prompt

def create_infotext(p):

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)

    generation_params = {
        #"Steps": p.steps,
        #"Sampler": processing.get_correct_sampler(p)[p.sampler_index].name,
        #"CFG scale": p.cfg_scale,
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        "Model": (None if not opts.add_model_name_to_info or not shared.sd_model.sd_checkpoint_info.model_name else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
        "Hypernet": (None if shared.loaded_hypernetwork is None else shared.loaded_hypernetwork.name),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Eta": (None if p.sampler is None or p.sampler.eta == p.sampler.default_eta else p.sampler.eta),
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta,
    }

    generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: \n" + p.negative_prompt if p.negative_prompt else ""

    return f"{p.prompt[0] if type(p.prompt) == list else p.prompt}{negative_prompt_text}\n{generation_params_text}".strip()


class Script(scripts.Script):
    def title(self):
        return f"랜덤 다이나믹 프롬프트 v{VERSION}"

    def ui(self, is_img2img):
        loops = gr.Slider(minimum=1, maximum=1000, step=1, label='Loops', value=100)
        #denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor', value=1)

        step1 = gr.Slider(minimum=1, maximum=150, step=1, label='step1 min/max', value=10)
        step2 = gr.Slider(minimum=1, maximum=150, step=1, label='step2 min/max', value=30)
        #stepc = gr.Slider(minimum=1, maximum=100, step=1, label='step cnt', value=10)
        cfg1 = gr.Slider(minimum=1, maximum=30, step=1, label='cfg1 min/max', value=6)
        cfg2 = gr.Slider(minimum=1, maximum=30, step=1, label='cfg2 min/max', value=15)
        #cfgc = gr.Slider(minimum=1, maximum=100, step=1, label='cfg cnt', value=10)
        
        fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=True)
        
        html = f"""
            <h3><strong>Combinations</strong></h3>
            리스트에서 단어들을 선택합니다. 이 경우에는 아티스트를 2가지 골라보죠 :
            <code>{{2$artist1|artist2|artist3}}</code>
            $가 입력되지 않으면, 1$가 자동으로 입력됩니다.
            <br>
            범위도 지정 가능합니다 :
            <code>{{1-3$artist1|artist2|artist3}}</code>
            이 경우에는 1~3가지의 아티스트가 랜덤하게 결정됩니다.
            <br/><br/>

            <h3><strong>Wildcards</strong></h3>
            <p>사용가능한 와일드카드</p>
            <ul style="overflow-y:auto;max-height:30rem;">
        """
        
        wildcards = wildcard_manager.get_wildcards()
        html += "".join([f"<li>{wildcard}</li>" for wildcard in wildcards])

        html += "</ul>"
        html += f"""
            <br/>
            <code>와일드카드 경로: {WILDCARD_DIR}</code><br/>
            <small>mywildcards.txt라는 이름의 텍스트 파일을 만들어, 1줄당 1개의 단어를 입력하면 와일드카드를 추가할 수 있습니다. 텍스트 파일은 {WILDCARD_DIR} 에 넣어주세요. 재시작 후 <code>__mywildcards__</code>를 사용할 수 있게 됩니다.</small>
        """
        info = gr.HTML(html)
        return [info,loops,step1, step2,  cfg1, cfg2,  fixed_seeds]

    def run(self, p, info, loops, step1, step2, cfg1, cfg2, fixed_seeds):
        print(f"{loops};{step1};{step2};{cfg1};{cfg2};{fixed_seeds};")
        print(f"{type(loops)};{type(step1)};{type(step2)};{type(cfg1)};{type(cfg2)};{type(fixed_seeds)};")
        
        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        original_seed = p.seed
        #print(f"original_prompt ; {original_prompt}")
        
        print(f"p.outpath_samples ; {p.outpath_samples}")
        os.makedirs(p.outpath_samples, exist_ok=True)
        
        namegen = FilenameGenerator(p, p.seed, p.prompt,Image.new('RGBA', (p.width, p.height)))
        print(f"opts.save_to_dirs ; {opts.save_to_dirs}")
        print(f"opts.directories_filename_pattern ; {opts.directories_filename_pattern}")
        file_decoration = namegen.apply( opts.directories_filename_pattern)
        print(f"file_decoration ; {file_decoration}")
        fullfn=os.path.join(p.outpath_samples,file_decoration)
        print(f"fullfn ; {fullfn}")
        os.makedirs(fullfn, exist_ok=True)
        
        if opts.save_txt and original_prompt is not None:
            txt_fullfn =os.path.join(fullfn,f"{file_decoration}-{self.title()}.txt") 
            print(f"txt_fullfn ; {txt_fullfn}")
            with open(txt_fullfn, "w", encoding="utf8") as file:
                infotexts=create_infotext(p)
                #print(f"p.info ; {p.info}")
                print(f"infotexts ; {infotexts}")
                #print(f"p.job_timestamp ; {p.job_timestamp}")
                file.write(infotexts + "\n")
            
        num_images = p.n_iter * p.batch_size
        
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
                
            all_prompts = [
                generate_prompt(original_prompt) for _ in range(num_images)
            ]
            print(f"loops: {i+1}/{loops} ; steps:{p.steps} ; cfg:{p.cfg_scale}")
            if fixed_seeds:
                p.seed=-1;
                fix_seed(p)

            print(f"p.seed ; {type(p.seed)} ; {p.seed}")
            #all_seeds = [int(p.seed[0] if type(p.seed) == list else p.seed) + (x if p.subseed_strength == 0 else 0) for x in range(num_images)]
            all_seeds = [int(p.seed) + (x if p.subseed_strength == 0 else 0) for x in range(num_images)]
            print(f"all_seeds ; {type(all_seeds)} ; {all_seeds}")
            print(f"Prompt matrix will create {len(all_prompts)} images in a total of {p.n_iter} batches.")
            #logger.info(f"Prompt matrix will create {len(all_prompts)} images in a total of {p.n_iter} batches.")

            p.prompt = all_prompts
            p.seed = all_seeds

            p.prompt_for_display = original_prompt
            processed = process_images(p)

        p.prompt = original_prompt
        p.seed = original_seed

        return processed

wildcard_manager.ensure_directory()
