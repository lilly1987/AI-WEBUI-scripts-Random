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
from modules import processing

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

WILDCARD_DIR = getattr(opts, "wildcard_dir", "scripts/wildcards")
MAX_RECURSIONS = 20
VERSION = "0.6.0"
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
        
class Script(scripts.Script):
    def title(self):
        return f"Random Dynamic Prompting v{VERSION}"

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
            Choose a number of terms from a list, in this case we choose two artists
            <code>{{2$$artist1|artist2|artist3}}</code>
            If $$ is not provided, then 1$$ is assumed.
            <br>
            A range can be provided:
            <code>{{1-3$$artist1|artist2|artist3}}</code>
            In this case, a random number of artists between 1 and 3 is chosen.
            <br/><br/>

            <h3><strong>Wildcards</strong></h3>
            <p>Available wildcards</p>
            <ul style="overflow-y:auto;max-height:30rem;">
        """
        
        wildcards = wildcard_manager.get_wildcards()
        html += "".join([f"<li>{wildcard}</li>" for wildcard in wildcards])

        html += "</ul>"
        html += f"""
            <br/>
            <code>WILDCARD_DIR: {WILDCARD_DIR}</code><br/>
            <small>You can add more wildcards by creating a text file with one term per line and name is mywildcards.txt. Place it in {WILDCARD_DIR}. <code>__mywildcards__</code> will then become available.</small>
        """
        info = gr.HTML(html)
        return [info,loops,step1, step2,  cfg1, cfg2,  fixed_seeds]

    def run(self, p, info, loops, step1, step2, cfg1, cfg2, fixed_seeds):
        print(f"{loops};{step1};{step2};{cfg1};{cfg2};{fixed_seeds};")
        print(f"{type(loops)};{type(step1)};{type(step2)};{type(cfg1)};{type(cfg2)};{type(fixed_seeds)};")
        
        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        original_seed = p.seed
        print(f"original_prompt ; {original_prompt}")
        
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
