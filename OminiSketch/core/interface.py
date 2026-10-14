import json
import threading
import uuid
from pathlib import Path
from PIL import PngImagePlugin
from .before import config, load_model, prepare_editor
from .pipeline import run_edit

PIPE = None
LOCK = threading.Lock()


def output_root():
    
    root = Path(__file__).resolve().parents[1]
    path = Path(config.get("output_dir", "../output")).expanduser()
    
    return (root/path).resolve() if not path.is_absolute() else path.resolve()


def _get_pipeline():
    
    global PIPE
    
    if PIPE is None:
        PIPE = load_model()
    
    return PIPE


def generate(source, mask, prompt, steps=28, cfg=2., seed=42, random_seed=False):
    
    with LOCK:
        
        result, source, preview, prompt, record = run_edit(
            source, mask, prompt, _get_pipeline, steps, cfg, seed, random_seed)
        
        folder = output_root()/uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        record["output_folder"] = str(folder)
        
        source.save(folder/"source.png")
        preview.save(folder/"mask.png")
        (folder/"prompt.txt").write_text(prompt, encoding="utf-8")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("ominisketch_parameters", json.dumps(record))
        result.save(folder/"result.png", pnginfo=metadata)
        
        return result, preview, record


__all__ = ["generate", "prepare_editor", "output_root", "config"]
