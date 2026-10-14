"""Run the same three-input editor as demo.py without launching a web page."""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'OminiSketch'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--mask', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--cfg', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    from PIL import Image
    from core.interface import generate
    with Image.open(args.source) as image:
        source = image.convert('RGB')
    with Image.open(args.mask) as image:
        mask = image.convert('RGB')
    _, _, record = generate(source, mask, args.prompt, args.steps, args.cfg, args.seed)
    print(record['output_folder'])
if __name__ == '__main__':
    main()
