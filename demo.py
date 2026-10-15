import argparse
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'OminiSketch'
sys.path.insert(0, str(CODE))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

def create_demo():
    import gradio as gr
    from core.interface import prepare_editor
    from core.interface import generate

    def generate_editor(editor, mask, mode, prompt, steps, cfg, seed, random_seed):
        try:
            source, _, _, region, prompt = prepare_editor(editor, mask, mode, prompt)
            result, _, _ = generate(source, region, prompt, steps, cfg, seed, random_seed)
            return result
        except (ValueError, TypeError, OverflowError, FileNotFoundError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

        
    with gr.Blocks(title='OminiSketch') as demo:

        gr.Markdown('## OminiSketch')
        
        with gr.Row():

            with gr.Column(min_width=320):

                editor = gr.ImageEditor(type='pil', image_mode='RGBA', sources=['upload', 'clipboard'], transforms=[], format='png', label='原图与涂鸦', height=512, brush=gr.Brush(default_size=35, colors=['#ef4444'], default_color='#ef4444', color_mode='fixed'))
                mode = gr.Radio(['绘制区域', '上传区域'], type='index', value='绘制区域', label='区域来源')
                mask = gr.Image(type='pil', image_mode='RGB', format='png', visible=False, sources=['upload'], label='区域文件（黑色为编辑区）', height=220)
                prompt = gr.Textbox(value='', label='编辑指令', lines=2)
                
                with gr.Row():
                    steps = gr.Slider(10, 40, value=28, step=1, label='步数')
                    cfg = gr.Slider(1, 4, value=2, step=0.25, label='文字引导')
                
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, minimum=0, maximum=2147483647, label='种子')
                    random_seed = gr.Checkbox(value=False, label='随机种子')
                
                button = gr.Button('生成', variant='primary')
            
            with gr.Column(min_width=320):
                
                output = gr.Image(label='生成结果', interactive=False, format='png', height=512)
                
        
        mode.change(lambda value: gr.update(visible=value == 1), mode, mask, queue=False)
        editor.upload(lambda: None, outputs=mask, queue=False)
        editor.clear(lambda: None, outputs=mask, queue=False)
        random_seed.change(lambda value: gr.update(interactive=not value), random_seed, seed, queue=False)
        
        button.click(generate_editor, [editor, mask, mode, prompt, steps, cfg, seed, random_seed], [output], api_name='generate', concurrency_limit=1, trigger_mode='once')
    
    return demo

def main():
    
    parser = argparse.ArgumentParser(description='Latest OminiSketch editor')
    parser.add_argument('--port', type=int, default=7863)
    args = parser.parse_args()
    
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required')
    
    cache = CODE / '.cache' / ('ui_' + uuid.uuid4().hex)
    cache.mkdir(parents=True, exist_ok=False)
    
    os.environ['GRADIO_TEMP_DIR'] = str(cache)
    os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1'
    
    import gradio as gr
    from core.interface import output_root
    from core.interface import config
    
    demo = create_demo()
    demo.queue(max_size=4, default_concurrency_limit=1)
    demo.launch(server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False, show_error=False, footer_links=[], max_file_size='10mb', enable_monitoring=False, allowed_paths=[str(output_root()), str(cache)], blocked_paths=[str(ROOT / 'model'), config.data_dir, str(Path.home() / '.ssh')], theme=gr.themes.Default(primary_hue='teal', secondary_hue='rose', neutral_hue='gray'))


if __name__ == '__main__':
    main()
