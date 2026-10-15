import argparse
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

CSS = """
.gradio-container {
    width: 100% !important;
    box-sizing: border-box;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 24px 32px 32px !important;
    background: #ffffff;
    letter-spacing: 0;
}
#app-header { border-bottom: 1px solid #e3e7eb; padding-bottom: 16px; margin-bottom: 16px; }
#app-header h1 { font-size: 22px; line-height: 1.4; font-weight: 650; margin: 0; }
#workspace { gap: 28px; align-items: stretch; }
#source-pane, #controls { min-width: 0 !important; gap: 14px; }
#source-title h2, #result-title h2 { font-size: 17px; line-height: 24px; font-weight: 600; margin: 0; }
#source-editor { border-radius: 6px; border-color: #dce2e7; }
#controls { padding-top: 38px; }
#generation-settings { border: 1px solid #dce2e7; border-radius: 6px; background: #f7f9fa; box-shadow: none; }
#generation-settings > button { padding: 14px 16px; }
#instruction { flex: 1 1 auto; display: flex; flex-direction: column; border: 0; padding: 0; }
#instruction > label { display: flex; flex-direction: column; flex: 1; gap: 8px; }
#instruction textarea { flex: 1; min-height: 280px; border: 1px solid #dce2e7; border-radius: 6px; padding: 14px; font-size: 15px; line-height: 1.65; resize: vertical; }
#instruction textarea:focus { border-color: #0d9488; outline: 2px solid #ccfbf1; }
#generate-button { flex: 0 0 50px; min-height: 50px; border: 0; border-radius: 6px; background: #0d9488; color: white; font-size: 16px; font-weight: 600; box-shadow: none; }
#generate-button:hover { background: #0f766e; }
#generate-button:disabled { opacity: 0.6; }
#result-section { margin: 24px auto 0; padding-top: 18px; border-top: 1px solid #e3e7eb; gap: 14px; width: 100%; scroll-margin-top: 20px; }
#result-frame { width: 100%; border: 1px solid #dce2e7; border-radius: 6px; background: #fafbfc; padding: 24px 0; }
#result-image { width: 83.333333% !important; min-width: 0 !important; max-width: calc(100% - 48px); margin: 0 auto; border: 0; border-radius: 0; min-height: 216px; background: transparent; }
#result-image img { object-fit: contain !important; width: 100%; height: auto; max-height: none !important; }
@media (max-width: 760px) {
    .gradio-container { padding: 16px !important; }
    #workspace { flex-direction: column; gap: 20px; }
    #source-pane, #controls { width: 100%; flex-basis: auto !important; }
    #controls { padding-top: 0; }
    #source-editor { height: 380px !important; }
    #instruction textarea { min-height: 200px; }
    #result-section { margin-top: 20px; padding-top: 16px; }
}
"""

SCROLL_TO_RESULT = """async () => {
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const images = document.querySelectorAll('#result-image img');
    await Promise.all(Array.from(images, image => image.decode().catch(() => {})));
    const page = document.scrollingElement || document.documentElement;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollTo({top: page.scrollHeight, behavior: reducedMotion ? 'instant' : 'smooth'});
    return [];
}"""


def create_demo():
    import gradio as gr
    from PIL import Image
    from core.interface import generate, prepare_editor

    def generate_editor(editor, mask, mode, prompt, steps, cfg, seed, random_seed):
        try:
            source, _, _, region, prompt = prepare_editor(editor, mask, mode, prompt)
            result, _, _ = generate(source, region, prompt, steps, cfg, seed, random_seed)
            # Restore the source aspect ratio for display; model inputs stay unchanged.
            original_size = editor['background'].size
            if result.size != original_size:
                result = result.resize(original_size, Image.Resampling.LANCZOS)
            return result
        except (ValueError, TypeError, OverflowError, FileNotFoundError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title='OminiSketch') as demo:
        gr.Markdown('# OminiSketch', elem_id='app-header')
        with gr.Row(elem_id='workspace'):
            with gr.Column(scale=3, min_width=320, elem_id='source-pane'):
                gr.Markdown('## 原图与涂鸦', elem_id='source-title')
                editor = gr.ImageEditor(
                    type='pil', image_mode='RGBA', sources=['upload', 'clipboard'],
                    transforms=[], format='png', show_label=False, label='原图与涂鸦',
                    height=512, elem_id='source-editor',
                    brush=gr.Brush(default_size=35, colors=['#ef4444'],
                                   default_color='#ef4444', color_mode='fixed'))
            with gr.Column(scale=2, min_width=300, elem_id='controls'):
                with gr.Accordion('生成设置', open=False, elem_id='generation-settings'):
                    mode = gr.Radio(['绘制区域', '上传区域'], type='index',
                                    value='绘制区域', label='区域来源')
                    mask = gr.Image(type='pil', image_mode='RGB', format='png',
                                    visible=False, sources=['upload'],
                                    label='区域文件（黑色为编辑区）', height=180)
                    steps = gr.Slider(10, 40, value=28, step=1, label='步数')
                    cfg = gr.Slider(1, 4, value=2, step=0.25, label='文字引导')
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, minimum=0,
                                         maximum=2147483647, label='种子')
                        random_seed = gr.Checkbox(value=False, label='随机种子')
                prompt = gr.Textbox(value='', label='文本指令', lines=12,
                                    max_lines=24, elem_id='instruction')
                button = gr.Button('生成', variant='primary', elem_id='generate-button')
        with gr.Column(elem_id='result-section'):
            gr.Markdown('## 生成结果', elem_id='result-title')
            with gr.Column(elem_id='result-frame'):
                output = gr.Image(label='生成结果', show_label=False, interactive=False,
                                  format='png', height=None, elem_id='result-image')

        mode.change(lambda value: gr.update(visible=value == 1), mode, mask, queue=False)
        editor.upload(lambda: None, outputs=mask, queue=False)
        editor.clear(lambda: None, outputs=mask, queue=False)
        random_seed.change(lambda value: gr.update(interactive=not value),
                           random_seed, seed, queue=False)
        generated = button.click(
            generate_editor, [editor, mask, mode, prompt, steps, cfg, seed, random_seed],
            output, api_name='generate',
            concurrency_limit=1, trigger_mode='once')
        generated.success(fn=None, inputs=None, outputs=None,
                          js=SCROLL_TO_RESULT, queue=False, api_name=False)
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
    from core.interface import config, output_root
    demo = create_demo()
    demo.queue(max_size=4, default_concurrency_limit=1)
    demo.launch(
        server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False,
        show_error=False, footer_links=[], max_file_size='10mb', enable_monitoring=False,
        allowed_paths=[str(output_root()), str(cache)],
        blocked_paths=[str(ROOT / 'model'), config.data_dir, str(Path.home() / '.ssh')],
        theme=gr.themes.Default(primary_hue='teal', secondary_hue='rose', neutral_hue='gray'),
        css=CSS)


if __name__ == '__main__':
    main()
