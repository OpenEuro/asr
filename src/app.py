import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from transcribe.transcribe import transcriber, languages
import gradio as gr
import torch
import torchaudio
import torch.cuda as cuda
import platform
from transformers import __version__ as transformers_version
from dotenv import load_dotenv
import shutil
from docx import Document
import logging
import subprocess
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = cuda.device_count() if torch.cuda.is_available() else 0
cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
cudnn_version = torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
os_info = platform.system() + " " + platform.release() + " " + platform.machine()

# Get the available VRAM for each GPU (if available)
vram_info = []
if torch.cuda.is_available():
    for i in range(cuda.device_count()):
        gpu_properties = cuda.get_device_properties(i)
        vram_info.append(f"**GPU {i}: {gpu_properties.total_memory / 1024**3:.2f} GB**")

pytorch_version = torch.__version__
torchaudio_version = torchaudio.__version__ if 'torchaudio' in dir() else "N/A"

device_info = f"""Running on: **{device}**

    Number of GPUs available: **{num_gpus}**

    CUDA version: **{cuda_version}**

    CuDNN version: **{cudnn_version}**

    PyTorch version: **{pytorch_version}**

    Torchaudio version: **{torchaudio_version}**

    Transformers version: **{transformers_version}**

    Operating system: **{os_info}**

    Available VRAM: 
    \t {', '.join(vram_info) if vram_info else '**N/A**'}
"""

css = """
#audio_input {
    padding-bottom: 50px;
}
"""

def format_srt_time(timestamp):
    """Formats the timestamp into SRT time format."""
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

def generate_srt_content(chunks):
    """Generates the content for an SRT file based on transcription chunks."""
    srt_content = ""
    for i, chunk in enumerate(chunks, start=1):
        try:
            start, end = chunk["timestamp"]
            start_time = format_srt_time(start)
            end_time = format_srt_time(end)
            text = chunk["text"]
            srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
        except:
            logging.info("couldn't add phrase")
            continue
    return srt_content.strip()

def create_black_screen_video(audio_file_path, output_video_path):
    """
    Creates a video with an empty black screen and the original audio from the input audio file.

    Parameters:
    - audio_file_path: Path to the input audio file.
    - output_video_path: Path where the output video will be saved.
    """
    # Check if the output directory exists, create if not
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'lavfi',  # Input format
        '-i', 'color=c=black:s=320x240:r=10',  # Generate a black color input, with 1280x720 resolution at 30 fps
        '-i', audio_file_path,  # The input audio file
        '-c:v', 'libx264',  # Video codec to use
        '-tune', 'stillimage',  # Optimize for still image
        '-c:a', 'aac',  # Audio codec to use
        '-b:a', '192k',  # Audio bitrate
        '-shortest',  # Finish encoding when the shortest input stream ends
        output_video_path  # The output video file path
    ]

    # Execute the command
    subprocess.run(command, check=True)


def process_folder(files_source, model, language, translate, diarize, diarization_token):
    output_folder_path = "./tmp"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_path in files_source:
        # Check if the file is an audio file (e.g., .mp3, .mp4, .wav)
        if file_path.endswith(('.mp3', '.mp4', '.wav')):
 
            file_name = os.path.basename(file_path)

            # Copy the original audio file to the output folder
            output_audio_filepath = os.path.join(output_folder_path, file_name)
            shutil.copy2(file_path, output_audio_filepath)

    #             output_filename_base = os.path.splitext(filename)[0]
    #             output_word_filepath = os.path.join(output_folder_path, output_filename_base + ".docx")
    #             output_srt_filepath = os.path.join(output_folder_path, output_filename_base + ".srt")
    #             output_summary_filepath = os.path.join(output_folder_path, output_filename_base + "_summary.docx")
    #             output_video_filepath = os.path.join(output_folder_path, output_filename_base + ".mp4")
    #             output_audio_filepath = os.path.join(output_folder_path, filename)

    #             # Skip processing if any of the output files already exist
    #             if os.path.exists(output_word_filepath) and os.path.exists(output_srt_filepath) and os.path.exists(output_summary_filepath) and os.path.exists(output_video_filepath) and os.path.exists(output_audio_filepath):
    #                 print(f"Skipping {filename} as output files already exist.")
    #                 continue
                    
            # Use the transcriber function to transcribe the audio file
            transcription_result = transcriber(file_path, model, language=language, translate=translate, diarize=diarize, input_diarization_token=diarization_token)
            print(transcription_result)
    
    #             transcribed_text = transcription_result["text"]
    #             chunks = transcription_result.get("chunks", [])
    
    #             # Create a new Word document with the transcribed text
    #             doc = Document()
    #             for chunk in chunks:
    #                 doc.add_paragraph(chunk["text"])
    #             output_filename_base = os.path.splitext(filename)[0]
    #             output_word_filepath = os.path.join(output_folder_path, output_filename_base + ".docx")
    #             doc.save(output_word_filepath)
    #             print(f"Transcription saved to {output_word_filepath}")
    
    #             # Create an SRT file with subtitles if chunks are available
    #             if chunks:
    #                 srt_content = generate_srt_content(chunks)
    #                 output_srt_filepath = os.path.join(output_folder_path, output_filename_base + ".srt")
    #                 with open(output_srt_filepath, "w", encoding='utf-8') as srt_file:
    #                     srt_file.write(srt_content)
    #                 print(f"Subtitles saved to {output_srt_filepath}")
    
    #             # Generate and save the summary
    #             output_summary_filepath = os.path.join(output_folder_path, output_filename_base + "_summary.docx")

    
    #             # Create empty video
    #             if filename.endswith(('.mp3', '.wav')):
    #                 create_black_screen_video(file_path, os.path.join(output_folder_path, output_filename_base + ".mp4"))
                
    


# def inference(input, diarize, num_speakers:int, strict, lan, trans, progress=gr.Progress()):
def inference(input, model, language, translate, diarize, input_diarization_token):
    tr = transcriber(input, model, language, translate, diarize, input_diarization_token)
    return {textbox: gr.update(value=tr)}

with gr.Blocks(title="Automatic speech recognition (beta)", css=css, analytics_enabled=False) as demo:
    with gr.Row():
        gr.Markdown(
            """
                # Automatic speech recognition

                [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
                
                ![Python 3.10](https://raw.githubusercontent.com/tools4eu/automatic-speech-recognition/main/badges/python3_10.svg)

                Report issues [here](https://github.com/tools4eu/automatic-speech-recognition/issues)
            """

        )
    

    with gr.Tab("Upload/record sound"):
        with gr.Column():
            dropdown_model = gr.Dropdown(
                    label='Model', 
                    choices = ["openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small", "openai/whisper-tiny"],
                    value="openai/whisper-large-v3", 
                    info="""
                        Larger models will increase the quality of the transcription, but reduce performance.
                    """)
        with gr.Row():
            with gr.Column():
                upl_input = gr.Audio(type='filepath', elem_id="audio_input")
                upl_language = gr.Dropdown(
                    label='Language', 
                    choices = ['Automatic detection']+sorted(list(languages.keys())), 
                    value='Automatic detection', 
                    info="""
                        Setting the language to "Automatic detection" will auto-detect the language based on the first 30 seconds.
                        If the language is known upfront, always set it manually.
                    """)

        with gr.Row():
            upl_translate = gr.Checkbox(label='Translate to English')

        with gr.Column():
            with gr.Group():
                input_diarization_token = gr.Textbox(label='Paste your HF token here for speaker diarization (or add it as an environment variable)', value=HF_AUTH_TOKEN)
                check_diarization = gr.Checkbox(label='Speaker diarization')
                with gr.Accordion("For more details click here...", open=False):
                    gr.Markdown("""
                                    An access token can be created [here](https://hf.co/settings/tokens)
                                
                                    If not done yet for your account, you need to [accept segmentation terms & conditions](https://huggingface.co/pyannote/segmentation-3.0)
                                
                                    If not done yet for your account, you need to [accept diarization terms & conditions](https://huggingface.co/pyannote/speaker-diarization-3.1)
                                """)
        
        with gr.Row():
            upl_btn = gr.Button("Transcribe")
        
        with gr.Row(variant='panel'):
            with gr.Column():
                textbox = gr.Textbox(label='Transciption',visible=True)

    with gr.Tab("Process multiple files"):
        files_source=gr.Files(label="Select Audio Files", file_count="multiple")
        with gr.Column():
            dropdown_model_multi = gr.Dropdown(
                    label='Model', 
                    choices = ["openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small", "openai/whisper-tiny"],
                    value="openai/whisper-large-v3", 
                    info="""
                        Larger models will increase the quality of the transcription, but reduce performance.
                    """)
        dropdown_lang_multi = gr.Dropdown(
                    label='Language', 
                    choices = ['Automatic detection']+sorted(list(languages.keys())), 
                    value='Automatic detection', 
                    info="""
                        Setting the language to "Automatic detection" will auto-detect the language based on the first 30 seconds.
                        If the language is known upfront, always set it manually.
                    """)
        checkbox_trans_multi = gr.Checkbox(label='Translate to English')
        with gr.Column():
            with gr.Group():
                input_diarization_token_multi = gr.Textbox(label='Paste your Hugging Face token here for speaker diarization (or add it as an environment variable)', value=HF_AUTH_TOKEN)
                check_diarization_multi = gr.Checkbox(label='Speaker diarization')
                with gr.Accordion("For more details click here...", open=False):
                    gr.Markdown("""
                                    An access token can be created [here](https://hf.co/settings/tokens)
                                
                                    If not done yet for your account, you need to [accept segmentation terms & conditions](https://huggingface.co/pyannote/segmentation-3.0)
                                
                                    If not done yet for your account, you need to [accept diarization terms & conditions](https://huggingface.co/pyannote/speaker-diarization-3.1)
                                """)
        btn_transcribe_multi= gr.Button("Transcribe")
        textbox_transcribe_multi= gr.Chatbot(label='Transciption',visible=True)

    with gr.Tab("Device info"):
        gr.Markdown(device_info, label="Hardware info & installed packages")
        # gr.Markdown(device_info, label="Hardware info & installed packages", lines=len(device_info.split("\n")), container=False)

    transcribe_event = upl_btn.click(fn=inference, inputs=[upl_input, dropdown_model, upl_language, upl_translate, check_diarization, input_diarization_token], outputs=[textbox], concurrency_limit=1)
    # transcribe_files_event = btn_transcribe_folder.click(fn=process_folder, inputs=[files_source, dropdown_lang_multi, checkbox_trans_multi, input_diarization_token], outputs=[textbox_transcribe_folder], concurrency_limit=1)
    transcribe_files_event = btn_transcribe_multi.click(fn=process_folder, inputs=[files_source, dropdown_model_multi, dropdown_lang_multi, check_diarization_multi, checkbox_trans_multi, input_diarization_token_multi], outputs=[], concurrency_limit=1)

demo.queue().launch(server_name="0.0.0.0")