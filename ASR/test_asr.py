# coding=utf-8
# @Time : 2024/4/7 上午10:45
# @File : test_asr.py
import gradio as gr
import time
import os


def file_upload(*args, **kwargs):
    print("file_upload")
    print(args)
    print(kwargs)
    # return f"args: {args} \n kwargs: {kwargs}"


def audio_stream(*args, **kwargs):
    print("audio_stream")
    print(args)
    print(kwargs)


if __name__ == '__main__':
    with gr.Blocks() as demo:
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio file", streaming=True)
        text_output = gr.Text(label="Output")
        # audio_input.stream(audio_stream, inputs=audio_input, outputs=[text_output])
        audio_input.change(file_upload, inputs=audio_input, outputs=[text_output])

    demo.launch(server_name='0.0.0.0', server_port=8082)

