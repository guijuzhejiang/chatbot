import gradio as gr
from xiangsheng_langchain import *

with gr.Blocks() as demo:
    url = gr.Textbox()
    chatbot = gr.Chatbot()
    submit_btn = gr.Button('生成相声')

    def generate_conversation(url):
        xiangsheng: XiangSheng = url2xiangsheng(url)
        chat_history = []
        def parse_line(line:Line):
            if line is None:
                return ''
            return f'{line.character}:{line.content}'
        for i in range(0,len(xiangsheng.script), 2):
            line1 = xiangsheng.script[i]
            line2 = xiangsheng.script[i+1] if (i+1)<len(xiangsheng.script) else None
            chat_history.append((parse_line(line1), parse_line(line2)))
        return chat_history

    submit_btn.click(fn=generate_conversation, inputs=url, outputs=chatbot)

if __name__=="__main__":
    demo.launch()