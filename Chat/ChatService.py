import logging
import os
import time
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


class ChatService():
    def __init__(self, LANG="CN",
                 MODEL_PATH='/media/zzg/GJ_disk01/pretrained_model/XeIaso/yi-chat-6B-GGUF/yi-chat-6b.Q5_K_M.gguf'):
        logging.info('Initializing ChatService Service...')
        if LANG == "CN":
            prompt_path = "Chat/prompts/example-cn.txt"
        else:
            prompt_path = "Chat/prompts/example-en.txt"
        with open(prompt_path, 'r', encoding='utf-8') as file:
            template = file.read().strip()  # {dialogue}
        self.prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])
        self.llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=1,  # Metal set to 1 is enough.
            n_batch=512,  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
            n_ctx=4096,  # Update the context window size to 4096
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            # callback_manager=callback_manager,
            stop=["<|im_end|>"],
            verbose=False,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
        )
        self.dialogue = ""
        logging.info('API Chatbot initialized.')

    def predict(self, text):
        stime = time.time()
        print("Generating...")
        self.dialogue += "*Q* {}\n".format(text)
        prompt = self.prompt_template.format(dialogue=self.dialogue)

        reply = self.llm(prompt, max_tokens=4096)
        # 运行流式推理
        # output = self.llm(prompt, max_tokens=4096, stream=True)
        # for result in output:
        #     if "choices" in result:
        #         text = result["choices"][0]["text"]
        #         yield text
        logging.info('Chat Response: %s, time used %.2f' % (reply, time.time() - stime))
        return reply

    def predict_stream(self, text):
        prev_text = ""
        complete_text = ""
        stime = time.time()
        if self.counter % 5 == 0 and self.chatVer == 1:
            if self.brainwash:
                logging.info('Brainwash mode activated, reinforce the tune.')
            else:
                logging.info('Injecting tunes')
            asktext = self.tune + '\n' + text
        else:
            asktext = text
        self.counter += 1
        for data in self.chatbot.predict(asktext) if self.chatVer == 1 else self.chatbot.predict_stream(text):
            message = data["message"][len(prev_text):] if self.chatVer == 1 else data

            if ("。" in message or "！" in message or "？" in message or "\n" in message) and len(complete_text) > 3:
                complete_text += message
                logging.info('ChatGPT Stream Response: %s, @Time %.2f' % (complete_text, time.time() - stime))
                yield complete_text.strip()
                complete_text = ""
            else:
                complete_text += message

            prev_text = data["message"] if self.chatVer == 1 else data

        if complete_text.strip():
            logging.info('ChatGPT Stream Response: %s, @Time %.2f' % (complete_text, time.time() - stime))
            yield complete_text.strip()
