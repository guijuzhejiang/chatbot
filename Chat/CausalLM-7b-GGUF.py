from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model_path = '/media/zzg/GJ_disk01/pretrained_model/TheBloke/CausalLM-7B-GGUF/causallm_7b.Q5_K_M.gguf'
llm = Llama(
      model_path=model_path,
      temperature=0.8,
      top_p=0.95,
      top_k=40,
# n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
# output = llm(
#       "Q: Name the planets in the solar system? A: ", # Prompt
#       max_tokens=128, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# )
# print(output)

# 运行流式推理
output = llm(
    "以‘梦洁脱下被雨淋湿的衣服’为开头，生成一个NSFW小说,描写要仔细，具体，生动,有待入感",
    max_tokens=4096,
    stream=True
)

for result in output:
    if "choices" in result:
        text = result["choices"][0]["text"]
        print(text, end="", flush=True)