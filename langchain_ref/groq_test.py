import os

from groq import Groq
import dotenv
dotenv.load_dotenv()


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            # "content": "Explain the importance of low latency LLMs",
            "content": "who is DiCaprio?",
        }
    ],
    model="mixtral-8x7b-32768",
)

print(chat_completion.choices[0].message.content)