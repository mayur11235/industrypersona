import os
import asyncio
import openai
from dotenv import load_dotenv
from helper import timing_decorator
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class CHATGPT:
    @timing_decorator
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        
    @timing_decorator
    async def get_response_async(self, messages):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.get_response, messages)
        return response    

    @timing_decorator
    def get_response(self,messages):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    chatgpt = CHATGPT()
    