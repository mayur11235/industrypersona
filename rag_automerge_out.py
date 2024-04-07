import os
import logging
import asyncio
from llama_index import Document
from llama_index.schema import MetadataMode
from llama_index import SimpleDirectoryReader
from llama_index.llms import OpenAI
from dotenv import load_dotenv
from helper import timing_decorator , create_metadata_filter
from llama_index.memory import ChatMemoryBuffer
from llama_index.postprocessor import LLMRerank
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.chat_engine import ContextChatEngine
from llama_index.core.llms.types import ChatMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PERSONA_WRITING_ASSISTANT =("You are a writing assistant for industry persona.\n"
                            "Your goal is to provide clear and engaging content ideas based on user prompt and query.\n"   
                            "Use the previous chat history, or the context below, to interact and help the user.\n"                                 
                            )
ADDITIONAL_SYSTEM_INSTRUCTION=("")                            

load_dotenv()
class RAGAMGPT:
    @timing_decorator
    def __init__(self,con=None,k=5,top=3,filters=None):
        self.llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0,max_tokens=500) 
        self.index = con.index
        self.storage_context=con.storage_context
        self.service_context=con.service_context
        self.metafilters=create_metadata_filter(filters)
        self.base_retriever = self.index.as_retriever(similarity_top_k=k,filters=self.metafilters)
        self.vector_retriever=AutoMergingRetriever(self.base_retriever, self.storage_context, verbose=True)
        self.llmreranker = LLMRerank(top_n=top,service_context=self.service_context) 
        self.context_template = ("Context information is below."
                                "\n--------------------\n"
                                "{context_str}"
                                "\n--------------------\n"+ADDITIONAL_SYSTEM_INSTRUCTION                                        
                                )                    
        self.chat_engine =  ContextChatEngine( 
                            retriever=self.vector_retriever,
                            llm=self.llm,
                            prefix_messages=[ChatMessage(content=PERSONA_WRITING_ASSISTANT, role=self.llm.metadata.system_role)],
                            memory=ChatMemoryBuffer.from_defaults(token_limit=15000),
                            context_template=self.context_template,
                            #node_postprocessors=[self.llmreranker]
                            )
        self.query_engine = self.index.as_query_engine(
                            retriever=self.vector_retriever,
                            llm=self.llm,
                            prefix_messages=[ChatMessage(content=PERSONA_WRITING_ASSISTANT, role=self.llm.metadata.system_role)],
                            context_template=self.context_template,
                            )                    
        self._query_log = []
        self._context_history = []   
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.chat_engine.reset()
    
    @timing_decorator
    def get_query_log(self):
        return self._query_log

    @timing_decorator
    def get_context_history(self):
        return self._context_history

    @timing_decorator
    async def get_response_async(self, prompt_txt,query_str=None):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.get_response,prompt_txt,query_str)
        return response    

    @timing_decorator
    def get_response(self,prompt_txt,query_str=None):
        if query_str is not None and query_str != "":
            chat_query = (f"{prompt_txt}\n"
                            "Consider additional user Instruction:"
                            "\n--------------------\n"
                            f"{query_str}"
                            "\n--------------------\n"                                                    
                        )
        else:
            chat_query = prompt_txt
        chat_response = self.chat_engine.chat(chat_query)
        response = chat_response.response
        context=str(chat_response.sources[0])
        if context is not None and context != "":
            if context.find("--------------------")!= -1:
                context =context.split("--------------------")[1]
            self._context_history.append(context)
        self._query_log.append({"role": "user", "content":{'query_str':query_str,'context_str':context,'prompt_txt':prompt_txt}})
        self._query_log.append({"role": "assistant", "content":response})
        return response , self._context_history[-1] if context is None or context == "" else context

if __name__ == "__main__":
    rag = RAGAMGPT()
