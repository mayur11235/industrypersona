import os
import logging
import asyncio
from llama_index import Document
from llama_index.schema import MetadataMode
from llama_index import SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index import PromptTemplate
from llama_index.response_synthesizers import TreeSummarize
from llama_index.agent import OpenAIAgent
from dotenv import load_dotenv
from helper import timing_decorator , create_metadata_filter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class RAGGPT:
    @timing_decorator
    def __init__(self,con=None,k=2,filters=None):
        self.llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0,max_tokens=300) 
        #self.role = role
        self.index = con.index
        self.agent = OpenAIAgent.from_tools(llm=self.llm)
        self.vector_retriever = self.initialize_retriever(k,filters)
        self.qa_prompt_tmpl = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "{prompt_txt}\n"
        "Query: {query_str}\n"
        "Answer: "
        )
        self.qa_prompt = PromptTemplate(self.qa_prompt_tmpl)
        self.summarizer = TreeSummarize(verbose=True, summary_template=self.qa_prompt)
        self._query_log = []
        self._context_history = []   
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    @timing_decorator
    def get_query_log(self):
        return self._query_log

    @timing_decorator
    def get_context_history(self):
        return self._context_history

    @timing_decorator
    def initialize_retriever(self,k,filters=None):
        metafilters=create_metadata_filter(filters)
        vector_retriever = self.index.as_retriever(similarity_top_k=k,filters=metafilters)
        return vector_retriever

    @timing_decorator
    async def retrieve_and_process(self,txt):
        #Await the coroutine and get the task result
        nodes_with_scores = await (self.vector_retriever.aretrieve(txt))
        return nodes_with_scores
    
    @timing_decorator
    def summarize_context(self,context_str):
        summarize_var="Summarize TextBelow:\n\n"+" ".join(context_str)
        summary=str(self.agent.chat(summarize_var))
        return summary
    
    @timing_decorator
    async def get_context_str_async(self, txt, include_history=False):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.get_context_str, txt, include_history)
        return response 

    @timing_decorator
    def get_context_str(self,txt, include_history=False):
        nodes_with_scores = asyncio.run(self.retrieve_and_process(txt))

        context_str = [x.text for x in nodes_with_scores]
        # Enable LLM call to summarize
        summarized_context = self.summarize_context(context_str)
        # If disabled above enable below statemet or viceversa
        #summarized_context = " ".join(context_str)
        
        self._context_history.append(summarized_context)
        if include_history:
            return context_str + self._context_history
        else:
            return context_str

    @timing_decorator
    async def get_response_async(self, query_str,context_str,prompt_txt):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.get_response, query_str,context_str,prompt_txt)
        return response    

    @timing_decorator
    def get_response(self,query_str,context_str,prompt_txt):
        rag_response = self.summarizer.get_response(query_str,context_str,prompt_txt=prompt_txt)
        self._query_log.append({"role": "system", "content":{'query_str':query_str,'context_str':context_str,'prompt_txt':prompt_txt}})
        self._query_log.append({"role": "assistant", "content":rag_response})
        #self._query_log[len(self._query_log)] = self.qa_prompt_tmpl.format(context_str=context_str, prompt_txt=prompt_txt, query_str=query_str) + rag_response
        return rag_response

if __name__ == "__main__":
    rag = RAGGPT()