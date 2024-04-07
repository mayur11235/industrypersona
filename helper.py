import os
import pandas as pd
import time
import inspect
from dotenv import load_dotenv
from collections import namedtuple
import pinecone
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.vector_stores.types import MetadataFilter,MetadataFilters,FilterOperator,FilterCondition
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.prompts import PromptTemplate
import gzip
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
        # parameters = inspect.signature(func).parameters
        # if 'self' in parameters:
        #      class_name = inspect.getmodule(func).__name__
        #      print(f"Calling: Class {class_name}, Function {func.__name__}")
        # else:
        #      print(f"Calling: Function: {func.__name__}")
        # start_time = time.time()
        # result = func(*args, **kwargs)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # if 'self' in parameters:
        #     class_name = inspect.getmodule(func).__name__
        #     print(f"Class {class_name}, Function {func.__name__} took {execution_time:.6f} seconds to execute.")
        # else:
        #     print(f"Function: {func.__name__} took {execution_time:.6f} seconds to execute.")
        # return result
    return wrapper

def log_chat_and_feedback(df,user_id='default',rating=-1,q1=-1,q2=-1,q3=-1,q4=-1,feedback=''):
    df.replace('\n', ' ', regex=True, inplace=True)
    logging.info(f"UserRating|^{user_id}|^{rating}")
    logging.info(f"UserQ1Feedback|^{user_id}|^{q1}")
    logging.info(f"UserQ2Feedback|^{user_id}|^{q2}")
    logging.info(f"UserQ3Feedback|^{user_id}|^{q3}")
    logging.info(f"UserQ4Feedback|^{user_id}|^{q4}")
    for i, row in df.iterrows():
        row_str = '|^'.join(map(str, ["|^"+str(i)] + row.tolist()))
        logging.info(f"UserChat|^{user_id}|{row_str}")
    logging.info(f"UserFeedback|^{user_id}|^{feedback}")

@timing_decorator
def initialize_index(mode="default"):
    pinecone_api_key = os.environ['PINECONE_API_KEY']
    pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
    pinecone_index = pinecone.Index('cfo-data')
    logging.info(pinecone_index.describe_index_stats())
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index,add_sparse_vector=False)
    if(mode=="advanced"):
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0,max_tokens=500),
                                                    embed_model="local:BAAI/bge-large-en-v1.5")
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="resources/docstorepath")
        storage_context = StorageContext.from_defaults(docstore=docstore)
    else:    
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0,max_tokens=500),
                                                    embed_model="local:BAAI/bge-large-en-v1.5",
                                                    chunk_size=512, 
                                                    chunk_overlap=50)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Result = namedtuple('Result', ['index', 'storage_context','service_context'])    
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context,service_context=service_context)
    return Result(index , storage_context ,service_context)

def create_metadata_filter(metadata_list):
    if metadata_list is None:
        return None
    filters = [MetadataFilter(key=key, value=value) for item in metadata_list for key, value in item.items()]
    return MetadataFilters(filters=filters, condition=FilterCondition.OR)

def create_bullet_points(strings_list):
    bullet_points = "\n\n".join([f"- {string}" for string in strings_list])
    return bullet_points

def limit_to_n_words(input_text,n=1000):
    input_text = input_text.replace('\n', ' ')
    words = input_text.split()
    limited_words = ' '.join(words[:n])
    return limited_words

def gunzip_file(input_path, output_path):
    with gzip.open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        f_out.write(f_in.read())
        