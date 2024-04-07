import os
from dotenv import load_dotenv
import numpy as np
from trulens_eval import Feedback,TruLlama,OpenAI
from trulens_eval.feedback import Groundedness
import nest_asyncio
from rag_automerge_out import RAGAMGPT
from rag_out import RAGGPT
from helper import log_chat_and_feedback, initialize_index ,gunzip_file ,limit_to_n_words
from trulens_eval import Tru

nest_asyncio.apply()
load_dotenv()
openai = OpenAI()
Tru().reset_database()

qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]

def get_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder

def get_vector_index():   
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if 'RAG_MODE' in os.environ and os.environ['RAG_MODE'] == 'advanced':
        rag_mode="advanced"
        gunzip_file("resources/docstorepath/docstore.json.gz","resources/docstorepath/docstore.json")
    else:
        rag_mode = "default"  
    return initialize_index(rag_mode)  

print("Creating Rag object.")
if 'RAG_MODE' in os.environ and os.environ['RAG_MODE'] == 'advanced':    
    rag_obj=RAGAMGPT(get_vector_index(),k=4,top=3)   
    print("<<Running in Advanced Rag Mode>>")
else:
    rag_obj=RAGGPT(get_vector_index(),k=4,top=3)
    

eval_questions = []
files = os.listdir("eval_prompts")
for file_name in files:
    file_path = os.path.join("eval_prompts", file_name)
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace('\n', ' ')

    eval_questions.append(content)

tru_recorder = get_trulens_recorder(
    rag_obj.query_engine,
    app_id ='Industry Persona'
)

def run_evals(eval_questions, tru_recorder, rag_obj):
    for question in eval_questions:
        with tru_recorder as recording:
            response = rag_obj.query_engine.query(question)
            
run_evals(eval_questions, tru_recorder, rag_obj)

Tru().get_leaderboard(app_ids=[])

Tru().run_dashboard()