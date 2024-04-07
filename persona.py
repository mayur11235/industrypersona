import os
import asyncio
import streamlit as st
import pandas as pd
from os.path import dirname, abspath
from dotenv import load_dotenv
from rag_out import RAGGPT
from rag_automerge_out import RAGAMGPT
from chatgpt_out import CHATGPT
from helper import log_chat_and_feedback, initialize_index ,gunzip_file ,limit_to_n_words
import streamlit as st
from streamlit_feedback import streamlit_feedback
import uuid
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

@st.cache_resource(show_spinner="Starting application - Connecting to Vector index.")
def get_vector_index():   
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if 'RAG_MODE' in os.environ and os.environ['RAG_MODE'] == 'advanced':
        rag_mode="advanced"
        gunzip_file("resources/docstorepath/docstore.json.gz","resources/docstorepath/docstore.json")
    else:
        rag_mode = "default"
    return initialize_index(rag_mode)    
get_vector_index()    

user_id=str(uuid.uuid4()) #User id

#ti is a csv with sp500 tickers and their industries; streamlit needs full 
script_path = abspath(dirname(__file__))
ticker_file= os.path.join(script_path,"resources" ,'sp_tickers_inds.csv')
ti = pd.read_csv(ticker_file)

st.markdown("<h3 style='text-align: center;'>Deloitte Industry Persona Prompt Generator</h3>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>How to use:</h3>", unsafe_allow_html=True)
    st.markdown("""
                Our current data corpus includes data specific to **CFOs** of the **Technology** sector.  \n  \
                  \n  Make a selection of **Industry**, **Role**, and **Company**, or leave any field unspecified as "All".  \n\
                  \n  Adjust the big 5 personality traits of the persona with the sliders if you would like, or leave them as Neutral  \n\
                  \n  **Input Article** is used to help guide the persona to relevant data sources for any potential queries or can be used for summarization tasks \n\
                  \n  **Personality Prompt** will be automatically generated based on the inputs provided.  \n\
                  \n  **Basic Queries** are there for your convienience. Use the Custom Query box to ask anything you like, example: \
                "How does the article impact your business?" \n\
                  \n  For comparison purposes each query will display the output of ChatGPT (**GPT Output**) and our persona (**RAG Output**), which has access to our data corpus. \n\
                  \n  We will also display the retrieved context (**RAG Context**) used to form the rationale for the persona's answer.
                  \n The user can ask follow up questions as needed, start over, provide feedback on the persona, and download the interaction.
                """
                )

ind_val = st.selectbox('Industry',['All',
'Consumer',
'Energy, Resources & Industrials',
'Financial Services',
'Government & Public Services',
'Life Sciences & Health Care',
'Technology, Media & Telecommunications'],index=6)

role_val = st.selectbox('Roles',['All','CEO','CFO','COO','CTO', 'CIO', 'CMO', 'CHRO', 'CLO', 'CSO', 'CDO', 'CAIO'],index=2)

comp_val = st.selectbox('Companies', ['All',*ti.loc[ti['Deloitte Industry Name']==ind_val,'Ticker and Name']] )
st.write('LLM Personality')
co = st.select_slider('Closed vs Open', options=['Very Closed', 'Closed','Neutral','Open',' Very Open'], value='Neutral',label_visibility='collapsed')
sc = st.select_slider('Spontaneous vs Conscientious', options=['Very Spontaneous', 'Spontaneous','Neutral','Conscientious','Very Conscientious'], value='Neutral',label_visibility='collapsed')
ie = st.select_slider('Introverted vs Extroverted', options=['Very Introverted', 'Introverted','Neutral','Extroverted','Very Extroverted'], value='Neutral',label_visibility='collapsed')
ha = st.select_slider('Hostile vs Agreeable', options=['Very Hostile', 'Hostile','Neutral','Agreeable','Very Agreeable'], value='Neutral',label_visibility='collapsed')
sn = st.select_slider('Stable vs Neurotic', options=['Very Stable', 'Stable','Neutral','Neurotic','Very Neurotic'], value='Neutral',label_visibility='collapsed')

txt = st.text_area("Input Article")
st.markdown(
    '<div style="text-align: right;">'
    '<a href="https://mayur11235.github.io/industrypersona/sample_articles.html" target="_blank" title="Explore Sample Articles" style="font-family: Streamlit; color: #666; text-decoration: none;">Sample Articles 📰</a>'
    '</div>',
    unsafe_allow_html=True
)

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def get_feedback_score(question_key, default=-1):
    feedback_mapping = {"👍": 1, "👎": 0}
    if question_key in st.session_state and st.session_state[question_key] is not None and 'score' in st.session_state[question_key]:
        return feedback_mapping.get(st.session_state[question_key]['score'], default)
    return default

def set_state(i):
    st.session_state.stage = i
    if i==0:
        clear_session_state()

def validate_and_set_state(i,txt):   
    if (len(txt)>49):
        set_state(i)
        if "errormsg" in st.session_state:
            del st.session_state["errormsg"]
    else:        
        st.session_state.errormsg="Input Article should be atleast 50 characters."
        
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def click_button():
    st.session_state.clicked = True

def display_original_query():
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("GPT Output",value=st.session_state.orig_query['orig_chatgpt'],key=uuid.uuid4(),height=600)
        with col2:
            st.text_area("RAG Output",value=st.session_state.orig_query['orig_rag'],key=uuid.uuid4(), height=600)
        with st.expander("RAG Context"):
            st.text_area("RAG_Context",value=st.session_state.orig_query['orig_context'],key=uuid.uuid4(), height=600,label_visibility='hidden',disabled=True)

def display_chat():
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if isinstance(message["content"],str):
                st.markdown(message["content"])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("GPT Output",value=message['content']['chatgpt'], key=uuid.uuid4(), height=600)
                with col2:
                    st.text_area("RAG Output",value=message['content']['rag'], key=uuid.uuid4(),height=600)
                with st.expander("RAG Context"):
                    st.text_area("RAG_Context",value=message['content']['context'], key=uuid.uuid4() , height=600,label_visibility='hidden',disabled=True)

####################### Initial Screen set  ###################
prompt = []

if role_val != 'All':
    if ind_val != 'All':
        if comp_val != 'All':
            prompt.append(f'Adopt the persona of the {role_val} of {comp_val} in the {ind_val} industry.')
        else:
            prompt.append(f'Adopt the persona of the {role_val} of a company in the {ind_val} industry.')
    else:
        prompt.append(f'Adopt the persona of the {role_val} of a company.')
else:
    prompt.append(f'Adopt the persona of a C-Suite executive of a company.')

chatgpt_prompt = prompt.copy()

ind_val_short = ind_val.split()[0].split(',')[0]
persona_txt_file = os.path.join(script_path,"resources","prompts",f'{role_val}_{ind_val_short}.txt')
persona_QA_file = os.path.join(script_path,"resources","qa",f'{role_val}_{ind_val_short}_QA.txt')
if os.path.exists(persona_txt_file):
    with open(persona_txt_file, 'r') as file:
        persona_lines = file.readlines()
    prompt.extend(persona_lines)


non_neutral_vars = [p for p in [co,sc,ie,ha,sn] if p != "Neutral"]

if len(non_neutral_vars) == 0:
    personality_statement = ""
elif len(non_neutral_vars) == 1:
    personality_statement = non_neutral_vars[0]
elif len(non_neutral_vars) == 2:
    personality_statement = ' and '.join(non_neutral_vars)
else:
    personality_statement = ', '.join(non_neutral_vars[:-1]) + ', and ' + non_neutral_vars[-1]

if personality_statement != "":

    prompt.append(f"""Adjust the persona's responses to embody a {personality_statement.lower()} personality.""")

if os.path.exists(persona_QA_file):
    with open(persona_QA_file, 'r') as file:
        QA_lines = file.readlines()
    prompt.extend(QA_lines)

prompt_txt_box = st.text_area("Personality Prompt",value="\n".join(prompt), height=300)

if txt:
    prompt.append('Use the following article to answer any questions from the perspective of the persona: \n\n'+limit_to_n_words(txt,1000))
    chatgpt_prompt.append('Use the following article to answer any questions from the perspective of the persona: \n\n'+limit_to_n_words(txt,1000))

chatgpt_prompt_txt = "\n".join(chatgpt_prompt)
prompt_txt = "\n".join(prompt)


if st.session_state.stage==0:  
    prebuilt_queries = st.radio(
        "Basic Queries",
        ["Custom Query","Short Summary", "Main Ideas", "Best Possible Summary", "Generate Questions"],
        captions = ["Ask anything you like","Provides a concise summary of the Input Article", "Medium length summary providing main ideas", "Long length summary of Input Article", "Generates 5 questions based on Input Article"])

    if prebuilt_queries=="Short Summary":
        st.session_state.query_text_area = "Provide a concise summary of the article in 50 words or less"
    elif prebuilt_queries=="Main Ideas":
        st.session_state.query_text_area = "Provide a summary of the article's main ideas in 100 words or less"
    elif prebuilt_queries=="Best Possible Summary":
        st.session_state.query_text_area = "Provide the best possible summary of the article"
    elif prebuilt_queries=="Generate Questions":
        st.session_state.query_text_area = "Generate 5 questions to ask the persona based on the article"
    else:
        st.session_state.query_text_area = st.text_area("Custom Query")



if "chatgpt_messages" not in st.session_state:
    st.session_state.chatgpt_messages = []
if st.session_state.stage ==0:
    colb, colm = st.columns(2)
    with colb:
        gp = st.button("Query Persona", on_click=validate_and_set_state,args=[1,txt])        
    with colm:
        st.markdown(f'<p style="color: red;">{st.session_state.get("errormsg", "")}</p>', unsafe_allow_html=True)

####################### Initial Query Submission ###################
if st.session_state.stage==1:    
    if 'rag' not in st.session_state:
            if comp_val == "All":
                metadata_list = None
            else:
                company_code = comp_val.split(' -')[0]
                metadata_list = [{'company': company_code},{'publisher':'CFO'}, {'publisher':'Forbes'}, {'publisher':'Tipalti'}, {'publisher':'vcfo'}, {'publisher':'WSJ'}]
            if 'RAG_MODE' in os.environ and os.environ['RAG_MODE'] == 'advanced':    
                rag_obj=RAGAMGPT(get_vector_index(),k=4,top=3,filters=metadata_list)   
                logging.info("<<Running in Advanced Rag Mode>>")
            else:
                rag_obj=RAGGPT(get_vector_index(),k=4,top=3 ,filters=metadata_list)
            st.session_state['rag'] = rag_obj
    if 'chatgpt' not in st.session_state:
            st.session_state['chatgpt']=CHATGPT()
        
    #st.session_state.chatgpt_messages.append({"role": "system", "content":prompt_txt + st.session_state.get("query_text_area", "")})
    st.session_state.chatgpt_messages.append({"role": "system", "content":chatgpt_prompt_txt + st.session_state.get("query_text_area", "")})
    with st.spinner('Waiting for response from LLM API'):
        async def get_response_async():
            task_orig_chat = asyncio.create_task(st.session_state.chatgpt.get_response_async(st.session_state.chatgpt_messages)) 
            task_orig_rag = asyncio.create_task(st.session_state.rag.get_response_async(prompt_txt,st.session_state.get("query_text_area", "")))
            await asyncio.gather(task_orig_chat,task_orig_rag)
            orig_chat = task_orig_chat.result()    
            orig_rag_response , orig_rag_context = task_orig_rag.result()
            st.session_state.chatgpt_messages.append({'role': 'assistant', 'content': orig_chat})   
            st.session_state.orig_query = {'orig_chatgpt':orig_chat, 'orig_rag':orig_rag_response,'orig_context':orig_rag_context}
        asyncio.run(get_response_async()) 

    st.session_state.stage = 2

####################### Follow up Dialogue ###################
if st.session_state.stage == 2:
    display_original_query()    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    display_chat()
    
    if re_prompt := st.chat_input("Message persona"):
        #Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(re_prompt)
            #Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": re_prompt})
            st.session_state.chatgpt_messages.append({"role": "user", "content": re_prompt})
            chatgpt_response=""
            rag_response=""
            rag_context=""
            reply={}
            with st.spinner('Waiting for response from LLM API'):
                async def get_response_async2():
                    global chatgpt_response 
                    global rag_response 
                    global reply
                    global rag_context
                    task_chatgpt_response = asyncio.create_task(st.session_state.chatgpt.get_response_async(st.session_state.chatgpt_messages))
                    task_rag_response = asyncio.create_task(st.session_state.rag.get_response_async(re_prompt))
                    await asyncio.gather(task_chatgpt_response,task_rag_response)
                    chatgpt_response = task_chatgpt_response.result()    
                    rag_response,rag_context = task_rag_response.result()
                    st.session_state.chatgpt_messages.append({'role': 'assistant', 'content': chatgpt_response})    
                    reply = {'chatgpt':chatgpt_response,'rag':rag_response,'context':rag_context}
                asyncio.run(get_response_async2())    
                
        with st.chat_message('assistant'):
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("GPT Output",value=chatgpt_response, height=600)
            with col2:
                st.text_area("RAG Output",value=rag_response, height=600)
            with st.expander("RAG Context"):
                st.text_area("RAG_Context",value=rag_context, height=600,label_visibility='hidden',disabled=True)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    
    col1,col2 = st.columns([.2,1])
    with col1:
        st.button('Start Over', on_click=set_state, args=[0])
    with col2:
        st.button('Provide Feedback', on_click=set_state, args=[3])        

####################### Submit Feedback and Start over ###################
if st.session_state.stage == 3:
    display_original_query()
    display_chat()
    
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    chat_df=pd.DataFrame(st.session_state.rag.get_query_log())
    csv = convert_df(chat_df)

    with st.expander("Feedback", expanded=True):
        user_rating = st.slider('Give the RAG a score',0, 10,5)
        col_1_test, col_2_test = st.columns([.2,1])
        with col_1_test:
            q1_feedback = streamlit_feedback(feedback_type="thumbs",key='q1')
            q2_feedback = streamlit_feedback(feedback_type="thumbs",key='q2')
            q3_feedback = streamlit_feedback(feedback_type="thumbs",key='q3')
            q4_feedback = streamlit_feedback(feedback_type="thumbs",key='q4')
        with col_2_test:
            st.markdown("""
            Was the Persona output better than the GPT Output?
            <div style='margin-bottom: 26px;'></div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            Was the context retrieved relevant?
            <div style='margin-bottom: 26px;'></div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            Was the Persona consistent throughout the interaction?
            <div style='margin-bottom: 26px;'></div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            Was the output useful?
            <div style='margin-bottom: 26px;'></div>
            """, unsafe_allow_html=True)

        user_feedback = st.text_area("Comments",key="Comments")
        tqmsg=""
        colb, colm = st.columns(2)
        with colb:
            if st.button('Submit Feedback', on_click=set_state, args=[3]):
                log_chat_and_feedback(chat_df,
                                      user_id,
                                      user_rating,
                                      get_feedback_score('q1'),
                                      get_feedback_score('q2'),
                                      get_feedback_score('q3'),
                                      get_feedback_score('q4'),
                                      user_feedback)
                tqmsg="Thank you!"
        with colm:
            st.markdown(f'<p style="color: red;">{tqmsg}</p>', unsafe_allow_html=True)
        st.download_button(
            label="Download Chat as CSV",
            data=csv,
            file_name='conversation.csv',
            mime='text/csv',
        )
    st.button('Start Over', on_click=set_state, args=[0])
    st.chat_input("Messaging Disabled",disabled=True)
    