import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="🦙💬 Codearch.io - Mistral Evaluator")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Codearch.io Mistral Evaluator')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models & parameters')
    selected_model = st.sidebar.selectbox('Choose a Mistral model', ['mixtral-8x7b-instruct-v0.1','dolphin-2.2.1-mistral-7b'], key='selected_model')
    max_token = 16384
    if selected_model == 'mixtral-8x7b-instruct-v0.1':
        llm = 'mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21'
    elif selected_model == 'dolphin-2.2.1-mistral-7b':
        llm = 'lucataco/dolphin-2.2.1-mistral-7b:0521a0090543fea1a687a871870e8f475d6581a3e6e284e32a2579cfb4433ecf'
    top_k = st.sidebar.slider('top_k', min_value=0, max_value=100, value=50, step=1)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.6, step=0.01)
    max_new_tokens = st.sidebar.slider('max_new_tokens', min_value=512, max_value=max_token, value=max_token, step=8)
    # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    # st.session_state.messages = [{}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    # st.session_state.messages = [{}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_response(prompt_input):
    # string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "top_k":top_k, "temperature":temperature, "top_p":top_p, 
                                  "max_new_tokens":max_new_tokens, "prompt_template": "<s>[INST] {prompt} [/INST] ",
                                  "presence_penalty": 0, "frequency_penalty": 0})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
