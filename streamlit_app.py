import streamlit as st
import replicate
import os
import boto3
import json 

REPLICATE = "replicate"
AWS = "AWS"

# newline, bold, unbold = "\n", "\033[1m", "\033[0m"
os.environ["AWS_DEFAULT_REGION"]='us-east-2'

# App title
st.set_page_config(page_title="🦙💬 Deep Insight - LLM Evaluator")

def manageSecrets(provider):
    if provider == REPLICATE:
        if "REPLICATE_API_TOKEN" in st.secrets:
            st.success('API REPLICATE_API_TOKEN already provided!', icon='✅')
            replicate_api = st.secrets["REPLICATE_API_TOKEN"]
        else:
            replicate_api = st.text_input(f'Enter REPLICATE_API_TOKEN API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ["REPLICATE_API_TOKEN"] = replicate_api
    elif provider == AWS:
        if "AWS_ACCESS_KEY_ID" in st.secrets and "AWS_SECRET_ACCESS_KEY" in st.secrets:
            st.success('AWS API credentials already provided!', icon='✅')
            aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
            aws_access_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
        else:
            aws_access_key = st.text_input(f'Enter AWS_ACCESS_KEY_ID token:', type='password')
            aws_access_secret = st.text_input(f'Enter AWS_SECRET_ACCESS_KEY token:', type='password')  
            if not len(aws_access_key) !=0 or not len(aws_access_secret) != 0:
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_access_secret
    return True

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Deep Insight LLM Evaluator')
    provider = st.radio(
        "Select provider",
        ["replicate","AWS"],
        index=0,
    )
    provider_api = manageSecrets(provider)

    st.subheader('Models & parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['deep-insight-v0.1','mixtral-8x7b-instruct-v0.1','dolphin-2.2.1-mistral-7b','aws-sagemaker-mixtral'], key='selected_model')
    max_token = 16384
    if selected_model == 'deep-insight-v0.1':
        llm = 'mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21'
    elif selected_model == 'mixtral-8x7b-instruct-v0.1':
        llm = 'mistralai/mixtral-8x7b-instruct-v0.1'
    elif selected_model == 'dolphin-2.2.1-mistral-7b':
        llm = 'lucataco/dolphin-2.2.1-mistral-7b:0521a0090543fea1a687a871870e8f475d6581a3e6e284e32a2579cfb4433ecf'
    elif selected_model == 'aws-sagemaker-mixtral':
        llm = 'jumpstart-dft-hf-llm-mixtral-8x7b-20240214-024901'
    top_k = st.sidebar.slider('top_k', min_value=0, max_value=100, value=50, step=1)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.6, step=0.01)
    max_new_tokens = st.sidebar.slider('max_new_tokens', min_value=512, max_value=max_token, value=max_token, step=8)

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

def construct_prompt():
    # string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    return f"{string_dialogue} \nAssistant"

def generate_response():
    output = replicate.run(llm, 
                           input={"prompt": construct_prompt(),
                                  "top_k":top_k, "temperature":temperature, "top_p":top_p, 
                                  "max_new_tokens":max_new_tokens, "prompt_template": "<s>[INST] {prompt} [/INST] ",
                                  "presence_penalty": 0, "frequency_penalty": 0})
    return output

def generate_aws_response():
    payload = {
        "inputs": f"<s>[INST] {construct_prompt()} [/INST] ",
        "parameters": {
            "temperature": temperature, 
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "presence_penalty": 0,
            "frequency_penalty": 0
        },
    }
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=llm, ContentType="application/json", Body=json.dumps(payload).encode("utf-8")
    )
    model_predictions = json.loads(response["Body"].read())
    generated_text = model_predictions[0]["generated_text"]
    return generated_text

# User-provided prompt
if prompt := st.chat_input(disabled=not provider_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if provider == REPLICATE:
                response = generate_response()
            elif provider == AWS:
                response = generate_aws_response()
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)