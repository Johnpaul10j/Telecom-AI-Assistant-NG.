import sys
from unittest.mock import MagicMock

# Trick Gradio into thinking HfFolder exists in huggingface_hub
import huggingface_hub
mock_hf = MagicMock()
huggingface_hub.HfFolder = mock_hf



# Rest of your code follows...
import gradio as gr
import os


os.environ["GRADIO_NO_AUDIO"] = "1"




from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
#from langchain.prompts import PromptTemplate

# ... rest of your code remains the same

# ====================== CONFIG ======================
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(token=hf_token)

# ====================== LOAD FAQ ======================
loader = TextLoader("faq.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ====================== PROMPT ======================
prompt_template = """You are a friendly and professional Telecom Customer Support Assistant.

Answer naturally, clearly and concisely (1-3 sentences max).
Never use "Q:" or "A:" format.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ====================== CHAT FUNCTION ======================
def chatbot(message, history):
    try:
        retrieved = retriever.invoke(message)
        context = "\n\n".join([doc.page_content for doc in retrieved])

        final_prompt = PROMPT.format(context=context, question=message)

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful Nigerian telecom support agent. Be friendly and concise."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=400,
            temperature=0.32,
        )

        answer = response.choices[0].message.content.strip()
        
        # Clean
        for prefix in ["Answer:", "Assistant:"]:
            if prefix in answer:
                answer = answer.split(prefix)[-1].strip()

        # Add citations
        sources = "\n\n**Sources:** " + ", ".join([f"Doc {i+1}" for i in range(len(retrieved))])
        
        return answer + sources

    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, please try again in a moment."

# ====================== CUSTOM CSS ======================
custom_css = """
.gradio-container { max-width: 1200px !important; margin: auto; }
"""

# ====================== UI ======================
#with gr.Blocks(title="Telecom AI Assistant", css=custom_css) as demo:
 #   gr.Markdown("""
    # 📡 Telecom AI Assistant
    
  #  **Smart • Fast • Reliable**  
   # AI-Powered Support for MTN • Glo • Airtel • 9mobile
    #""")
    
    #gr.ChatInterface(
     #   fn=chatbot,
      #  chatbot=gr.Chatbot(height=720),
       # textbox=gr.Textbox(placeholder="Ask any telecom question..."),
        #examples=[
         #   ["How can I check my data balance?"],
          #  ["Why is my network slow?"],
           # ["What should I do if my SIM is lost?"],
            #["Tell me about current data plans"]
        #],
    #)
    
    #gr.Markdown("""
    #---
    #⚡ Built with LangChain • FAISS • Llama-3.2  
    #**Portfolio Project** | With Source Citations
    #""")

# Launch (Correct way for Gradio 6.0+)
#demo.launch(
 #   server_name="0.0.0.0",
  #  server_port=int(os.environ.get("PORT", 7860)),
   # theme=gr.themes.Soft()
#)

# ====================== UI ======================
# MOVED: The theme is now here inside gr.Blocks()
with gr.Blocks(title="Telecom AI Assistant", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📡 Telecom AI Assistant
    
    **Smart • Fast • Reliable**  
    AI-Powered Support for MTN • Glo • Airtel • 9mobile
    """)
    
    gr.ChatInterface(
        fn=chatbot,
        type="messages", #to avoid the deprecation warning
        chatbot=gr.Chatbot(height=720, type="messages"), 
        textbox=gr.Textbox(placeholder="Ask any telecom question..."),
        examples=[
            ["How can I check my data balance?"],
            ["Why is my network slow?"],
            ["What should I do if my SIM is lost?"],
            ["Tell me about current data plans"]
        ],
    )
    
    gr.Markdown("""
    ---
    ⚡ Built with LangChain • FAISS • Llama-3.2  
    **Portfolio Project** | With Source Citations
    """)

# Launch (REMOVED: theme argument from here)
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)