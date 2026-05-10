# 📡 Nigerian Telecom AI Assistant (RAG Pipeline)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-red.svg)
![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)

An AI-powered support agent designed to bridge the information gap during the NCC Harmonized USSD transition in Nigeria. Built with a Retrieval-Augmented Generation (RAG) architecture.

### 🔗 [🚀 View Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Joannes10/Telecom-chatbot)

---

## 📖 Project Overview
This project solves a real-world problem: helping 100M+ Nigerian subscribers navigate the 2024/2025 transition to unified USSD codes. By using RAG, the model provides grounded, factual answers based on official NCC standards, significantly reducing LLM hallucinations.

## 🏗️ Technical Architecture

The system follows a standard RAG (Retrieval-Augmented Generation) workflow:
1. **Document Loading:** Ingests `faq.txt` using LangChain's `TextLoader`.
2. **Chunking:** Splits text via `RecursiveCharacterTextSplitter` (700 char chunks).
3. **Embedding:** Generates 384-dimensional vectors using `all-MiniLM-L6-v2`.
4. **Vector Store:** Stores embeddings in **FAISS** for efficient similarity search.
5. **Inference:** Retrieves top-k contexts and passes them to **Llama-3.2-3B** via Hugging Face Inference API.

## 🛠️ Installation & Local Setup
To run this project locally for development:

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Johnpaul10j/Telecom-AI-Assistant-NG.](https://github.com/Johnpaul10j/Telecom-AI-Assistant-NG.)
   cd Telecom-AI-Assistant-NG.
   pip install -r requirements.txt
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   python app.py
   ```

## 🧠 Key Challenges & Solutions
**Version Mismatch:** Solved a critical ModuleNotFoundError by implementing a MagicMock monkey-patch to maintain compatibility between gradio 5.0 and huggingface-hub.

**Response Precision:** Engineered custom prompt templates to force the model to stay within the 1-3 sentence "Support Agent" persona.

## 👤 Author
**Umeh Johnpaul**

**LinkedIn:** www.linkedin.com/in/johnpaul-umeh-524071289   
   
   
