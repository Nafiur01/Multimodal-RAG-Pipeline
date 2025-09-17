# 🧠 Multimodal Retrieval Using Unstructured Data

## 📌 Overview
This project demonstrates a **Multimodal Retrieval-Augmented Generation (RAG) pipeline** for **PDF documents** containing **text, tables, and images**. It uses **Unstructured.io** to parse documents, **LangChain** to manage retrieval and generation, and integrates **Chroma** as a vector database. The pipeline supports advanced querying over heterogeneous data types to create context-aware responses.

## ✨ Features
- 📄 **PDF Partitioning** – Extracts text, tables, and images from PDFs using `unstructured.partition.pdf`.
- 🖼 **Multimodal Support** – Separates and processes text, table, and image elements.
- 🧠 **Vector Store Integration** – Stores embeddings in **Chroma** for efficient retrieval.
- ⚡ **LangChain Components** – Uses `MultiVectorRetriever` and HuggingFace embeddings.
- 🤖 **LLM Querying** – Employs **Groq** or **OpenAI** models via LangChain for generating answers.
- 🔗 **Flexible Prompting** – Utilizes `ChatPromptTemplate` and `StrOutputParser`.
- 🧰 **Interactive Visualization** – Displays extracted images directly in notebooks.

## 🧭 Workflow
```
PDF Document 
    │
    ▼
Partition PDF → Extract [Text, Tables, Images]
    │
    ▼
Generate Embeddings (HuggingFace)
    │
    ▼
Store in Chroma Vector DB
    │
    ▼
Retrieve Relevant Chunks (MultiVectorRetriever)
    │
    ▼
LLM (ChatGroq / OpenAI) → Context-Aware Answer
```

## 🚀 Getting Started

### 1️⃣ Prerequisites
- Python 3.9+
- Jupyter Notebook or Google Colab
- API Keys for **Groq** or **OpenAI**

### 2️⃣ Installation
Install required dependencies:
```bash
pip install unstructured langchain langchain-groq langchain-huggingface chromadb openai
```

### 3️⃣ Usage
1. Open the notebook:
   ```bash
   jupyter notebook Multimodal_Retrieval_Unstructured.ipynb
   ```
2. Upload a PDF file in the notebook or Google Colab.
3. Set your API key:
   ```python
   from google.colab import userdata
   os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
   ```
4. Run the notebook cells to:
   - Partition the PDF into text, tables, and images.
   - Embed and store the elements in Chroma.
   - Query the LLM for context-rich answers.

## 📂 Directory Structure
```
├── Multimodal_Retrieval_Unstructured.ipynb   # Main pipeline
├── requirements.txt                          # (Optional) List of dependencies
└── data/                                     # (Optional) PDF files and samples
```

## 🧰 Technologies Used
- **[Unstructured.io](https://unstructured.io/)** – PDF parsing and element extraction
- **[LangChain](https://www.langchain.com/)** – Framework for RAG pipeline
- **[Chroma](https://www.trychroma.com/)** – Vector database for embeddings
- **[Hugging Face Transformers](https://huggingface.co/)** – Embedding models
- **[Groq LLM](https://groq.com/)** or **[OpenAI](https://openai.com/)** – Large language models

## 🧪 Example Query
```python
query = "Summarize the key points from this PDF."
result = retriever.get_relevant_documents(query)
print(result)
```

## 🧭 Roadmap
- [ ] Integrate **table-to-text** parsing for richer table insights.
- [ ] Add **image captioning** for image-based retrieval.
- [ ] Deploy as a **FastAPI** or **Streamlit** web app.

## 🙌 Acknowledgements
Special thanks to **LangChain**, **Unstructured**, **Chroma**, and **Hugging Face** for their powerful open-source tools.
