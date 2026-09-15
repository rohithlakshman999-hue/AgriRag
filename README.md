🌾 AgriRAG — Evidence-Grounded Irrigation Knowledge Assistant

AgriRAG is a retrieval-augmented agricultural assistant designed to help farmers find reliable irrigation and crop water-management information from indexed agricultural documents.

The system first understands the farmer's question and context, retrieves relevant evidence from the document collection, combines semantic and keyword search, reranks the results, and then generates a concise answer grounded in the retrieved evidence.

✨ What AgriRAG Does

A farmer can ask questions such as:

My tomato crop is in the flowering stage and I have limited water. What irrigation practices should I follow to save water?

AgriRAG identifies useful context such as:

Crop

Growth stage

Irrigation intent

Irrigation method, when specified

Water availability

It then retrieves the most relevant agricultural passages and presents an evidence-grounded answer with document and page references.

🚀 Key Features

🧠 Farmer Context Understanding

Extracts useful information from a natural-language question, such as crop, growth stage, water availability and user intent.

🔎 Semantic Search

Uses sentence embeddings and FAISS to find passages that are semantically related to the farmer's question, even when the exact words differ.

🔤 Keyword Search

Uses BM25 to improve retrieval when important agricultural terms or exact keywords appear in the question.

🔀 Hybrid Retrieval

Combines semantic and keyword retrieval so the system can use both meaning and exact terminology.

🏆 Evidence Reranking

Reranks retrieved passages so the most relevant evidence is presented first.

🤖 Evidence-Grounded Answer Generation

The language model is instructed to answer using the supplied agricultural evidence and avoid unsupported recommendations.

📚 Source & Evidence Display

The application displays supporting document names, pages, sections and the retrieved evidence used for the answer.

⚠️ Knowledge-Gap Handling

When the available documents do not contain enough information, the system is designed to say that the information is insufficient instead of inventing facts.

🌐 English & Tamil Support

The answer-generation layer supports farmer-friendly English and simple Tamil responses.

🏗️ System Workflow

                         👨‍🌾 Farmer
                             │
                             ▼
                     Natural-language question
                             │
                             ▼
                    query_context.py
                             │
                Extract crop / stage / intent /
                  water availability / method
                             │
                             ▼
                    section_router.py
                             │
                             ▼
                    Hybrid Retrieval
                       /            \
                      /              \
                  FAISS             BM25
               semantic search   keyword search
                      \              /
                       \            /
                        ▼          ▼
                         Candidates
                             │
                             ▼
                    rerank_search.py
                             │
                             ▼
                       Best evidence
                             │
                             ▼
                         answer.py
                             │
                             ▼
                   Grounded farmer answer
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 Sources           Evidence
                    │                 │
                    └────────┬────────┘
                             ▼
                           app.py
                             │
                             ▼
                     Streamlit interface

📁 Project Structure

AgriRag/
│
├── app.py                 # Streamlit application and UI
├── answer.py              # Evidence-grounded LLM answer generation
├── rag.py                 # Main RAG orchestration pipeline
├── search.py              # Semantic search using FAISS
├── hybrid_search.py       # BM25 + semantic retrieval fusion
├── rerank_search.py       # Evidence reranking
├── query_context.py       # Farmer context and intent extraction
├── section_router.py      # Routes questions to relevant sections
├── ingest.py              # PDF extraction and chunk creation
├── build_index.py         # Builds FAISS and BM25 indexes
├── check_sections.py      # Section inspection / validation utility
│
├── data/
│   ├── raw/               # Original agricultural PDF documents
│   └── processed/         # Generated searchable data
│       ├── chunks.pkl
│       ├── faiss.index
│       └── bm25.pkl
│
├── requirements.txt       # Python dependencies
├── .env                   # Local secrets/configuration (do not commit)
└── README.md              # Project documentation

🛠️ Technology Stack

Component

Technology

Language

Python

User Interface

Streamlit

PDF Processing

PyMuPDF

Embeddings

Sentence Transformers

Semantic Retrieval

FAISS

Keyword Retrieval

BM25

LLM Inference

Hugging Face Inference

Environment Management

python-dotenv

📦 Installation

1. Clone the repository

git clone <your-repository-url>
cd AgriRag

2. Create a virtual environment

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

🔐 Environment Variables

Create a .env file in the project root:

HF_TOKEN=your_huggingface_token
HF_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

Never commit .env or expose your Hugging Face token in GitHub.

📚 Add Agricultural Documents

Place the agricultural PDFs you want the assistant to use inside:

data/raw/

Recommended document types include:

Irrigation guides

Crop manuals

Water-management documents

Micro-irrigation guidelines

Official agricultural manuals and guidelines

For trustworthy answers, prefer authoritative or organizer-provided sources.

⚙️ Build the Knowledge Index

After adding or changing the PDFs, rebuild the processed data.

Step 1 — Extract and chunk PDFs

python ingest.py

This creates:

data/processed/chunks.pkl

Step 2 — Build FAISS and BM25 indexes

python build_index.py

This creates/updates:

data/processed/faiss.index
data/processed/bm25.pkl

Important Embedding Rule

The same embedding model must be used both when building the FAISS index and when searching it.

This project uses:

sentence-transformers/all-MiniLM-L6-v2

This prevents FAISS dimension-mismatch errors such as:

AssertionError: assert d == self.d

▶️ Run the Application

Start the Streamlit application with:

streamlit run app.py

The app will open in your browser, usually at:

http://localhost:8501

🧪 Example Questions

Example 1 — Context-aware

My tomato crop is in the flowering stage and I have limited water. What irrigation practices should I follow to save water?

Example 2 — Water conservation

How can farmers save water using micro-irrigation?

Example 3 — Irrigation scheduling

What are the principles of irrigation scheduling?

Example 4 — Source-grounded question

What does the irrigation guidance say about efficient water use?

Example 5 — Knowledge-gap test

What exact number of litres of water should I give my tomato crop every day during flowering?

For the last type of question, the system should avoid inventing a number when that exact value is not supported by the indexed documents.

🧠 Example End-to-End Flow

Suppose the farmer asks:

My tomato crop is in the flowering stage and water is limited. What irrigation practices should I follow?

AgriRAG processes it roughly as follows:

Detects Tomato as the crop.

Detects Flowering as the growth stage.

Detects Low / Limited water as the water condition.

Identifies the question as related to irrigation / water saving.

Builds a context-aware retrieval query.

Searches using both semantic similarity and keywords.

Reranks the candidate passages.

Sends the strongest evidence to the language model.

Generates a farmer-friendly answer.

Displays supporting documents, pages and retrieved evidence.

🎯 Design Principles

Evidence First

Answers should be grounded in the agricultural documents available to the system.

No Invented Exact Values

The assistant should not invent irrigation intervals, water quantities, timings, frequencies or percentages when they are not documented.

Partial Answers Are Allowed

If some parts of a farmer's question are supported and another requested detail is missing, the system should answer the supported parts and clearly identify the missing information.

Explainability

Users should be able to inspect the supporting evidence behind an answer.

🏆 Hackathon Value

AgriRAG is designed to demonstrate more than a basic chatbot. Its main technical components are:

Context-aware retrieval

Hybrid semantic + keyword search

Evidence reranking

Retrieval-augmented generation

Source-aware answers

Knowledge-gap handling

Multilingual farmer interaction

🔮 Future Extensions

Potential future additions include:

Voice-based farmer queries

More regional languages

Weather-aware irrigation context

Soil-moisture sensor integration

Crop growth-stage intelligence

Document version comparison

Evidence/claim verification

Retrieval and groundedness evaluation dashboard

Offline or low-connectivity support

⚠️ Disclaimer

AgriRAG provides information based on the agricultural documents available in its knowledge base. It should not invent or assume missing information. Farmers should use authoritative agricultural guidance and local expert advice for decisions that require field-specific assessment.

👨‍💻 Running in Development

Whenever the source documents are changed:

python ingest.py
python build_index.py
streamlit run app.py

If only Python application/UI code changes, rebuilding the document indexes is generally not necessary unless the underlying documents or embedding model have changed.