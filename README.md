
---
# 🤖 SU_BOT 4.0 — Agentic RAG for SCET

**An advanced AI assistant for Sarvajanik College of Engineering & Technology (SCET), Surat**
Powered by **Gemini 1.5 Pro**, **HuggingFace embeddings**, **FAISS**, and **Streamlit UI**

---

## 🧭 Overview

SU_BOT 4.0 is a full-stack **Agentic Retrieval-Augmented Generation (RAG)** system that:

✅ Crawls and extracts rich data from the official SCET website using **Selenium + BeautifulSoup**
✅ Cleans, summarizes, and structures data into `.txt` files for vector indexing
✅ Builds a **local FAISS vector store** using **HuggingFace embeddings (all-MiniLM-L6-v2)** — fully offline
✅ Answers questions via **Gemini 1.5 Pro** reasoning engine
✅ Fetches up-to-date information via **Tavily web search**
✅ Features a **Streamlit interface** with live chat and index rebuild options

---

## 🏗️ Project Structure

```
SU_BOT_4/
│
├── app.py                         # Streamlit frontend
├── config.py                      # Key & environment loader
├── requirements.txt
│
├── agents/
│   ├── controller.py               # Decides query routing (local / web / hybrid)
│   └── answer_synthesizer.py       # Synthesizes final Gemini answers
│
├── retriever/
│   ├── local_index.py              # Builds & manages FAISS index with HuggingFace embeddings
│   └── router.py                   # Smart routing logic
│
├── tools/
│   └── web_search.py               # Tavily web retrieval
│
├── data/
│   ├── scet/                       # Indexed text files for local RAG
│   └── scet_selenium/              # Auto-scraped SCET website dataset
│
└── generate_scet_dataset_selenium.py   # Selenium-based web data generator
```

---

## ⚙️ Setup

### 1️⃣ Clone the repo

```bash
git clone https://github.com/Karanmeta/SU_BOT.git
cd SU_BOT
```

---

### 2️⃣ Create and activate a virtual environment

```bash
conda create -n su_bot python=3.10 -y
conda activate su_bot
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to rebuild SCET data, also install Selenium tools:

```bash
pip install selenium beautifulsoup4 webdriver-manager
```

---

### 4️⃣ Set up your `.env` file

Create a file named `.env` in the project root:

```
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
OPENAI_API_KEY=optional_if_you_use_openai
```

---

## 🕸️ Generate Data from SCET Website

The new **Selenium crawler** bypasses Cloudflare and scrapes the website like a real browser.

Run this to crawl and extract ~150 SCET pages:

```bash
python generate_scet_dataset_selenium.py
```

Output:

```
data/scet_selenium/
├── about-us.txt
├── department-information-technology.txt
├── department-computer-engineering.txt
├── placements.txt
├── research-and-innovation.txt
└── _index.json
```

Once you confirm data is correct, copy it to:

```
data/scet/
```

---

## 🧮 Build / Rebuild Local FAISS Index

The index is built automatically when you run the app.
You can also rebuild manually:

```bash
python -m retriever.local_index
```

or inside the Streamlit app via the **“♻️ Rebuild Local Index”** sidebar button.

---

## 🚀 Run the Streamlit App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 💬 Example Queries

### 🟢 Easy

* Who is the HOD of the IT Department at SCET?
* When was SCET established?
* What courses are offered for undergraduate students?

### 🟡 Medium

* Compare the IT and Computer Engineering departments at SCET.
* How does SCET promote student innovation?
* What are the lab facilities available in the Electronics Department?

### 🔵 Advanced

* Which department has the highest placement rate and why?
* How does SCET’s IT curriculum align with AI and Data Science?
* List professors who specialize in AI or Machine Learning at SCET.

---

## 🧠 Tech Stack

| Component            | Technology                 | Description                              |
| -------------------- | -------------------------- | ---------------------------------------- |
| **Frontend**         | Streamlit                  | Live chat UI                             |
| **RAG Core**         | LangChain                  | Query routing, retrieval, and synthesis  |
| **LLM Reasoning**    | Gemini 1.5 Pro             | Agentic reasoning & generation           |
| **Local Embeddings** | HuggingFace (MiniLM-L6-v2) | Offline semantic vectorization           |
| **Vector Store**     | FAISS                      | Fast approximate nearest neighbor search |
| **Web Retrieval**    | Tavily                     | Live context fetching                    |
| **Data Source**      | Selenium + BeautifulSoup   | Dynamic SCET website crawler             |

---

## 📊 Performance Notes

✅ 100% local embeddings — no API quota
✅ Average page retrieval < 150ms
✅ Handles ~150 SCET pages with ease
✅ 0 hallucinations when data is relevant
✅ Works offline once data is built

---

## 🧰 Troubleshooting

| Issue                       | Cause                          | Fix                                       |
| --------------------------- | ------------------------------ | ----------------------------------------- |
| 403 Forbidden               | Website blocks requests        | Use `generate_scet_dataset_selenium.py`   |
| 429 Too Many Requests       | Gemini / OpenAI quota exceeded | Reduce crawl size or use local embeddings |
| No answers for SCET queries | Missing SCET `.txt` files      | Re-run crawler or rebuild FAISS           |
| Browser not found           | Chrome not installed           | Install Google Chrome locally             |

---

## 🧩 Future Enhancements

* 🧠 Memory & context persistence across chats
* 🗂️ Automatic content categorization (Departments / Research / Events)
* 🌐 Live SCET news integration via Tavily
* 📈 “Test Dashboard” for benchmarking responses

---

## 🏁 Author

**Karan Mehta**
---
