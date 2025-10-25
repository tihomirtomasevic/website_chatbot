---
title: FAQ Chatbot
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# FAQ Chatbot for Company Website

A RAG-based chatbot built with Gradio, LangChain, and OpenAI to answer visitor questions from your company's FAQ and knowledge base. Designed for deployment on HuggingFace Spaces (free tier).

## Features

- Conversational AI powered by OpenAI GPT
- Semantic search over company knowledge base using local embeddings
- Session-based chat memory for follow-up questions
- Clean, customizable Gradio interface
- Ready for HuggingFace Spaces deployment

## Architecture

- **Frontend**: Gradio ChatInterface with session memory
- **RAG Pipeline**: LangChain with FAISS vector store
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) - runs locally, no API cost
- **LLM**: OpenAI GPT-3.5-turbo via API
- **Vector Store**: FAISS in-memory for fast retrieval

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- OpenAI API key

### 2. Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root with your OpenAI API key (you can use `env_template.txt` as a reference):

```
OPENAI_API_KEY=sk-your-actual-api-key-here
MODEL_NAME=gpt-3.5-turbo
```

**Note**: Copy `env_template.txt` to `.env` and fill in your actual API key.

### 4. Knowledge Base

Place your markdown files in the `data/knowledge_base/` directory. The chatbot will automatically load and index all `.md` files.

### 5. Run Locally

```bash
python app.py
```

The chatbot will be available at `http://localhost:7860`

## Customization

### Theme and Branding

Edit the theme settings in `app.py` (lines 124-133) to match your company colors:

```python
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
).set(
    button_primary_background_fill="#2563eb",  # Change to your brand color
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
)
```

### System Prompt

Modify the chatbot's behavior by editing the prompt in the `get_qa_prompt()` function (lines 85-106).

### Retrieval Settings

Adjust the number of relevant documents retrieved by changing `k` value in `create_conversation_chain()` (line 73):

```python
search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
```

## Deployment to HuggingFace Spaces

### Option 1: Using HuggingFace Interface

1. Create a new Space on HuggingFace (https://huggingface.co/spaces)
2. Select "Gradio" as the SDK
3. Upload all files from this project
4. Add your `OPENAI_API_KEY` as a secret in Space settings
5. The Space will automatically build and deploy

### Option 2: Using Git

```bash
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "Initial commit"
git push space main
```

### Important for HuggingFace Deployment

1. **Secrets**: Add `OPENAI_API_KEY` in Space Settings → Repository Secrets
2. **Memory**: Free tier has limited RAM. If you encounter issues, reduce chunk size or limit knowledge base size
3. **Cold starts**: First load may take 30-60 seconds while downloading the embedding model

## Project Structure

```
website_chatbot/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── env_template.txt            # Environment variables template
├── .env                        # Environment variables (create this, not tracked)
├── .gitignore                  # Git ignore file
├── README.md                   # This file
└── data/
    └── knowledge_base/         # Markdown files with company info
        ├── ai_topics.md
        ├── agentic_ai.md
        ├── clients_engagement.md
        ├── company_overview.md
        ├── contact_support.md
        ├── faq_general.md
        ├── founder.md
        ├── portfolio.md
        ├── pricing.md
        ├── services.md
        └── workflow.md
```

## Troubleshooting

**Issue**: "No markdown files found" error  
**Solution**: Ensure your `.md` files are in `data/knowledge_base/` directory

**Issue**: API key errors  
**Solution**: Verify your `.env` file exists and contains valid `OPENAI_API_KEY`

**Issue**: Memory errors on HuggingFace  
**Solution**: Reduce the number of markdown files or decrease chunk size in `create_vector_store()`

**Issue**: Slow first response  
**Solution**: This is normal - the embedding model downloads on first run. Subsequent responses will be faster.

## Cost Considerations

- **Embeddings**: Free (runs locally using sentence-transformers)
- **Vector Store**: Free (FAISS runs in-memory)
- **LLM API**: Pay-per-use (OpenAI charges per token)
- **Hosting**: Free on HuggingFace Spaces

Estimated cost: ~$0.002 per conversation (varies based on question/answer length)

## Support

For issues or questions, please contact your development team or refer to:
- [LangChain Documentation](https://python.langchain.com/)
- [Gradio Documentation](https://www.gradio.app/docs)
- [HuggingFace Spaces Guide](https://huggingface.co/docs/hub/spaces)

