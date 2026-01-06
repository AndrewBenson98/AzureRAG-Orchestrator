import os
from fastapi import FastAPI
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from urllib.parse import urlparse
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI

# Load environment variables from a .env file into process environment
load_dotenv()

# Create the FastAPI application instance that will serve HTTP requests
app = FastAPI()

# -----------------------------
# Clients
# -----------------------------

# Azure AI Search client
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

# Azure OpenAI client
ai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# -----------------------------
# Endpoints
# ----------------------------- 
# Define a GET endpoint at /ask that takes a question as a query parameter
@app.get("/ask")
async def ask_question(question: str):
    """
    Endpoint: GET /ask?question=...

    Flow:
    1) RETRIEVE: run a search against Azure AI Search to find
       the most relevant document chunks for the user's question.
    2) CLEAN: extract only the text snippets from the search results and
       compose a single `context` string to give to the model.
    3) GENERATE: call the Azure OpenAI chat completion API with the
       provided context and the original question, then return the model's
       answer.
    """

    # 1. RETRIEVE: Perform the search (returns an iterable of results)
    results = search_client.search(search_text=question, top=3)
    
    # 2. CLEAN: Build a single string with only the textual chunks.
    #    Each result is expected to be a document containing a `chunk` field.
    context = ""
    for result in results:
        # Append each retrieved chunk to the context. The exact field name
        # (`chunk`) depends on how your index documents were constructed.
        context += f"\n: {result['chunk']}\n"

    # 3. GENERATE: Send the composed context and the user's question to the
    #    Azure OpenAI chat completions API. We pass a `system` message to
    #    instruct the model to use ONLY the provided context.
    response = ai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer the user question using ONLY the provided context. If the information is spread across multiple snippets, synthesize them into a clear answer. If the context absolutely does not contain the answer, explain what information is missing rather than making up a fact."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )

    # Return the assistant's textual reply as JSON
    return {"answer": response.choices[0].message.content}