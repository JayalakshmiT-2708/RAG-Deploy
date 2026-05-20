from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os
import uuid

from database import (
    load_pdf,
    split_docs,
    create_vectorstore
)

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = Flask(__name__)

#folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#global variables
vectorstore = None
retriever = None

llm = ChatMistralAI(model="mistral-small-2506")


#home page
@app.route("/")
def home():
    return render_template("index.html")


#upload pdf
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vectorstore, retriever

    # safety check
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"})

    # safe filename to avoid overwrite issues
    file_path = os.path.join(
        UPLOAD_FOLDER,
        str(uuid.uuid4()) + ".pdf"
    )

    file.save(file_path)

    # -------------------------
    # RAG PIPELINE
    # -------------------------
    docs = load_pdf(file_path)
    chunks = split_docs(docs)

    vectorstore = create_vectorstore(chunks)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return jsonify({"message": "PDF processed successfully"})


# -------------------------
# ASK QUESTION
# -------------------------
@app.route("/ask", methods=["POST"])
def ask_question():

    global retriever

    if retriever is None:
        return jsonify({"answer": "Please upload a PDF first"})

    query = request.json.get("question")

    if not query:
        return jsonify({"answer": "Question is empty"})

    # retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
           "system",
        """You are a strict and precise AI assistant for a document-based question answering system.

RULES YOU MUST FOLLOW:
1. Answer ONLY using the provided context from the document.
2. Do NOT use any external knowledge or prior training information.
3. If the answer is not explicitly present in the context, respond exactly with:
   "Not available in document."
4. Do not guess, assume, or generate information outside the context.
5. Keep answers short, accurate, and strictly based on the document.

Your goal is to ensure 100% factual grounding in the given document."""
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    return jsonify({
        "answer": response.content,
        "status": "success"
    })

#run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
