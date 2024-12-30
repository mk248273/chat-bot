
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Load from .env file for security
genai.configure(api_key=GOOGLE_API_KEY)

# Flask setup
app = Flask(__name__)

# Helper functions
def get_pdf_text(pdf):
    text = ""                           
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

def get_vector_store(chunks):
    #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(
        api_key="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",
        model_name="mixtral-8x7b-32768"
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Replace with actual embedding model ID
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    docs = docs[:1]
    chain = get_conversational_chain()
    # response = chain({
    #     context=context,
    #     question=user_question}
    # )
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

# Flask route for handling the API request
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_question = data.get('query')

    if not user_question:
        return jsonify({"error": "No query provided"}), 400
    try:
        # Get response from the chatbot
        response = user_input(user_question)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main function for testing purposes (can be removed when deploying)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=7888)



