import os
import json
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from docx import Document
from typing import List, Dict
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import agentops

# -------------------- Setup -------------------- #
st.set_page_config(page_title="College Assistant", layout="wide")
agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

# Initialize services
@st.cache_resource
def init_services():
    mongo_client = MongoClient(os.getenv("MONGO_URI"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return mongo_client, model, groq_client

mongo_client, model, groq_client = init_services()

# Database connection
db = mongo_client["college_db"]
collection = db["college_info"]

# Create vector search index
def create_search_index():
    try:
        db.command("dropSearchIndex", "college_info", "vector_index")
    except:
        pass
    
    index_definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "type": "knnVector",
                    "dimensions": 384,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    db.command({
        "createSearchIndexes": "college_info",
        "indexes": [{
            "name": "vector_index",
            "definition": index_definition
        }]
    })

# -------------------- Document Processing -------------------- #
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_text(text)

def load_and_embed_documents():
    with st.spinner("Processing documents..."):
        collection.delete_many({})
        # Modify to add your own files
        FILE_PATHS = [
            r"C:\\Users\\chakr\\OneDrive\\Desktop\\Replaice Project\\data\\college_docs\\college_details.docx",
            r"C:\\Users\\chakr\\OneDrive\\Desktop\\Replaice Project\\data\\college_docs\\teachers.docx",
            r"C:\Users\chakr\OneDrive\Desktop\Replaice Project\data\college_docs\student_story.docx",
            #A csv file with Student ID and Course Name for verification
            r"C:\\Users\\chakr\\OneDrive\\Desktop\\Replaice Project\\data\\students_results.csv"
        ]
        
        for file_path in FILE_PATHS:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    content = " | ".join(f"{col}: {row[col]}" for col in df.columns)
                    chunks = chunk_text(content)
                    for chunk in chunks:
                        embedding = model.encode([chunk])[0].tolist()
                        collection.insert_one({
                            "content": chunk,
                            "title": os.path.basename(file_path),
                            "type": "csv",
                            "embedding": embedding
                        })
            elif file_path.endswith('.docx'):
                doc = Document(file_path)
                full_text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
                chunks = chunk_text(full_text)
                for chunk in chunks:
                    embedding = model.encode([chunk])[0].tolist()
                    collection.insert_one({
                        "content": chunk,
                        "title": os.path.basename(file_path),
                        "type": "docx",
                        "embedding": embedding
                    })
        
        create_search_index()
        st.success(f"Processed {len(FILE_PATHS)} documents with {collection.count_documents({})} chunks!")

# -------------------- Core Functions -------------------- #
def is_grade_query(question: str) -> bool:
    """Use Groq function calling to determine if this is a grade-related query"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "determine_query_type",
                "description": "Determine if a question is asking about student grades/marks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_grade_query": {
                            "type": "boolean",
                            "description": "Whether the question is about student grades/marks"
                        },
                        "requires_student_id": {
                            "type": "boolean",
                            "description": "Whether student ID is mentioned in the query"
                        },
                        "requires_course": {
                            "type": "boolean",
                            "description": "Whether course name is mentioned in the query"
                        }
                    },
                    "required": ["is_grade_query"]
                }
            }
        }
    ]
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="gemma2-9b-it",
            tools=tools,
            tool_choice="auto"
        )
        
        if response.choices[0].message.tool_calls:
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("is_grade_query", False)
        
        return False
    except Exception:
        return False

def query_documents(question: str, top_k: int = 5) -> List[Dict]:
    """Retrieve relevant documents using vector search"""
    query_embedding = model.encode([question])[0].tolist()
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 126,
                "limit": top_k
            }
        },
        {
            "$project": {
                "content": 1,
                "title": 1,
                "type": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    return list(collection.aggregate(pipeline))

def get_student_grade(student_id: str, course: str) -> str:
    STUDENT_CSV_PATH = r"C:\\Users\\chakr\\OneDrive\\Desktop\\Replaice Project\\data\\students_results.csv"
    student_df = pd.read_csv(STUDENT_CSV_PATH)
    
    row = student_df.loc[(student_df['student_id'] == student_id) &
                         (student_df['results (JSON)'].str.contains(course, case=False))]
    if not row.empty:
        results_json = json.loads(row.iloc[0]['results (JSON)'])
        for res in results_json:
            if res['course'].lower() == course.lower():
                return f"The grade for Student ID {student_id} in {course} is {res['grade']}."
    return None

def generate_answer(question: str, context: str) -> str:
    """Generate answer using Groq with the retrieved context"""
    prompt = f"""
    You are a college assistant. Use ONLY the following context to answer.
    If you don't know, say you don't have information. Never make up answers.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def verify_answer(question: str, answer: str, context: str) -> str:
    """Verify the answer against the context"""
    prompt = f"""
    Verify if the following answer is fully supported by the context:
    
    Question: {question}
    Answer: {answer}
    Context: {context[:3000]}  # Truncate to avoid token limits
    
    If the answer is correct, return it as-is.
    If incorrect or incomplete, provide an improved version.
    remember the user just want the answer, not the explanation. and dont tell thing like its factual or stuff. just create a polished answer for the user according to his question
    Verified Answer:
    """
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"{answer}\n\n[Verification failed: {str(e)}]"

# -------------------- Streamlit UI -------------------- #
def main():
    st.title("🎓 College Assistant - Intelligent Query Handling")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        load_and_embed_documents()
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input("Ask a question about college information:", key="question")
            
            if question.lower().strip() == "exit":
                st.success("Session ended successfully.")
                return
            
            if question:
                # Use function calling to determine query type
                if is_grade_query(question):
                    with st.form("grade_form"):
                        student_id = st.text_input("Student ID:")
                        course = st.text_input("Course Name:")
                        submitted = st.form_submit_button("Get Grade")
                        
                        if submitted:
                            with st.spinner("Searching for grade..."):
                                grade = get_student_grade(student_id, course)
                                if grade:
                                    st.session_state.chat_history.append(f"User: {question}\nAssistant: {grade}")
                                    st.success(grade)
                                else:
                                    st.error("No matching record found")
                                    st.session_state.chat_history.append(f"User: {question}\nAssistant: No matching record found")
                else:
                    with st.spinner("Analyzing your question..."):
                        try:
                            # Retrieve relevant documents
                            relevant_docs = query_documents(question)
                            context = "\n\n---\n\n".join(
                                f"Source {i+1} ({doc['type']}): {doc['content']}"
                                for i, doc in enumerate(relevant_docs)
                            ) if relevant_docs else "No relevant information found."
                            
                            # Generate initial answer
                            answer = generate_answer(question, context)
                            
                            # Verify the answer
                            verified_answer = verify_answer(question, answer, context)
                            
                            # Update chat history
                            st.session_state.chat_history.append(f"User: {question}\nAssistant: {verified_answer}")
                            
                            # Display response
                            st.write("### Response")
                            st.write(verified_answer)
                            
                            with st.expander("Source Documents"):
                                if relevant_docs:
                                    for i, doc in enumerate(relevant_docs, 1):
                                        st.write(f"#### Document {i} (Score: {doc['score']:.3f})")
                                        st.write(f"**Title:** {doc['title']}")
                                        st.text(doc['content'][:500] + ("..." if len(doc['content']) > 500 else ""))
                                else:
                                    st.warning("No documents found")
                        except Exception as e:
                            st.error(f"Error processing request: {str(e)}")
        
        with col2:
            st.write("### Conversation History")
            history_text = "\n\n".join(st.session_state.chat_history[-10:])
            st.text_area("", history_text, height=400, disabled=True)
            
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()
