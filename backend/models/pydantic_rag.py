import re
import json
import torch
import spacy
import ollama
import chromadb

from typing import Literal
from pydantic import BaseModel
from ollama import chat
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# 🔹 NLP & Embedding Models
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# 🔹 ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="Reccourse")

# 🔹 Pydantic Model 
class UserQueryParsed(BaseModel):
    topic: str
    level: Literal["beginner", "intermediate", "advanced"]
    number_of_courses: int = 3 

def extract_info_from_llm(query: str) -> UserQueryParsed:
    system_prompt = """
You are an intelligent assistant. Your role is to extract structured information from a learning query.
You must return a JSON containing:

- topic: the subject of the course (in English, string)
- level: the level (beginner, intermediate, advanced)
- number_of_courses: the number of courses requested (int); if user don't specify return 3

⚠️ Reply ONLY in JSON, without any explanation.
    """

    response = chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    content = response["message"]["content"].strip()

    # 🔍 Extraire automatiquement le JSON à l'intérieur des ``` si présent
    json_match = re.search(r"{[\s\S]*?}", content)
    if not json_match:
        print("❌ No valid JSON structure found in response")
        print("Full content:", content)
        raise ValueError("Invalid response format from LLM")

    cleaned_content = json_match.group(0)

    try:
        data = json.loads(cleaned_content)
        data["level"] = data["level"].lower()
        return UserQueryParsed(**data)
    except Exception as e:
        print("❌ JSON Parsing error:", e)
        print("Cleaned JSON:", cleaned_content)
        raise


# 🔹 Search in the database
def search_courses(query):
    info = extract_info_from_llm(query)
    levels = [info.level]
    print(levels)
    number_of_courses_requested = info.number_of_courses
    print(number_of_courses_requested)
    query_embedding = model.encode(query, truncation=True)
    results = collection.query(query_embeddings=[query_embedding], n_results=10)

    filtered_results = [
        res for res in results["metadatas"][0] if res["level"].lower() in [n.lower() for n in levels]
    ]

    if len(filtered_results) < number_of_courses_requested:
        print(f"⚠️ Only {len(filtered_results)} courses available out of {number_of_courses_requested} requested.")
    return filtered_results[:number_of_courses_requested], levels

# 🔹 Create the prompt for generation
def build_prompt(courses, query):
    prompt = f"You are a teaching assistant. Here's a user request: \"{query}\".\n\n"
    prompt += "Here are the recommended courses:\n"
    for i, c in enumerate(courses[0]):
        prompt += f"{i+1}. **{c['title']}** - **{c['skills']}** ({c['level']})\n"
        prompt += f"   🔗 {c['url']}\n"
        if "rating" in c:
            prompt += f"   ⭐ Rating: {c['rating']}/5\n"
    prompt += "\nCan you summarize these courses with their   URLs and the level; dont' specify any couses except the recommended courses"
    return prompt

#  Final generation with Ollama
def generate_response(prompt, model="llama3.2:3b"):
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# 🔹 Full RAG pipeline
def rag_pipeline(query):
    courses = search_courses(query)
    prompt = build_prompt(courses, query)
    response = generate_response(prompt)
    return response
import time

# 🔹 CLI Interface
if __name__ == "__main__":
    start_time = time.time()
    user_input = input("🎓 What would you like to learn today?\n> ")
    results = rag_pipeline(user_input)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Temps d'exécution du pipeline : {execution_time} secondes")
    print(results)


