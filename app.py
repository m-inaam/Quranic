from flask import Flask, jsonify, request
import cohere
from pinecone import Pinecone
import os
from flask_cors import CORS, cross_origin 

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
cors = CORS(app, resources={r'*': {'origins':'*'}})


pinecone_api_key = os.getenv('p_api_key')
cohere_api_key = os.getenv('c_api_key')

@app.route("/")
@cross_origin(origins='*')
def hello_world():
    return "<p>Hello, World!</p>"

co = cohere.Client(cohere_api_key)
pc = Pinecone(pinecone_api_key) # add your pinecone API key here

index_name = 'quranic'
index = pc.Index(index_name)

limit = 3000

def retrieve(query):
    xq = co.embed(
        texts=[query],
        model='multilingual-22-12',
        truncate='NONE'
    ).embeddings
    # search pinecone index for context passage with the answer
    xc = index.query(vector=xq, top_k=3, include_metadata=True)

    # Extract relevant information from the matches
    surahs = [str(x['metadata']['Surah']) for x in xc['matches']]
    ayahs = [str(x['metadata']['Ayat']) for x in xc['matches']]
    arabics = [str(x['metadata']['Arabic']) for x in xc['matches']]
    tafaseers = [str(x['metadata']['Tafaseer1']) for x in xc['matches']]

    # Combine the information into formatted contexts
    contexts = [
        f"Surah: {surah}\n Ayah: {ayah}\n Arabic: {arabic}\n"
        for surah, ayah, arabic in zip(surahs, ayahs, arabics)
    ]

    # Build the prompt with the retrieved contexts included
    prompt_start = (
        f"Answer the Query based on the contexts, if it's not in the contexts say 'I don't know the answer'. \n\n"
        f"Context:\n"
    )
    prompt_end = (
        f"\n\nQuery: {query}\nAnswer in the language of Query, if Query is in English Answer in English. Please provide reference Quran verses."
    )

    # Append contexts until hitting the limit
    for i in range(1, len(contexts)):
        if len("".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "".join(contexts) +
                prompt_end
            )
    return prompt

def complete(prompt):
  response = co.generate(
                          model='c4ai-aya',
                          prompt=prompt,
                          max_tokens=3000,
                          temperature=0.4,
                          k=0,
                          stop_sequences=['\n\n'],
                          return_likelihoods='NONE'
                        )
  return response.generations[0].text.strip()

@app.route('/api/predict', methods=['POST'])
def predict():
    query = request.json.get("query")
    query_with_contexts = retrieve(query)
    print(query_with_contexts)
    bot = complete(query_with_contexts)

    return {"bot": bot}
    
if __name__ == '__main__':
    app.run(debug=True)