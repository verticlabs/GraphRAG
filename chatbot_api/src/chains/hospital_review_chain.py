import os
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

HOSPITAL_QA_MODEL = os.getenv("AGENT_MODEL")

# Define the list of properties to include in the text dictionary
# retrieval_query = """WITH node AS Post, score as similarity
#                     ORDER BY similarity DESC LIMIT 5
#                     CALL { WITH Post
#                             WHERE Post.`Full Text` IS NOT NULL
#                             RETURN Post.`Full Text` as text, Post AS result
#                         }
#                     WITH result, text, similarity
#                     CALL {
#                         WITH result
#                         OPTIONAL MATCH (Author)-[:CREATED]->(:Post), (Author)<-[:OWNS]-(entity:Entity)
#                         WITH result, Author.`Full Name` as authorName, apoc.text.join(collect(entity.EntityName),';') as entities
#                         WHERE authorName IS NOT NULL OR entities > ""
#                         WITH result, authorName, entities
#                         ORDER BY result.score DESC
#                         RETURN result as document, result.score as similarity, authorName, entities
#                     }
#                     RETURN  text,
#                         similarity as score,
#                         {documentId: coalesce(document.ResourceId,''), author: coalesce(authorName,''), entities: coalesce(entities,''), source: document.`Page Type Name`} AS metadata
#                         """
# neo4j_vector_index_ = Neo4jVector.from_existing_index(
#     embedding=OpenAIEmbeddings(),
#     url=os.getenv("NEO4J_URI"),
#     username=os.getenv("NEO4J_USERNAME"),
#     password=os.getenv("NEO4J_PASSWORD"),
#     index_name="post_index",
#     node_label="Post",
#     retrieval_query=retrieval_query,
#     embedding_node_property="embedding",
# )

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="post_index",
    node_label="Post",
    text_node_properties=[
        "Page Type Name",
        "Full Text",
    ],
    embedding_node_property="embedding",
)

review_template = """Your job is to use user posts to answer questions about the social media posts related to cardiovascular disease and the conversation around the posts.
Please try to answer as accurate as possible using the provided context.
Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=20),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt

if __name__ == "__main__":
    # Test code for debugging
    question = "which kind of problems are people facing with cardiovascular disease based on social media posts?"
    
    # Retrieve the context
    context = reviews_vector_chain.retriever.get_relevant_documents(question)
    context_str = "\n".join([doc.page_content for doc in context])
    
    # Generate the full prompt
    full_prompt = review_prompt.format(context=context_str, question=question)
    
    # Invoke the chain
    result = reviews_vector_chain.invoke({"query": question})
    
    print(f"Question: {question}")
    print(f"Full Prompt:")
    print(full_prompt)
    print(f"\nAnswer: {result['result']}")
