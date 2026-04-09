# FAQ + RAG Chatbot
The chatbot divides into main two parts:
(part 1)
FAQ-Based:
Uses predefined Q&A from a CSV file.
Stores data in ChromaDB using embeddings
Finds similar questions and returns answers using llms.
User queries like: payment methods, return policy and other FAQs.

(part 2)
Retrieval-Based
Retrieves data from a database.
Used for changing info like Product prices,Discounts / sales

Python, Pandas, ChromaDB, SentenceTransformers, ChatGroq
User Query → Retrieve Data → LLM → Final Answer

If no answer found:
"contact our team for further details.."
