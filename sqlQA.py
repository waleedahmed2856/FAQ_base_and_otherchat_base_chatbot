import sqlite3
import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re

load_dotenv(".env.txt")

sql_prompt = """
You are a SQL generator.

<schema>
table: product
fields:
product_link, title, brand, price, discount, avg_rating, total_ratings
</schema>
RULES:
ONLY SELECT queries allowed
Always use SELECT *
Use LIKE '%value%' for text search
best/top → ORDER BY avg_rating DESC LIMIT 5
popular → ORDER BY total_ratings DESC
cheap → ORDER BY price ASC
OUTPUT FORMAT (STRICT):
<SQL>
SELECT ...
</SQL>

NO explanation, NO markdown, NO extra text.
"""

comprehension_prompt = """You are an expert in understanding the context of the question and replying based on the 
data pertaining to the question provided. You will be provided with Question: and Data:. The data will be in the form of 
an array or a dataframe or dict. Reply based on only the data provided as Data for answering the question asked as Question.
Do not write anything like 'Based on the data' or any other technical words. Just a plain simple natural language response.
The Data would always be in context to the question asked. For example is the question is “What is the average rating?” 
and data is “4.3”, then answer should be “The average rating for the product is 4.3”. So make sure the response is curated with 
the question and data. Make sure to note the column names to have some context, if needed, for your response.
There can also be cases where you are given an entire dataframe in the Data: field. Always remember that the data field contains the
answer of the question asked. All you need to do is to always reply in the following format when asked about a product: 
Produt title, price in indian rupees, discount, and rating, and then product link. Take care that all the products are listed in 
list format, one line after the other. Not as a paragraph.
For example:
1. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
2. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
3. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
"""

def sql_data(query):
    if query.strip().upper().startswith("SELECT"):
        with sqlite3.connect("db.sqlite") as conn:
            data = pd.read_sql_query(query, conn)
            return data
        
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=200
)

def create_query(question):
    global llm, sql_prompt
    full_prompt = f"""{sql_prompt}
    Question:
    {question}
    """
    response = llm.invoke(full_prompt)
    answer = response.content.strip()

    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, answer, re.DOTALL)

    if matches:
        return " ".join(matches[0].split())
    else:
        return "for the specified product contact our team."


def get_final_answer(question, data):
    if data.empty:
        return "No products found matching your criteria."
    
    data_str = data.to_dict(orient='records')
    full_prompt = f"{comprehension_prompt}\nQuestion:\n{question}\nData:\n{data_str}"
    response = llm.invoke(full_prompt)
    return response.content.strip()

if __name__==("__main__"):

    question = "women best shoes and highest rating shoes"
    get_query = create_query(question)
    print(get_query)
    retrive_data_from_sql = sql_data(get_query)
    result = get_final_answer(question, retrive_data_from_sql)

    print(result)
