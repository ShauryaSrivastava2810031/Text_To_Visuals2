from flask import Flask, request, jsonify, render_template
import os
from gemini_api import get_gemini_response
from db_utils import read_sql_query

app = Flask(__name__)
DB_PATH = r"C:/Users\shaurya.s\OneDrive - Comviva Technologies LTD\Desktop\T2S/retails.db"

# Define the prompt
prompt = [
    """
    You are an expert in SQL and can translate any English question into a precise and accurate 
    SQL query, even if the input contains grammatical errors, punctuation mistakes, or poorly 
    structured sentences. You have a comprehensive understanding of SQL, including SELECT, 
    INSERT, UPDATE, DELETE, JOIN, GROUP BY, ORDER BY, WHERE, HAVING, aggregate functions 
    (e.g., COUNT, AVG, SUM), string functions (e.g., CONCAT, SUBSTRING), date and time functions, 
    subqueries, indexing, and advanced techniques like window functions and CTEs 
    (Common Table Expressions).

    Make sure the SQL Query should and must not have any or at both the start and end of the 
    query. Also, it should not contain "sql" at the beginning of the query.

    **IMPORTANT RULE**:
    - The correct format is: SELECT * FROM retails_sales;

    The SQL database has the name retails and includes the following table retails_sales with 
    columns and data types:    
    Transaction_ID INT,
    Date DATE,
    Customer_ID VARCHAR(10),
    Gender VARCHAR(10),
    Age INT,
    Product_Category VARCHAR(50),
    Quantity INT,
    Price_per_Unit DECIMAL(10, 2),
    Total_Amount DECIMAL(10, 2).

    Examples:
    Question 1: How many records are in the table?
    SQL Query: SELECT COUNT(*) FROM retails_sales;

    Question 2: List all transactions for male customers.
    SQL Query: SELECT * FROM retails_sales WHERE Gender = "Male";

    Question 3: Find the average total amount spent by customers in the "Electronics" category.
    SQL Query: SELECT AVG(Total_Amount) FROM retails_sales WHERE Product_Category = "Electronics";

    Question 4: Show customer IDs and their total spending, ordered by spending in descending 
    order.
    SQL Query: SELECT Customer_ID, SUM(Total_Amount) AS Total_Spending FROM retails_sales GROUP BY Customer_ID ORDER BY Total_Spending DESC;

    Question 5: Retrieve total quantities sold grouped by product category.
    SQL Query: SELECT Product_Category, SUM(Quantity) AS Total_Quantity FROM retails_sales GROUP BY Product_Category;

    Question 6: Retrieve all records.
    SQL Query: SELECT * FROM retails_sales;

    Question 7: Add a column in the table named "Discount" with a default value of 0.
    SQL Query: ALTER TABLE retails_sales ADD COLUMN Discount DECIMAL(10, 2) DEFAULT 0;

    Always ensure the SQL query is optimized and adheres to best practices. Correct any 
    grammatical or sequence errors in the input and generate the most appropriate SQL query. 
    Handle all SQL-related questions, including complex joins, subqueries, and database 
    management tasks, with precision and efficiency.
"""
]

@app.route('/')
def index():
    return render_template('index.html')

@app.before_request
def log_request_info():
    print("\n--- New Request ---")
    print("Headers:", request.headers)
    print("Method:", request.method)
    print("Data:", request.get_data(as_text=True))

@app.route('/query', methods=['POST'])
def query():
    print("Inside /query route")
    try:
        data = request.get_json
        print("Recieved Data:", data)

        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No questions provided."}), 400

        sql_query = get_gemini_response(question, prompt)
        print(sql_query)

        query_result = read_sql_query(sql_query, DB_PATH)

        if isinstance(query_result, str):
            return jsonify({"query": sql_query, "result": query_result})

        result_json = query_result.to_dict(orient="records")

        return jsonify({"query": sql_query, "result": result_json})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)