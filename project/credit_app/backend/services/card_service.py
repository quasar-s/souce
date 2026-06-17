from fastapi import UploadFile
from backend.services.db_service import get_connection
import csv
from langchain_core.prompts import ChatPromptTemplate
from backend.ai.llm import watson_llm
from langchain_core.output_parsers import StrOutputParser
from backend.services.db_service import get_table_columns
from langchain_core.runnables import RunnablePassthrough


async def upload_csv(file: UploadFile):
    """
    csv 파일을 테이블에 저장
    """
    conn = get_connection()
    cursor = conn.cursor()

    # file.read() : 비동기 함수임
    contents = await file.read()
    csv_text = contents.decode("utf-8")
    reader = csv.DictReader(csv_text.splitlines())
    count = 0
    for row in reader:
        cursor.execute(
            """
INSERT INTO transactions(date,category,merchant,amount)
VALUES(?,?,?,?)
""",
            (row["date"], row["category"], row["merchant"], row["amount"]),
        )
        count += 1

    conn.commit()
    conn.close()
    return {"message": f"{count} 건 저장 완료"}


def card_history():
    """
    db 에서 카드 정보 조회
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute("""
    SELECT * FROM transactions ORDER BY date DESC
    """)
        rows = cursor.fetchall()
        query_result = [dict(row) for row in rows]

        conn.close()
    except Exception as e:
        query_result = [f"SQL 실행 오류 : {e}"]

    return query_result


def get_dashboard():
    conn = get_connection()
    cursor = conn.cursor()

    # 카테고리 별 사용금액
    cursor.execute("""
SELECT category, SUM(amount)
FROM transactions
GROUP BY category
ORDER BY SUM(amount) DESC
""")

    category_rows = cursor.fetchall()

    # 월별 사용금액
    cursor.execute("""
SELECT strftime('%Y-%m',date), SUM(amount)
FROM transactions
GROUP BY strftime('%Y-%m',date)
ORDER BY strftime('%Y-%m',date)
""")

    monthly_rows = cursor.fetchall()
    conn.close()

    return {
        "category": [{"category": row[0], "amount": row[1]} for row in category_rows],
        "monthly": [{"month": row[0], "amount": row[1]} for row in monthly_rows],
    }


def sql_generate_llm(question):
    """자연어 -> SQL로 변경"""
    sql_prompt = ChatPromptTemplate.from_template("""
당신은 SQLite 전문가입니다.
                                                  
테이블명 : transactions

컬럼 : {columns}

질문 : {question}

SQL 만 출력하세요                                                  
""")

    columns = get_table_columns("transactions")
    sql_chain = sql_prompt | watson_llm | StrOutputParser()

    sql = sql_chain.invoke({"columns": columns, "question": question})
    print(f"sql : {sql}")

    sql = sql.replace("```sql", "").replace("```", "").strip()

    # sql 문 실제 실행 결과 받기
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(sql)
    query_result = cursor.fetchall()
    conn.close()

    return query_result


def card_analysis(question):

    query_result = sql_generate_llm(question)

    analysis_prompt = ChatPromptTemplate.from_template("""
사용자 질문 : {question}

SQL 결과
{result}

결과를 설명하고 소비 습관을 분석하고 절약 팁을 제시해 주세요.                                                                                                  
""")

    analysis_chain = analysis_prompt | watson_llm | StrOutputParser()
    answer = analysis_chain.invoke({"question": question, "result": query_result})

    return {"message": answer}
