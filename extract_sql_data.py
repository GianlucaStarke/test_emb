import pyodbc
import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine

_ = load_dotenv(find_dotenv())

odbc_str = quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SERVER_DB')};"
    f"DATABASE={os.getenv('DATABASE')};"
    f"UID={os.getenv('USER_DB')};"
    f"PWD={os.getenv('PASS_DB')};"
)
tabelas = [
    "SE1010" , "SB1010", "SA1010" , "SD1010", "SF2010" , "SF1010" , "SE2010"
]

connection_string = f"mssql+pyodbc:///?odbc_connect={odbc_str}"
engine = create_engine(connection_string)

with open("dataset/sql_schema.jsonl", "w", encoding="utf-8") as f:
    for tabela in tabelas:
        try:
            print(f"🔎 Processando {tabela}...")
            df = pd.read_sql(f"SELECT TOP 10000 * FROM {tabela}", engine)
            schema = [{"column": col, "dtype": str(df[col].dtype)} for col in df.columns]

            def convert_row(row):
                new_row = {}
                for k, v in row.items():
                    if pd.isna(v):
                        new_row[k] = ""
                    elif isinstance(v, pd.Timestamp):
                        new_row[k] = v.isoformat()
                    else:
                        new_row[k] = v
                return new_row

            example_rows = [convert_row(row) for row in df.head(5).to_dict(orient="records")]

            doc = {
                "table": tabela,
                "schema": schema,
                "examples": example_rows
            }

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f" Erro em {tabela}: {e}")
