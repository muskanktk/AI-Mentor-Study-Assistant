# sets the connection for postgres database using the DATABASE_URL from the .env file
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    os.getenv("DATABASE_URL")
)

print("Connected!")