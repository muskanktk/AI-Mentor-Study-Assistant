from database import get_database

db = get_database()

print("Connected!")

print(db.list_collection_names())