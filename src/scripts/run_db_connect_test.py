from src.database.database_utils import query_db
from dotenv import load_dotenv


def main():
    load_dotenv()
    res = query_db('SELECT COUNT(*) FROM people', db_name='cordis_temporary')
    print(res)

if __name__ == "__main__":
    main()
