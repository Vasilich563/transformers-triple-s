from sqlalchemy import create_engine, text

EMBEDDING_DIM = 768
SCHEMA_NAME = "triple_s"
LEVEL_TABLE_NAME_PREFIX = "snippet_level"
CATALOG_TABLE_NAME = "documents_catalog"


def actions_on_snippet_level(connection, level):
    connection.execute(
        text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{LEVEL_TABLE_NAME_PREFIX}{level}(
                snippet_name TEXT NOT NULL PRIMARY KEY,
                document_path TEXT NOT NULL,
                document_name TEXT NOT NULL,
                snippet TEXT NOT NULL,
                embedding vector(:dim)
            );
        """), {"dim": EMBEDDING_DIM}
    )

    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS
                {LEVEL_TABLE_NAME_PREFIX}{level}_embedding_hnsw_index 
                ON {SCHEMA_NAME}.{LEVEL_TABLE_NAME_PREFIX}{level} USING hnsw (embedding vector_cosine_ops);
        """)
    )



if __name__ == "__main__":
    db_engine = create_engine("postgresql://postgres:ValhalaWithZolinks@localhost:5432/postgres")

    with db_engine.begin() as connection:
        connection.execute(text("""CREATE EXTENSION IF NOT EXISTS vector;"""))

        connection.execute(text(f"""CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};"""))

        connection.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{CATALOG_TABLE_NAME}(
                    document_path TEXT NOT NULL PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    snippet TEXT NOT NULL
                );
            """)
        )

        connection.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS
                    {CATALOG_TABLE_NAME}_document_name_hash_index 
                    ON {SCHEMA_NAME}.{CATALOG_TABLE_NAME} USING HASH (document_name);
            """)
        )

        actions_on_snippet_level(connection, 1)

        actions_on_snippet_level(connection, 2)

        #actions_on_snippet_level(connection, 3)







