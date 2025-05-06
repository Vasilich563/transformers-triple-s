import enum
from sqlalchemy import text
from sqlalchemy import Engine
from backend.embedding_system.make_db import LEVEL_TABLE_NAME_PREFIX, SCHEMA_NAME, EMBEDDING_DIM, CATALOG_TABLE_NAME
from threading import Thread


class SelectIndexes(enum.Enum):
    document_path = 0
    document_name = 1
    snippet = 2
    cos_distance = 3


class DBCrud:

    __insert_template = """
        INSERT INTO {schema_name}.{table_name} 
            (snippet_name, document_path, document_name, snippet, embedding) 
            VALUES (:snippet_name, :document_path, :document_name, :snippet, :embedding)}
    """

    __insert_catalog_table_template = """
        INSERT INTO {schema_name}.{table_name} 
            (document_path, document_name, snippet) 
            VALUES (:document_path, :document_name, :snippet)}
    """

    __select_by_embedding_template = """
        SELECT DISTINCT ON (document_path) document_path, document_name, snippet, cos_distance FROM
            (SELECT 
                  document_path,
                  document_name, 
                  snippet, 
                  embedding <=> CAST(:query_embedding AS vector({embedding_dim})) AS cos_distance
                FROM {schema_name}.{table_name} ORDER BY embedding <=> CAST(:query_embedding AS vector({embedding_dim}))
            ) distances_query 
            ORDER BY document_path, cos_distance
            LIMIT :limit
    """

    __select_by_name_template = """
        SELECT
            document_path, document_name, snippet 
            FROM {schema_name}.{table_name}
            WHERE document_name ILIKE :document_name
            ORDER BY document_name
            LIMIT :limit
        """

    __select_by_name_exactly_template = """
        SELECT
            document_path, document_name, snippet 
            FROM {schema_name}.{table_name} 
            WHERE document_name = :document_name
            ORDER BY document_name
            LIMIT :limit
        """

    __select_files_asc_template = """
        SELECT
            document_path, document_name, snippet 
            FROM {schema_name}.{table_name} 
            ORDER BY document_name ASC
            LIMIT :limit
            OFFSET :offset
    """

    __select_files_desc_template = """
        SELECT
            document_path, document_name, snippet 
            FROM {schema_name}.{table_name} 
            ORDER BY document_name DESC
            LIMIT :limit
            OFFSET :offset
        """

    __select_amount_of_files_template = """
        SELECT count(*) FROM {schema_name}{table_name}
    """


    __delete_template = """
        DELETE FROM {schema_name}.{table_name} WHERE document_path = document_path
    """


    def __init__(self, db_engine: Engine):
        self._db_engine: Engine = db_engine
        self._level1_name = f"{LEVEL_TABLE_NAME_PREFIX}1"
        self._level2_name = f"{LEVEL_TABLE_NAME_PREFIX}2"
        #self._level3_name = f"{LEVEL_TABLE_NAME_PREFIX}3"


    def write_level1_snippet_rows(self, list_of_rows):
        """
        :param list_of_rows: List[
            Dict[
                "snippet_name": str,
                "document_path": str,
                "document_name": str,
                "snippet": str,
                "embedding": List[float]
            ]
        ]
        :return: None
        """
        with self._db_engine.begin() as connection:
            thread = Thread(
                target=connection.execute,
                args=(
                    text(self.__insert_template.format(schema_name=SCHEMA_NAME, table_name=self._level1_name)),
                    list_of_rows
                ),
                daemon=True
            )
            thread.start()

    def write_level2_snippet_rows(self, list_of_rows):
        """
        :param list_of_rows: List[
            Dict[
                "snippet_name": str,
                "document_path": str,
                "document_name": str,
                "snippet": str,
                "embedding": List[float]
            ]
        ]
        :return: None
        """
        with self._db_engine.begin() as connection:
            thread = Thread(
                target=connection.execute,
                args=(
                    text(self.__insert_template.format(schema_name=SCHEMA_NAME, table_name=self._level2_name)),
                    list_of_rows
                ),
                daemon=True
            )
            thread.start()

    # def write_level3_snippet_rows(self, list_of_rows):
    #     """
    #     :param list_of_rows: List[
    #         Dict[
    #             "snippet_name": str,
    #             "document_path": str,
    #             "document_name": str,
    #             "snippet": str,
    #             "embedding": List[float]
    #         ]
    #     ]
    #     :return: None
    #     """
    #     with self._db_engine.begin() as connection:
    #         thread = Thread(
    #             target=connection.execute,
    #             args=(
    #                 text(self.__insert_template.format(schema_name=SCHEMA_NAME, table_name=self._level3_name)),
    #                 list_of_rows
    #             ),
    #             daemon=True
    #         )
    #         thread.start()


    def write_catalog_row(self, row):
        """
        :param row: Dict[
            "document_path": str,
            "document_name": str,
            "snippet": str
        ]
        :return: None
        """
        with self._db_engine.begin() as connection:
            thread = Thread(
                target=connection.execute,
                args=(
                    text(self.__insert_catalog_table_template.format(schema_name=SCHEMA_NAME, table_name=CATALOG_TABLE_NAME)),
                    [row]
                ),
                daemon=True
            )
            thread.start()


    def _select_by_name(self, connection, document_name, limit, exactly_flag):
        if exactly_flag:
            return (
                connection.execute(
                    text(self.__select_by_name_exactly_template.format(schema_name=SCHEMA_NAME, table_name=CATALOG_TABLE_NAME)),
                    {"document_name": document_name, "limit": limit}
                ).all()
            )
        else:
            return (
                connection.execute(
                    text(self.__select_by_name_template.format(schema_name= SCHEMA_NAME, table_name=CATALOG_TABLE_NAME)),
                    {"document_name": f"%{document_name}%", "limit": limit}
                ).all()
            )

    def select_by_name(self, document_name, limit, exactly_flag):
        with self._db_engine.begin() as connection:
            queries_results = self._select_by_name(connection, document_name, limit, exactly_flag)

        return queries_results[:limit]


    def _select_from_level_for_one_embedding(self, connection, level_table_name, query_embedding_list, limit):
        return (
            connection.execute(
                text(
                    self.__select_by_embedding_template.format(
                        schema_name=SCHEMA_NAME, table_name=level_table_name, embedding_dim=EMBEDDING_DIM
                    )
                ),
                {"limit": limit, "query_embedding": query_embedding_list[0]}
            )
        ).all()

    def _select_from_level(self, connection, level_table_name, query_embedding_list, limit):
        queries_results = []
        for i in range(len(query_embedding_list)):
            queries_results.extend(
                self._select_from_level_for_one_embedding(connection, level_table_name, query_embedding_list[i], limit)
            )
        queries_results.sort(key=lambda x: x[SelectIndexes.cos_distance.value])

        return queries_results

    def select_from_level1(self, query_embedding_list, limit):
        with self._db_engine.begin() as connection:
            queries_results = self._select_from_level(connection, self._level1_name, query_embedding_list, limit)

        return queries_results[:limit]

    def select_from_level2(self, query_embedding_list, limit):
        with self._db_engine.begin() as connection:
            queries_results = self._select_from_level(connection, self._level2_name, query_embedding_list, limit)

        return queries_results[:limit]

    # def select_from_level3(self, query_embedding_list, limit):
    #     with self._db_engine.begin() as connection:
    #         queries_results = self._select_from_level(connection, self._level3_name, query_embedding_list, limit)
    #
    #     return queries_results[:limit]

    def _delete_from_table(self, connection, table_name, document_path):
        connection.execute(
            text(self.__delete_template.format(schema_name=SCHEMA_NAME, table_name=table_name)),
            {"document_path": document_path}
        )

    def remove_from_all_levels(self, document_path):
        with self._db_engine.begin() as connection:
            catalog_thread = Thread(
                target=self._delete_from_table,
                args=(connection, CATALOG_TABLE_NAME, document_path),
                daemon=True
            )
            catalog_thread.start()

            l1_thread = Thread(
                target=self._delete_from_table,
                args=(connection, self._level1_name, document_path),
                daemon=True
            )
            l1_thread.start()

            l2_thread = Thread(
                target=self._delete_from_table,
                args=(connection, self._level2_name, document_path),
                daemon=True
            )
            l2_thread.start()

            # l3_thread = Thread(
            #     target=self._delete_from_table,
            #     args=(connection, self._level3_name, document_path),
            #     daemon=True
            # )
            # l3_thread.start()




