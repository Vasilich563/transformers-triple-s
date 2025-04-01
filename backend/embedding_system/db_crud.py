import enum
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from make_db import LEVEL_TABLE_NAME_PREFIX, SCHEMA_NAME, EMBEDDING_DIM


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
        SELECT DISTINCT ON (document_path)
            document_path, :document_name AS document_name, snippet, 0 AS cos_distance 
            FROM {schema_name}.{table_name}
            WHERE document_name = :document_name
            ORDER BY document_path, snippet_name
            LIMIT :limit
        """

    __delete_template = """
        DELETE FROM {schema_name}.{table_name} WHERE document_path = document_path
    """


    def __init__(self, db_engine: AsyncEngine):
        self._db_engine: AsyncEngine = db_engine
        self._level1_name = f"{LEVEL_TABLE_NAME_PREFIX}1"
        self._level2_name = f"{LEVEL_TABLE_NAME_PREFIX}2"
        self._level3_name = f"{LEVEL_TABLE_NAME_PREFIX}3"


    async def write_level1_snippet_rows(self, list_of_rows):
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
        async with self._db_engine.begin() as connection:
            await connection.execute(
                text(self.__insert_template.format(schema_name=SCHEMA_NAME, table_name=self._level1_name)),
                list_of_rows
            )

    async def write_level2_snippet_rows(self, list_of_rows):
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
        async with self._db_engine.begin() as connection:
            await connection.execute(
                text(self.__insert_template.format(schema_name=SCHEMA_NAME, table_name=self._level2_name)),
                list_of_rows
            )

    async def write_level3_snippet_rows(self, list_of_rows):
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
        async with self._db_engine.begin() as connection:
            await connection.execute(
                text(self.__insert_template.format(schema_name=SCHEMA_NAME, table_name=self._level3_name)),
                list_of_rows
            )

    async def _select_by_name(self, connection, level_table_name, document_name, limit):
        return (
            await connection.execute(
                text(self.__select_by_name_template.format(schema_name= SCHEMA_NAME, table_name=level_table_name)),
                {"document_name": document_name, "limit": limit}
            ).all()
        )

    async def _select_from_level_for_one_embedding(self, connection, level_table_name, query_embedding_list, limit):
        return (
            await connection.execute(
                text(
                    self.__select_by_embedding_template.format(
                        schema_name=SCHEMA_NAME, table_name=level_table_name, embedding_dim=EMBEDDING_DIM
                    )
                ),
                {"limit": limit, "query_embedding": query_embedding_list[0]}
            )
        ).all()

    async def _select_from_level(self, connection, level_table_name, query_embedding_list, limit):
        queries_results = []
        for i in range(len(query_embedding_list)):
            queries_results.extend(
                await self._select_from_level_for_one_embedding(connection, level_table_name, query_embedding_list[i], limit)
            )
        queries_results.sort(key=lambda x: x[SelectIndexes.cos_distance.value])

        return queries_results

    async def select_from_level1(self, query_embedding_list, limit):
        async with self._db_engine.begin() as connection:
            queries_results = await self._select_from_level(connection, self._level1_name, query_embedding_list, limit)

        return queries_results[:limit]

    async def select_from_level2(self, query_embedding_list, limit):
        async with self._db_engine.begin() as connection:
            queries_results = await self._select_from_level(connection, self._level2_name, query_embedding_list, limit)

        return queries_results[:limit]

    async def select_from_level3(self, query_embedding_list, limit):
        async with self._db_engine.begin() as connection:
            queries_results = await self._select_from_level(connection, self._level3_name, query_embedding_list, limit)

        return queries_results[:limit]

    async def _delete_from_table(self, connection, table_name, document_path):
        await connection.execute(
            text(self.__delete_template.format(schema_name=SCHEMA_NAME, table_name=table_name)),
            {"document_path": document_path}
        )

    async def remove_from_level1(self, document_path):
        async with self._db_engine.begin() as connection:
            await self._delete_from_table(connection, self._level1_name, document_path)

    async def remove_from_level2(self, document_path):
        async with self._db_engine.begin() as connection:
            await self._delete_from_table(connection, self._level2_name, document_path)

    async def remove_from_level3(self, document_path):
        async with self._db_engine.begin() as connection:
            await self._delete_from_table(connection, self._level3_name, document_path)

    async def remove_from_all_levels(self, document_path):
        async with self._db_engine.begin() as connection:
            await self._delete_from_table(connection, self._level1_name, document_path)
            await self._delete_from_table(connection, self._level2_name, document_path)
            await self._delete_from_table(connection, self._level3_name, document_path)



