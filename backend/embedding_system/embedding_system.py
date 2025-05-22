from typing import List
import pathlib
import torch
from transformers import RobertaTokenizerFast
from backend.transformer.bidirectional_transformer import BidirectionalTransformer
from backend.embedding_system.snippet_bounds import SnippetBounds
from backend.embedding_system.db_crud import DBCrud

# TODO select, insert and delete should be called using Thread on the higher level

class EmbeddingSystem:
    _tokenizer: RobertaTokenizerFast = None
    _embedding_model: BidirectionalTransformer = None
    _db_crud: DBCrud = None

    _level_1_max_len = 16
    _level_1_stride = 8

    _level_2_max_len = 64
    _level_2_stride = 32

    # _level_3_max_len = 256
    # _level_3_stride = 128

    @classmethod
    def class_init(cls, tokenizer, embedding_model, db_crud: DBCrud):
        cls._tokenizer: RobertaTokenizerFast = tokenizer
        cls._embedding_model: BidirectionalTransformer = embedding_model
        cls._embedding_model.eval()

        cls._db_crud: DBCrud = db_crud


    @staticmethod
    def _get_snippet_bounds(offset_mapping):
        snippet_bounds = []
        for i in range(offset_mapping.shape[0]):
            # offset_mappin[i][j] is array of len(2)
            # offset_mapping[i][0] is always [0, 0] and maps to [CLS] special token
            start = offset_mapping[i][1][0].cpu().item()
            # offset_mapping[i][j] is always [0, 0] and maps to [SEP] special token if j = (offset_mapping.shape[1] - 1)
            j = offset_mapping.shape[1] - 2
            while offset_mapping[i][j][1] == 0 and j > 0:  # j == 0 is maps to [CLS] special token
                j -= 1  # offset_mapping[i][j] = [0, 0] maps to [PAD], but the part of the text is needed
            end = offset_mapping[i][j][1].cpu().item()
            # offset_mapping[i][j][1] is the index of the element after the last element of the window
            # for the last window offset_mapping[i][j][1] == len(original_text)
            snippet_bounds.append(SnippetBounds(start, end))
        return snippet_bounds

    @classmethod
    def _tokenize_text(cls, text, max_len, stride, return_snippet_bounds):
        with torch.no_grad():
            tokenized_text = cls._tokenizer(
                text, padding="max_length", truncation=True, max_length=max_len, stride=stride,
                return_overflowing_tokens=True, return_tensors="pt", return_offsets_mapping=return_snippet_bounds
            )
            if return_snippet_bounds:
                snippet_bounds = cls._get_snippet_bounds(tokenized_text["offset_mapping"])
                return tokenized_text["input_ids"], tokenized_text["attention_mask"], snippet_bounds
            else:
                return tokenized_text["input_ids"], tokenized_text["attention_mask"]

    @classmethod
    def _count_text_embeddings(cls, text_input_ids, text_attention_mask, mean_across_batch):
        with torch.no_grad():
            text_input_ids = text_input_ids.to(cls._embedding_model.device)
            text_attention_mask = text_attention_mask.to(cls._embedding_model.device)

            # text_embedding_batch = cls._embedding_model.forward(text_input_ids, text_attention_mask)

            text_embedding_batch = cls._embedding_model.forward(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
            text_embedding_batch = text_embedding_batch.mean(dim=-2)  # mean across sequence

            if mean_across_batch:
                return [text_embedding_batch.mean(dim=0).detach().cpu().tolist()]
            else:
                return text_embedding_batch.detach().cpu().tolist()


    @staticmethod
    def _windows_before_next_level(next_level_max_len, cur_level_max_len, cur_level_stride):
        return ((next_level_max_len - cur_level_max_len) // cur_level_stride) + 1

    @staticmethod
    def _prepare_rows_for_db(document_text, document_path, snippet_bounds: List[SnippetBounds], embeddings_batch):
        list_of_rows = []
        document_name =  pathlib.Path(document_path).stem  # filename without extension
        for i in range(len(embeddings_batch)):
            snippet_name = document_path + str(i)
            list_of_rows.append({
                "snippet_name": snippet_name,
                "document_path": document_path,
                "document_name": document_name,
                "snippet": document_text[snippet_bounds[i].snippet_start_index: snippet_bounds[i].snippet_end_index],
                "embedding": embeddings_batch[i]
            })
        return list_of_rows

    @staticmethod
    def _prepare_row_for_catalog(document_text, document_path, snippet_bounds: List[SnippetBounds]):
        document_name = pathlib.Path(document_path).stem
        return {
            "document_path": document_path,
            "document_name": document_name,
            "snippet": document_text[snippet_bounds[0].snippet_start_index: snippet_bounds[0].snippet_end_index]
        }

    @classmethod
    def index_new_text(cls, document_text, document_path):
        text_input_ids, text_attention_mask, snippet_bounds = cls._tokenize_text(
            document_text, cls._level_1_max_len, cls._level_1_stride, return_snippet_bounds=True
        )
        text_embeddings = cls._count_text_embeddings(text_input_ids, text_attention_mask, mean_across_batch=False)
        list_of_rows_for_db = cls._prepare_rows_for_db(document_text, document_path, snippet_bounds, text_embeddings)
        cls._db_crud.write_level1_snippet_rows(list_of_rows_for_db)

        cls._db_crud.write_catalog_row(cls._prepare_row_for_catalog(document_text, document_path, snippet_bounds))

        # if text is big enough to place it on the next level too
        if text_input_ids.shape[0] > cls._windows_before_next_level(cls._level_2_max_len, cls._level_1_max_len, cls._level_1_stride):
            text_input_ids, text_attention_mask, snippet_bounds = cls._tokenize_text(
                document_text, cls._level_2_max_len, cls._level_2_stride, return_snippet_bounds=True
            )
            text_embeddings = cls._count_text_embeddings(text_input_ids, text_attention_mask, mean_across_batch=False)
            list_of_rows_for_db = cls._prepare_rows_for_db(document_text, document_path, snippet_bounds, text_embeddings)
            cls._db_crud.write_level2_snippet_rows(list_of_rows_for_db)

            # # if text is big enough to place it on the next level too
            # if text_input_ids.shape[0] > cls._windows_before_next_level(cls._level_3_max_len, cls._level_2_max_len, cls._level_2_stride):
            #     text_input_ids, text_attention_mask, snippet_bounds = cls._tokenize_text(
            #         document_text, cls._level_3_max_len, cls._level_3_stride, return_snippet_bounds=True
            #     )
            #     text_embeddings = cls._count_text_embeddings(text_input_ids, text_attention_mask, mean_across_batch=False)
            #     list_of_rows_for_db = cls._prepare_rows_for_db(document_text, document_path, snippet_bounds, text_embeddings)
            #     cls._db_crud.write_level3_snippet_rows(list_of_rows_for_db)


    @classmethod
    def handle_search_by_name(cls, document_name, limit, exactly_flag):
        return cls._db_crud.select_by_name(document_name, limit, exactly_flag)

    @classmethod
    def handle_user_query(cls, query, search_by_name_flag, exactly_flag, limit=100):
        if search_by_name_flag:
            return cls.handle_search_by_name(query, limit, exactly_flag)
        level = 1
        input_ids, attention_mask = cls._tokenize_text(
            query, cls._level_1_max_len, cls._level_1_stride, return_snippet_bounds=False
        )
        # if query is big the next level is used to find snippet
        if input_ids.shape[0] > cls._windows_before_next_level(cls._level_2_max_len, cls._level_1_max_len, cls._level_1_stride):
            level = 2
            input_ids, attention_mask = cls._tokenize_text(
                query, cls._level_2_max_len, cls._level_2_stride, return_snippet_bounds=False
            )
            # if query is big the next level is used to find snippet
            # if input_ids.shape[0] > cls._windows_before_next_level(cls._level_3_max_len, cls._level_2_max_len, cls._level_2_stride):
            #     level = 3
            #     input_ids, attention_mask = cls._tokenize_text(
            #         query, cls._level_3_max_len, cls._level_3_stride, return_snippet_bounds=False
            #     )

        text_embedding_batch = cls._count_text_embeddings(input_ids, attention_mask, mean_across_batch=False)
        if level == 1:
            result = cls._db_crud.select_from_level1(text_embedding_batch, limit)
        elif level == 2:
            result = cls._db_crud.select_from_level2(text_embedding_batch, limit)
        #else:
        #    result = cls._db_crud.select_from_level3(text_embedding_batch, limit)
        return result

    @classmethod
    def remove_document(cls, document_path):
        cls._db_crud.remove_from_all_levels(document_path)

    @classmethod
    def update_document(cls, document_path, new_text):
        cls._db_crud.remove_from_all_levels(document_path)
        cls.index_new_text(new_text, document_path)












