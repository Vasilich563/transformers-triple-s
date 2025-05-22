import urllib.parse as parse
import os
import torch
from flask import Flask, url_for, render_template, request, send_file as flask_send_file, abort
from markupsafe import escape
from transformers import RobertaTokenizerFast
from sqlalchemy import create_engine
from backend.embedding_system.embedding_system import EmbeddingSystem
from backend.transformer.bidirectional_transformer import BidirectionalTransformer
from backend.embedding_system.db_crud import DBCrud
from backend.embedding_system.make_db import make_db
from backend.crawler import observe_directory_daemon


RESULTS_PER_PAGE = 3

tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-large")
vocab_size = len(tokenizer.get_vocab())
max_len = 64
stride = 0
num_layers = 12
d_model = 768
num_attention_heads = 12
d_ffn_hidden = 3072
dropout_p = 0.1
padding_index = tokenizer.pad_token_type_id
device = torch.device("cuda:0")
dtype = torch.float32
embedding_model = BidirectionalTransformer(
    vocab_size, max_len, num_layers, d_model, num_attention_heads, d_ffn_hidden, dropout_p, device, dtype, padding_index
)
checkpoint = torch.load("/home/yackub/PycharmProjects/Diploma/backend/transformer/checkpoint_after_training.pth", map_location=device)
embedding_model.load_state_dict(checkpoint["best_weights"])
embedding_model.eval()

d_model = 1024
from transformers import RobertaModel
embedding_model = RobertaModel.from_pretrained("FacebookAI/roberta-large").to(device)

make_db(d_model)

db_engine = create_engine("postgresql://postgres:ValhalaWithZolinks@localhost:5432/postgres")
db_crud = DBCrud(db_engine)
EmbeddingSystem.class_init(tokenizer, embedding_model, db_crud)

directory_to_check = "/home/yackub/PycharmProjects/Diploma/triple-s-storage"
observe_directory_daemon(directory_to_check)


app = Flask(__name__)


def process_db_select_results(query_results):
    results_to_show = []
    for query_result_row in query_results:
        results_to_show.append(
            {
                "document_name": query_result_row.document_name,
                "document_path": query_result_row.document_path,
                "url_to_get_file": url_for('send_file', path_to_file=parse.quote(query_result_row.document_path, safe='')),
                "snippet": query_result_row.snippet
            }
        )
    return results_to_show


@app.route("/", methods=["GET"])
@app.route("/transformers-triple-s/", methods=["GET"])
@app.route("/transformers-triple-s/main/", methods=["GET"])
def main_page(limit=100, search_by_name_flag=False, exactly_flag=False):
    return render_template(
        "MainPage.html",
        url_for_search_page=url_for('search_page'),
        url_for_main_page=url_for('main_page'),
        url_for_about_page=url_for('about_page'),
        limit=limit,
        search_by_name_flag=search_by_name_flag,
        exactly_flag=exactly_flag
    )


@app.route("/transformers-triple-s/about/", methods=["GET"])
def about_page():
    return render_template("AboutPage.html", url_for_main_page=url_for('main_page'))


@app.route("/transformers-triple-s/search/", methods=["GET", "POST"])
def search_page():
    user_query = request.form.get("user-query").strip()
    limit = int(request.form.get("limit"))
    search_by_name_flag = False if request.form.get("search_by_name_flag") is None else True
    exactly_flag = False if request.form.get("exactly_flag") is None else True


    if not user_query:

        return main_page(limit, search_by_name_flag, exactly_flag)

    result_list = process_db_select_results(
        EmbeddingSystem.handle_user_query(d_model, user_query, search_by_name_flag, exactly_flag, limit)
    )
    # result_list = [
    #     {
    #         "document_name": "Example Search Result Title - This is what a result looks like",
    #         "document_path": "/home/yackub/PycharmProjects/Diploma/frontend/templates/nigga.html",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote("/home/yackub/PycharmProjects/Diploma/frontend/templates/nigga.html", safe='')),
    #         "snippet": "This is a sample search result description. It typically contains a brief excerpt from the webpage that includes your search terms. The relevant words are often highlighted in bold."
    #     },
    #     {
    #         "document_name": "Third Search Result Example",
    #         "document_path": "/home/yackub/PycharmProjects/Diploma/temp/Метрика TF-IDF (Term frequencyinverse document frequency). Loginom Wiki.pdf",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote("/home/yackub/PycharmProjects/Diploma/temp/Метрика TF-IDF (Term frequencyinverse document frequency). Loginom Wiki.pdf", safe='')),
    #         "snippet": "The description here shows how the page content relates to the search terms. Different search engines have different algorithms for selecting which part of the page to display in the snippet."
    #     },
    #     {
    #         "document_name": "Video Result Example",
    #         "document_path": "/home/yackub/PycharmProjects/Diploma/temp/Якубовский_для_диплома.docx",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote("/home/yackub/PycharmProjects/Diploma/temp/Якубовский_для_диплома.docx", safe='')),
    #         "snippet": "This would be a video result. Sometimes special rich snippets are displayed for different types of content like videos, recipes, or products."
    #     },
    #     {
    #         "document_name": "Final Example Search Result",
    #         "document_path": "/home/yackub/PycharmProjects/Diploma/temp/prihod.txt",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote("/home/yackub/PycharmProjects/Diploma/temp/prihod.txt", safe='')),
    #         "snippet": "The last example result in our demonstration. Real search results would typically have 10 items per page with pagination controls to navigate through more results."
    #     },
    #     {
    #         "document_name": "Final Example Search Result",
    #         "document_path": "C:/Users/amis-/PycharmProjects/semantic_search_system/backend/requirements.txt",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote("C:/Users/amis-/PycharmProjects/semantic_search_system/backend/requirements.txt", safe='')),
    #         "snippet": "The last example result in our demonstration. Real search results would typically have 10 items per page with pagination controls to navigate through more results."
    #     },
    #     {
    #         "document_name": "6",
    #         "document_path": "C:/Users/amis-/PycharmProjects/semantic_search_system/frontend/templates/AboutPage.html",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote("C:/Users/amis-/PycharmProjects/semantic_search_system/frontend/templates/AboutPage.html",safe='')),
    #         "snippet": "The last example result in our demonstration. Real search results would typically have 10 items per page with pagination controls to navigate through more results."
    #     },
    #     {
    #         "document_name": "7",
    #         "document_path": "C:/Users/amis-/PycharmProjects/semantic_search_system/refrences/CrossEntropyLoss --- PyTorch 2.6 Documentation.pdf",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote("C:/Users/amis-/PycharmProjects/semantic_search_system/refrences/CrossEntropyLoss --- PyTorch 2.6 Documentation.pdf", safe='')),
    #         "snippet": "The last example result in our demonstration. Real search results would typically have 10 items per page with pagination controls to navigate through more results."
    #     },
    #     {
    #         "document_name": "8",
    #         "document_path": "C:/Users/amis-/Downloads/ddpg.docx",
    #         "url_to_get_file": url_for('send_file', path_to_file=parse.quote(                "C:/Users/amis-/Downloads/ddpg.docx",                safe='')),
    #         "snippet": "The last example result in our demonstration. Real search results would typically have 10 items per page with pagination controls to navigate through more results."
    #     },
        # {
        #     "document_name": "9",
        #     "document_path": "https://www.finalexample.com/blog/post",
        #     "snippet": "The last example result in our demonstration. Real search results would typically have 10 items per page with pagination controls to navigate through more results."
        # }
    # ]
    return render_template(
        "SearchPage.html",
        user_query=escape(user_query),
        url_for_search_page=url_for('search_page'),
        url_for_main_page=url_for('main_page'),
        url_for_about_page=url_for('about_page'),
        limit=limit,
        search_by_name_flag=search_by_name_flag,
        exactly_flag=exactly_flag,
        result_list=result_list,
        results_per_page=RESULTS_PER_PAGE,
        page_index=0
    )


@app.route("/transformers-triple-s/files/<path_to_file>", methods=["GET"])
def send_file(path_to_file):
    path_to_file = parse.unquote(path_to_file)
    if os.path.exists(path_to_file):
        return flask_send_file(path_to_file)
    else:
        print(f"File {path_to_file} not founded")
        abort(404)


