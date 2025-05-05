from flask import Flask, url_for, render_template, request
from markupsafe import escape


app = Flask(__name__)



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
    print(request.form)
    user_query = request.form.get("user-query").strip()
    limit = request.form.get("limit", int)
    search_by_name_flag = False if request.form.get("search_by_name_flag") is None else True
    exactly_flag = False if request.form.get("exactly_flag") is None else True


    if not user_query:

        return main_page(limit, search_by_name_flag, exactly_flag)

    # TODO handle user query

    return render_template(
        "SearchPage.html",
        user_query=escape(user_query),
        url_for_search_page=url_for('search_page'),
        url_for_main_page=url_for('main_page'),
        url_for_about_page=url_for('about_page'),
        limit=limit,
        search_by_name_flag=search_by_name_flag,
        exactly_flag=exactly_flag
    )