import flask
from flask import Flask, render_template, request
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
import openai

openai.api_key = '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'


def search_courses(course_description, n=3, pprint=True, graduate=False):
    df = pd.read_csv('courses_cs_all_with_embeddings.csv', converters={
                     "embedding": eval}, encoding='utf-8-sig')
    # print(df.head(2))
    if graduate:
        df = df[df.course_number >= 500]
    course_embedding = get_embedding(
        course_description,
        engine="text-embedding-ada-002"
    )
    # print(course_embedding)
    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, course_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )

    # if pprint:
    #     for r in results:
    #         print(r[:200])
    #         print()
    return results


app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = flask.request.form['query']
    graduate = flask.request.form.get('graduate')
    results = search_courses(query, n=5, pprint=True, graduate=graduate)
    results['similarity'] = results['similarity'].apply(
        lambda x: round(x, 2))
    return flask.render_template('search.html', query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)
