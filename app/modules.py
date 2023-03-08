import json
import logging

import plotly
from flask import Blueprint, render_template, request
from plotly.graph_objs import Bar

from app.utils import load_data

log = logging.getLogger(__name__)
main = Blueprint('main', __name__)

# Load data for app
df, model = load_data(sql_path='./data/DisasterResponse', table_name='response', model_path='./models/model.joblib')


@main.route('/')
@main.route('/index')
def index():
    # calculate 'genre' value counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # calculate 'category' value counts
    category_counts = (df.iloc[:, 4:] != 0).sum().values
    category_names = df.iloc[:, 4:].columns

    # create visuals
    graphs = [{'data': [Bar(x=genre_names,
                            y=genre_counts)],
               'layout': {'title': 'Distribution of Message Genres',
                          'yaxis': {'title': "Count"},
                          'xaxis': {'title': "Genre"}}
               },

              {'data': [Bar(x=category_names,
                            y=category_counts)],
               'layout': {'title': 'Distribution of Message Categories',
                          'yaxis': {'title': "Count"},
                          'xaxis': {'title': "Category", 'tickangle': 35}}
               }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render 'main.html' page with plotly graphs
    return render_template('main.html', ids=ids, graphJSON=graphJSON)


@main.route('/go')
def go():
    # get the query from the input box
    query = request.args.get('query', '')

    # use model to predict classification for query
    res_labels = model.predict([query])[0]
    res_dict = dict(zip(df.columns[4:], res_labels))

    # render 'res.html' page with prediction
    return render_template('res.html', query=query, classification_result=res_dict)
