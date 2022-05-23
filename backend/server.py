from flask import Flask, send_from_directory, request, make_response, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import time, subprocess
import sqlite3 as sql


app = Flask(__name__, static_url_path='')
cors = CORS(app)

conn = sql.connect('analysed_energy.db')
df = pd.read_sql('SELECT * FROM analysed_energy', conn)
df = df.drop('index', axis=1)


@app.route('/reset', methods=["GET"])
def reset_func():
	"""	subprocess.check_output("python3 inferBERT.py && mv new_analysed_energy.db analysed_energy.db", shell=True)
	conn = sql.connect('analysed_energy.db')
	df = pd.read_sql('SELECT * FROM analysed_energy', conn)
	df = df.drop('index', axis=1)"""
	time.sleep(10)
	return 'OK', 200



@app.route('/portal', methods=["GET"])
def basic_func():
	time.sleep(5)
	providers_arr = df['businessUnitDisplayName'].unique().tolist()
	df['month'] = df['publishedDate'].apply(lambda x: datetime.strptime(x.split('T')[0], '%Y-%m-%d').strftime('%B'))
	resp = {'radar': {}, 'treemap': [], 'stacked': {}, 'chart_basic': {'data': {}}, 'multipleYaxis': {}, 'multipleYaxis_stars': {}}
	for p in providers_arr:
		resp['radar'][p] = {}
		sum_rev = df.loc[df['businessUnitDisplayName']==p].shape[0]
		radar_dimensions = df.loc[df['businessUnitDisplayName']==p]['criteria_str'].unique().tolist()
		resp['radar'][p]['metadata'] = [tr[:12] for tr in radar_dimensions]
		resp['radar'][p]['data'] = []
		for c in radar_dimensions:
			resp['radar'][p]['data'].append(df.loc[(df['businessUnitDisplayName']==p) & (df['criteria_str']==c)].shape[0])


	treemap_dimensions = df['criteria_str'].unique().tolist()
	for c in treemap_dimensions:
		resp['treemap'].append({'x': c, 'y': df.loc[df['criteria_str']==c].shape[0]})

	categories_arr = df['month'].unique().tolist()
	categories_arr = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
	for p in providers_arr:
		tmp_ = []
		for c in categories_arr:
			tmp_.append(df.loc[(df['businessUnitDisplayName']==p) & (df['month']==c)].shape[0])
		resp['stacked'][p] = tmp_

	resp['chart_basic']['criteria'] = [l for l in radar_dimensions]
	for p in providers_arr:
		tmp_ = []
		for c in radar_dimensions:
			x = df.loc[(df['businessUnitDisplayName']==p) & (df['criteria_str']==c)]['predictions'].tolist()
			tmp_.append(int(100*sum(x)/len(x)))
		resp['chart_basic']['data'][p] = tmp_

	for p in providers_arr:
		tmp_ = []
		for c in categories_arr:
			x = df.loc[(df['businessUnitDisplayName']==p) & (df['month']==c)]['predictions'].tolist()
			if len(x):
				tmp_.append(int(100*sum(x)/len(x)))
			else:
				tmp_.append(50)
		resp['multipleYaxis'][p] = tmp_


	for p in providers_arr:
		tmp_ = []
		for c in categories_arr:
			x = df.loc[(df['businessUnitDisplayName']==p) & (df['month']==c) & (df['rating'] > 3)].shape[0]
			y = df.loc[(df['businessUnitDisplayName']==p) & (df['month']==c)].shape[0]
			if y:
				tmp_.append(int(100*x/y))
			else:
				tmp_.append(50)
		resp['multipleYaxis_stars'][p] = tmp_

	return resp, 200




@app.route('/')
def index():
	return "I'm only an empty endpoint"


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=8080, debug=True)
