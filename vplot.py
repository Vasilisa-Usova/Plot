from qsweepy import*
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import webcolors

#import exdir
#from data_structures import *
import plotly.graph_objs as go
from pony.orm import *
#from database import database
from plotly import*
from cmath import phase
#from datetime import datetime
from dash.dependencies import Input, Output, State
import pandas as pd
import psycopg2
import pandas.io.sql as psql
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, static_folder='static')
app.config['suppress_callback_exceptions']=True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
db = database.database()

			
direct_db = psycopg2.connect(database='qsweepy', user='qsweepy', password='qsweepy')
default_query = 'SELECT * FROM data;'

def string_to_list(string):
	if string == '': return []
	position = string.find(',')
	list_res = []
	while position > -1:
		#print(position, 'and', string[:10], 'and', string[position+1:])
		list_res.append(string[:position])
		string = string[position+2:]
		position = string.find(',')
	if string != '': list_res.append(string)
	return list_res

def data_to_dict(data):
		return { 'id': data.id,
				 #comment': data.comment,
				 'sample_name': data.sample_name,
				'time_start': data.time_start,
				 'time_stop': data.time_stop,
				 #'filename': data.filename,
				 'type_revision': data.type_revision,
				 'incomplete': data.incomplete,
				 'invalid': data.invalid,
				 'owner': data.owner,
				} 

def generate_table(dataframe, max_rows=10):
	return html.Table(
			# Header
			[html.Tr([html.Th(col) for col in dataframe.columns])] +

			# Body
			[html.Tr([
				html.Td(str(dataframe.iloc[i][col])) for col in dataframe.columns
			]) for i in range(min(len(dataframe), max_rows))])
	
meas_ids = ''
fit_ids = ''
dim_2 = False

layout = {}
figure = {}

def measurement_table():
	return dash_table.DataTable(id="meas-id", columns=[{'id': 'id', 'name':'id'}, {'id': 'label', 'name':'label'}], data=[], editable=True, row_deletable=True)

@app.callback(
	Output(component_id="meas-id", component_property="data"),
	[Input(component_id="query-results-table", component_property="derived_virtual_data"),
    Input(component_id="query-results-table", component_property="derived_virtual_selected_rows")],
	state=[State(component_id="meas-id", component_property="derived_virtual_data")]
)
def render_measurement_table(query_results, query_results_selected, current_measurements):
	print ('render_measurement_table called')
	#current_measurements = []
	if current_measurements is None:
		return []
	
	print(query_results, query_results_selected, current_measurements)
	selected_measurement_ids = [query_results[measurement]['id'] for measurement in query_results_selected]
	deselected_measurement_ids = [measurement['id'] for measurement in query_results if not measurement['id'] in selected_measurement_ids]
	old_measurement_ids = [measurement['id'] for measurement in current_measurements]
	old_measurements = [measurement for measurement in current_measurements if not measurement['id'] in deselected_measurement_ids]
	new_measurements = [{'id':query_results[measurement]['id'], 
						 'label': (query_results[measurement]['label'] if 'label' in query_results[measurement] else query_results[measurement]['id'])} 
							for measurement in query_results_selected if (not query_results[measurement]['id'] in old_measurement_ids)]
	
	return old_measurements+new_measurements

def available_traces_table(data=[], column_static_dropdown=[], column_conditional_dropdowns=[]):
	#print (column_static_dropdown, column_conditional_dropdowns)
	return dash_table.DataTable(id="available-traces-table", columns=[{'id': 'id', 'name': 'id'}, 
																	  {'id': 'dataset', 'name': 'dataset'}, 
																	  {'id': 'op', 'name': 'op'}, 
																	  {'id': 'style', 'name': 'style', 'presentation':'dropdown'},
																	  {'id': 'color', 'name': 'color', 'presentation':'dropdown'},
																	  {'id': 'x-axis', 'name': 'x-axis', 'presentation':'dropdown'},
																	  {'id': 'y-axis', 'name': 'y-axis', 'presentation':'dropdown'},
																	  {'id': 'row', 'name':'row'},
																	  {'id': 'col', 'name':'col'}], data=data, editable=True, row_selectable='multi', selected_rows=[], column_static_dropdown=column_static_dropdown, column_conditional_dropdowns=column_conditional_dropdowns)	
																
@app.callback(
	Output(component_id="available-traces-container", component_property="children"),
	[Input(component_id="meas-id", component_property="derived_virtual_data")],
	state=[State(component_id="available-traces-table", component_property="derived_virtual_data")]
	)
def render_available_traces_table(loaded_measurements, current_traces):
	# traverse loaded measurements and add all datasets
	with db_session:
		data = []
		#x_axis_conditional_dropdowns = []
		conditional_dropdowns = []
		#y_axis_conditional_dropdowns = []
		colors = [c for c in webcolors.CSS3_NAMES_TO_HEX.keys()]
		styles = ['2d', '-', '.', 'o']
		for m in loaded_measurements:
			measurement_id = m['id']
			measurement_state = save_exdir.load_exdir(db.Data[int(measurement_id)].filename, db)
			for dataset in measurement_state.datasets.keys():
				parameter_names = [p.name for p in measurement_state.datasets[dataset].parameters]
				#dropdown_row_condition = 'id eq "{}" and dataset eq "{}"'.format(measurement_id, dataset)
				dropdown_row_condition = 'dataset eq "{}"'.format(dataset)
				#y_dropdown_row_condition = 'dataset eq "{}"'.format(dataset)
				if np.iscomplexobj(measurement_state.datasets[dataset].data): # if we are dealing with complex object, give the chance of selecting which op we want to apply
					operations = ['Re', 'Im', 'Abs', 'Ph']
				else:
					operations = ''
				conditional_dropdowns.append({'condition':dropdown_row_condition, 'dropdown':[{'label': p, 'value': p} for p in parameter_names]+[{'label':'data', 'value':'data'}]})
				#y_axis_conditional_dropdowns.append({'condition':y_dropdown_row_condition, 'dropdown':[{'label': p, 'value': p} for p in parameter_names]+[{'label':'data', 'value':'data'}]})
				for operation in operations:
					row = {'id': measurement_id, 
						   'dataset': dataset, 
						   'op':operation, 
						   'style':'-', 
						   'color': 'black', 
						   'x-axis': parameter_names[0] if len(parameter_names) > 0 else 'data', 
						   'y-axis': parameter_names[0] if len(parameter_names) > 1 else 'data', #[1]?
						   'row': 0, 
						   'col': 0}
					data.append(row)
				
						
				# check if dataset is in current traces, if not, update the cell values with current values
		#print (data)
		return available_traces_table(data, [{'id': 'style', 'dropdown': [{'label':s, 'value': s} for id, s in enumerate(styles)]},
											 {'id': 'color', 'dropdown': [{'label':c, 'value': c} for c in colors]}],
											[{'id': 'x-axis', 'dropdowns': conditional_dropdowns}, 
											 {'id': 'y-axis', 'dropdowns': conditional_dropdowns}])
	
def app_layout():
	return html.Div(children=[html.Div(id="modal-select-measurements", className= "modal",  style={'display':'none'}, children=modal_content()),
		html.Div([
			html.H1(id = 'list_of_meas_types', style = {'fontSize': '30', 'text-align': 'left', 'text-indent': '5em'}),
			dcc.Graph(id = 'live-plot-these-measurements', style={'height':'100%', 'width': '70%'})],
				style = {'position': 'absolute', 'width': '100%', 'height': '100%'}), #style = {'position': 'absolute', 'top': '30', 'left': '30', 'width': '1500' , 'height': '1200'}),
		html.Div([
			#html.H2(children = 'Measurements', style={'fontSize': 25}),
			html.Div(id = 'table_of_meas'),
			#html.Div(html.P('Started at: ' + str(start)), style={'fontSize': 14}),
			#html.Div(html.P('Stopped at: ' + str(stop)), style={'fontSize': 14}),
			#html.Div(html.P('Owner: ' + str(owner)), style={'fontSize': 14}),
			#html.Div(html.P('Metadata: ' + met_arr), style={'fontSize': 14}),
			html.H3(children = 'Measurement info', style={'fontSize': 25}),
			html.Div(id = 'dropdown', style = {'width': '100'}),
			html.Div(id = 'meas_info'),
			html.Div([html.P('Measurements: '), measurement_table()]),
			html.Button(id="modal-select-measurements-open", children=["Add measurements..."]),
			html.Div(children=[html.P('Available traces: '), html.Div(id='available-traces-container', children=[available_traces_table()])]),
					  #dcc.Input(id='meas-id2', value = str(meas_ids), type = 'string')]), 
			#html.Div([html.P('You chose following fits: '), dcc.Input(id='fit-id', value = str(fit_ids), type = 'string')]),
			],
					 style={'position': 'absolute', 'top': '5%', 'left': '68%', 'width': '30%' , 'height': '80%',#'position': 'absolute', 'top': '80', 'left': '1500', 'width': '350' , 'height': '800',
							'padding': '0px 10px 15px 10px',
							  'marginLeft': 'auto', 'marginRight': 'auto', #'background': 'rgba(167, 232, 170, 1)',
							'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'},#rgba(190, 230, 192, 1)'},
		   ),
		   dcc.Interval(
				id='interval-component',
				interval=1*1000, # in milliseconds
				n_intervals=0
			),
			#html.Div([html.Div(id='my-div', style={'fontSize': 14}), dcc.Input(id='meas-id', value = str(meas_ids), type='string')])]),
		html.Div(id='intermediate-value-meas', style={'display': 'none'}),
		html.Div(id='intermediate-value-fit', style={'display': 'none'}),
		#dcc.Input(id='meas-id', value = str(meas_ids), style={}),
		html.Div([html.H4(children='References', style={'fontSize': 25}), html.Div(id = 'table_of_references')], style = {'position': 'absolute', 'top': '1100', 'left': '50'})
		#(html.Div(a) for a in state.metadata.items())
	])

@app.callback(
	Output(component_id = 'meas_info', component_property = 'children'),
	[Input(component_id = 'my_dropdown', component_property='value')])
def write_meas_info(value):
	with db_session:
		if value == None: return 
		state = save_exdir.load_exdir(db.Data[int(value)].filename ,db)
		met_arr = ''
		for k in state.metadata:
			if k[0] != 'parameter values': met_arr += str(k[0]) + ' = ' + str(k[1]) + '\n '
		#print(met_arr)
		return (html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 14}),
						html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 14}),
						html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 14}),
						html.Div(html.P('Metadata: ' + met_arr), style={'fontSize': 14}))
					
@app.callback(
	Output(component_id = 'dropdown', component_property = 'children'),
	[Input(component_id = 'meas-id', component_property='derived_virtual_data')])
	#[Input(component_id = 'meas-id2', component_property='value')])
def create_dropdown(meas_ids):
	print ('create_dropdown called with meas_ids', meas_ids)
	meas_ids_list = pd.DataFrame(meas_ids, columns=['id', 'label'])['id'].tolist()
	print ('returned list: ', meas_ids_list)
	#meas_ids_list = meas_ids.split(',')
	return dcc.Dropdown(id = 'my_dropdown', options = [{'label': str(i), 'value': str(i)} for i in meas_ids_list])

#@app.callback(
	#Output(component_id = 'table_of_meas', component_property = 'children'),
	#[Input(component_id = 'meas-id', component_property='value')])
# def generate_table_of_meas(meas_ids):
	# with db_session:
		# ids = []
		# type_ref = []
		# meas_accordance_to_references = []
		# df = pd.DataFrame()
		# if meas_ids != '':
			# for i in string_to_list(meas_ids):
				# state = save_exdir.load_exdir(db.Data[int(i)].filename, db)
				# df = df.append({'id': state.id, 'owner': state.owner}, ignore_index=True)
		# return generate_table(df)

@app.callback(
	Output(component_id = 'table_of_references', component_property = 'children'),
	[Input(component_id = 'meas-id', component_property='derived_virtual_data')])
	#[Input(component_id = 'meas-id2', component_property='value')])
def generate_table_of_references(meas_ids):
	print ('generate_table_of_references called with meas_ids', meas_ids)
	meas_ids_list = pd.DataFrame(meas_ids, columns=['id', 'label'])['id'].tolist()
	print ('returned list: ', meas_ids_list)
	#meas_ids_list = meas_ids.split(',')
	with db_session:
		meas_accordance_to_references = []
		df = pd.DataFrame()
		if meas_ids != '':
			for i in meas_ids_list:
				ids = []
				type_ref = []
				state_references = save_exdir.load_exdir(db.Data[int(i)].filename, db).references.items()
				for index, type_r in state_references:
					ids.append(index)
					#id_keys.update({index: i})
					type_ref.append(type_r)
				query_for_table = select(c for c in db.Data if (c.id in ids))
				df_new = pd.DataFrame()
				df_new = pd.DataFrame(list(data_to_dict(x) for x in list(query_for_table)))
				if not df_new.empty:
					df_new = df_new.assign(reference_type=pd.Series(type_ref), ignore_index=True)
					df_new = df_new.assign(measurement = pd.Series(np.full(len(ids), i)), ignore_index=True)
					df = df.append(df_new, ignore_index=True)
					#for i in df_new['id']:
						#meas_accordance_to_references.append(id_keys.get(i))
		#if not df.empty:
			#df = df.assign(reference_type=pd.Series(type_ref), ignore_index=True)
			#df = df.assign(measurement = pd.Series(np.full(len(ids), i)), ignore_index=True)
			#df = df.assign(measurement = pd.Series(meas_accordance_to_references), ignore_index=True)
		#print(df)
		return generate_table(df)
@app.callback(
	Output(component_id = 'list_of_meas_types', component_property = 'children'),
	[Input(component_id = 'meas-id', component_property='derived_virtual_data')])
	#[Input(component_id = 'meas-id2', component_property='value')])
def add_meas_type(meas_ids):
	print ('add_meas_type called with meas_ids', meas_ids)
	with db_session:
		list_of_states = pd.DataFrame(meas_ids, columns=['id', 'label'])['id'].tolist()
		print ('returned list: ', list_of_states)
		#meas_ids_list = meas_ids.split(',')
		list_of_meas_types = []
		for i in list_of_states:
			if (save_exdir.load_exdir(db.Data[int(i)].filename, db)).measurement_type not in list_of_meas_types:
				if list_of_meas_types != []: list_of_meas_types.append(', ') 
				list_of_meas_types.append((save_exdir.load_exdir(db.Data[int(i)].filename)).measurement_type) 
		return list_of_meas_types
			
@app.callback(
	Output(component_id='intermediate-value-meas', component_property='children'),
	[Input(component_id = 'meas-id', component_property='derived_virtual_data')])
	#[Input(component_id = 'meas-id2', component_property='value')])
def add_meas(meas_ids):
	print ('add_meas called with meas_ids', meas_ids)
	print ('returned list: ', pd.DataFrame(meas_ids, columns=['id', 'label'])['id'].tolist())
	return pd.DataFrame(meas_ids, columns=['id', 'label'])['id'].tolist()
	return meas_ids.split(',')
	
#@app.callback(
#	Output(component_id='intermediate-value-fit', component_property='children'),
#	[Input(component_id='fit-id', component_property='value')])
#def add_meas2(input_value):
#	return string_to_list(str(input_value))

# @app.callback(Output('live-plot-these-measurements', 'figure'),
			  # [#Input('intermediate-value-fit', 'children'), 
			  # #Input('intermediate-value-meas', 'children')])
			  # [Input(component_id="available-traces-table", component_property="derived_virtual_data"),
			   # Input(component_id="available-traces-table", component_property="derived_virtual_selected_rows")])
# #def plot_these_measurements(fit_ids_saved, meas_ids_saved): 
# def plot(all_traces, selected_trace_ids):
	# # load all measurments
	# all_traces = pd.DataFrame(all_traces, columns=['id', 'dataset', 'op', 'style', 'color', 'x-axis', 'y-axis', 'row', 'col'])
	# selected_traces = all_traces[selected_trace_ids]
	# measurements_to_load = selected_traces['id'].unique()
	# measurements = []
	# # load measurements
	# with db_session:
		# for measurement_id in measurements_to_load:
			# measurements[measurement_id] = save_exdir.load_exdir(db.Data[int(measurement_id)].filename, db)
	
	# layout = {}
	# figure = {}
	# layout['height'] = 1000
	# layout['annotations'] = []
	# layout['width'] = 1500
	# layout['showlegend'] = False
	# figure['data'] = []
	
	# # building subplot grid
	# num_rows = all_traces['row'].max()+1
	# num_cols = all_traces['col'].max()+1
	
	# layout
	
		# #print(measurement_to_plot)
		# #print(meas_ids_saved)
		# measurement_to_fit = {}
		# measurement_to_plot = {}
		# if meas_ids_saved != '':
			# for index, i in enumerate(meas_ids_saved):
				# state = save_exdir.load_exdir(db.Data[int(i)].filename, db)
				# measurement_to_plot.update({str(index):state})#({db.Data[int(i)].sample_name: state})
		# if fit_ids_saved != '':
			# for index, i in enumerate(fit_ids_saved):
				# state = save_exdir.load_exdir(db.Data[int(i)].filename, db)
				# measurement_to_fit.update({str(index): state})#({db.Data[int(i)].sample_name: state})

		# #print('I am working ', n_intervals)
		# layout['height'] = 1000
		# layout['annotations'] = []
		# layout['width'] = 1500
		# layout['showlegend'] = False
		# figure['data'] = []
		# if dim_2: type = 'heatmap' ### 
		# else: type = 'scatter'
		# number_of_qubits = len(measurement_to_plot)
		# if number_of_qubits < 3: layout['height'] = 900
		# for qubit_index, qubit in enumerate(measurement_to_plot.keys()):
			# state = measurement_to_plot[qubit]
			# for i, key in enumerate(state.datasets.keys()):
				# number_of_datasets = len(state.datasets.keys())
				# #print(state.datasets[key].parameters[0].values, state.datasets[key].parameters[1].values, state.datasets[key].data)
				# if (number_of_datasets == 1) and (number_of_qubits < 3): layout['width'] = 1000
				# number = number_of_qubits*number_of_datasets
				# index = i + qubit_index
				# layout['xaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [index/number, (index + 0.8)/number], 
									# 'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #RE
				# layout['yaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [0, 0.45], 
									# 'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
				# layout['xaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [index/number, (index + 0.8)/number],
									# 'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #IM
				# layout['yaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [0.55, 1], 
									# 'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
				# dataset = state.datasets[key]
				# figure['data'].append({'colorbar': {'len': 0.4,
									   # 'thickness': 0.025,
									   # 'thicknessmode': 'fraction',
									   # 'x': (index + 0.8)/number,
									   # 'y': 0.2},
						  # 'type': type,
						  # 'mode': 'markers' if not dim_2 else '',
						  # 'uid': '',
						  # 'xaxis': 'x' + str((index + 1)*2),
						  # 'yaxis': 'y' + str((index + 1)*2),
						  # 'x': np.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
						  # 'y': np.memmap.tolist(np.imag(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
						  # 'z': np.memmap.tolist(np.imag(dataset.data))})
				# layout['annotations'].append({'font': {'size': 16},
									# 'showarrow': False,
									# 'text': str(state.id) + ': Re(' + key + ')',
									# 'x': (index + 0.4)/number,
									# 'xanchor': 'center',
									# 'xref': 'paper',
									# 'y': 1,
									# 'yanchor': 'bottom', 'yref': 'paper'})
				# figure['data'].append({'colorbar': {'len': 0.4,
									   # 'thickness': 0.025,
									   # 'thicknessmode': 'fraction',
									   # 'x': (index + 0.8)/number,
									   # 'y': 0.8},
						  # 'type': type,
						  # 'mode': 'markers' if not dim_2 else '',
						  # 'uid': '',
						  # 'xaxis': 'x' + str((index + 1)*2 + 1),
						  # 'yaxis': 'y' + str((index + 1)*2 + 1),
						  # 'x': np.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
						  # 'y': np.memmap.tolist(np.real(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
						  # 'z': np.memmap.tolist(np.real(dataset.data))})
				# layout['annotations'].append({'font': {'size': 16},
									# 'showarrow': False,
									# 'text': str(state.id) + ': Im(' + key + ')',
									# 'x': (index + 0.4)/number,
									# 'xanchor': 'center',
									# 'xref': 'paper',
									# 'y': 0.45,
									# 'yanchor': 'bottom', 'yref': 'paper'})  
				# if (len(fit_ids_saved) > 0) and (qubit in measurement_to_fit.keys()):
					# fit_state = measurement_to_fit[qubit]
					# for key in fit_state.datasets.keys(): 
						# figure['data'].append({'colorbar': {'len': 0.4,
										   # 'thickness': 0.025,
										   # 'thicknessmode': 'fraction',
											# 'x': (index + 0.8)/number,
										   # 'y': 0.2},
							  # 'type': type,
							  # 'mode': 'lines' if not dim_2 else '',
							  # 'uid': '',
							  # 'xaxis': 'x' + str((index + 1)*2),
							  # 'yaxis': 'y' + str((index + 1)*2),
							  # 'x': np.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
							  # 'y': np.memmap.tolist(np.imag(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
							  # 'z': np.memmap.tolist(np.imag(dataset.data))})
						# figure['data'].append({'colorbar': {'len': 0.4,
										   # 'thickness': 0.025,
										   # 'thicknessmode': 'fraction',
										   # 'x': (index + 0.8)/number,
										   # 'y': 0.8},
							  # 'type': type,
							  # 'mode': 'lines' if not dim_2 else '',
							  # 'uid': '',
							  # 'xaxis': 'x' + str((index + 1)*2 + 1),
							  # 'yaxis': 'y' + str((index + 1)*2 + 1),
							  # 'x': np.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
							  # 'y': np.memmap.tolist(np.real(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
							  # 'z': np.memmap.tolist(np.real(dataset.data))})
		# figure['layout'] = layout
		# return figure 

@app.callback(Output('live-plot-these-measurements', 'figure'),
				[Input(component_id="available-traces-table", component_property="derived_virtual_data"), Input('interval-component', 'n_intervals')])
				#[Input(component_id="available-traces-container", component_property="children")])
			  #[Input('intermediate-value-fit', 'children'), Input('intermediate-value-meas', 'children')])
def plot_these_measurements(info, n_intervals):#meas_ids_saved, fit_ids_saved): 
	list_of_files_to_load = []
	num_of_cols = 0
	num_of_rows = 0
	for i in range(len(info)):
		if info[i]['id'] not in list_of_files_to_load:
			list_of_files_to_load.append(info[i]['id'])
		if int(info[i]['col']) >  num_of_cols: num_of_cols = int(info[i]['col'])
		if int(info[i]['row']) >  num_of_rows: num_of_rows = int(info[i]['row'])
	#print('I will load this, axaxa')
	#print(list_of_files_to_load)
	with db_session:
		measurement_to_plot = {}
		if list_of_files_to_load != []:
			for index in list_of_files_to_load:
				state = save_exdir.load_exdir(db.Data[index].filename, db)
				if str(index) not in measurement_to_plot.keys():
					measurement_to_plot.update({str(index):state})#({db.Data[int(i)].sample_name: state})
		layout = {}
		#layout['height'] = 1000
		layout['annotations'] = []
		#layout['width'] = 1500
		layout['showlegend'] = False
		figure['data'] = []
		number_of_qubits = len(info)
		#if number_of_qubits < 3: layout['height'] = '80%'#900
		# print('Your params: ', num_of_cols, ', ', num_of_cols)
		# print(info)
		for row in range(num_of_rows):
			for column in range(num_of_cols):
				#layout['xaxis' + str(row*num_of_rows + column + 1)] = {'anchor': 'x' + str(row*num_of_rows + column + 1), 'domain': [column/num_of_cols, (column + 0.8)/num_of_cols],} 
				#layout['yaxis' + str(row*num_of_rows + column + 1)] = {'anchor': 'x' + str(row*num_of_rows + column + 1), 'domain': [row/num_of_rows, (row + 0.8)/num_of_rows], }
				layout['xaxis' + str(row + 1) + str(column + 1)] = {'anchor': 'y' + str(row + 1) + str(column + 1), 'domain': [(column+0.2)/num_of_cols, (column + 0.8)/num_of_cols],} 
				layout['yaxis' + str(row + 1) + str(column + 1)] = {'anchor': 'x' + str(row + 1) + str(column + 1), 'domain': [(row+0.2)/num_of_rows, (row + 0.8)/num_of_rows], }
				print('xaxis', str(row + 1) + str(column + 1), ': ', (column+0.2)/num_of_cols, (column + 0.8)/num_of_cols)
				print('yaxis', str(row + 1) + str(column + 1), ': ', (row+0.2)/num_of_rows, (row + 0.8)/num_of_rows)
		# {'id': 3, 'dataset': 'random', 'op': 'Im', 'style': '-', 'color': 'black', 'x-axis': 'randomize', 'y-axis': 'Voltage', 'row': 0, 'col': 0}
		if (num_of_cols > 0) and (num_of_rows > 0):
			for i in range(len(info)):
				key = info[i]['dataset']
				state = measurement_to_plot[str(info[i]['id'])] 
				dataset = state.datasets[key]
				#dataset_x = state.datasets[key].parameters[int(info[i]['x-axis'])]
				#dataset_y = state.datasets[key].parameters[int(info[i]['y-axis'])]
				x_axis_id = -1
				y_axis_id = -1
				title_x = key
				title_y = key
				for itt, itt_param in enumerate(state.datasets[key].parameters):
					print(itt, itt_param)
					if itt_param.name == info[i]['x-axis']: 
						dataset_x = np.memmap.tolist(state.datasets[key].parameters[itt].values)
						title_x = state.datasets[key].parameters[itt].name + ', ' + state.datasets[key].parameters[itt].unit
						x_axis_id = itt
					if itt_param.name == info[i]['y-axis']: 
						dataset_y = np.memmap.tolist(state.datasets[key].parameters[itt].values)
						title_y = state.datasets[key].parameters[itt].name + ', ' + state.datasets[key].parameters[itt].unit
						y_axis_id = itt
				#new_shape = [i for i in state.datasets[key].data.shape]
				#new_shape[]
				if info[i]['style'] != '2d':
					#new_axis_order = np.arange(len(state.datasets[key].data.shape))
					data_axis_id = x_axis_id if not x_axis_id == -1 else y_axis_id
					new_axis_order = [data_axis_id]+[i for i in range(len(state.datasets[key].data.shape)) if i != data_axis_id]
					print ('1d:', state.datasets[key].data.shape, new_axis_order, (state.datasets[key].data.shape[data_axis_id], -1))
					trace = np.reshape(np.transpose(state.datasets[key].data, tuple(new_axis_order)), (state.datasets[key].data.shape[data_axis_id]))
				else:
					new_axis_order = [x_axis_id, y_axis_id] + [i for i in range(len(state.datasets[key].data.shape)) if i != x_axis_id and i != y_axis_id]
					print ('2d:', state.datasets[key].data.shape, new_axis_order, (state.datasets[key].data.shape[x_axis_id], state.datasets[key].data.shape[y_axis_id], -1))
					#trace = np.reshape(state.datasets[key].data, (state.datasets[key].data.shape[x_axis_id], state.datasets[key].data.shape[y_axis_id], -1))[:,:,0]
					trace = np.reshape(np.transpose(state.datasets[key].data, tuple(new_axis_order)), (state.datasets[key].data.shape[x_axis_id], state.datasets[key].data.shape[y_axis_id], -1))[:,:,0]
					#x = dataset_x
					#y = dataset_y
				if info[i]['op'] == 'Im': data_to_plot = np.imag(trace)
				if info[i]['op'] == 'Re': data_to_plot = np.real(trace)
				if info[i]['op'] == 'Abs': data_to_plot = np.abs(trace)
				if info[i]['op'] == 'Ph': data_to_plot = np.angle(trace)
				x = dataset_x if x_axis_id != -1 else data_to_plot
				y = dataset_y if y_axis_id != -1 else data_to_plot
				print ('new trace shape:', trace.shape, 'x shape:',np.asarray(x).shape, 'y shape:', np.asarray(y).shape)
				
				#print(info[i]['y-axis'])
				#if info[i]['y-axis'] == 'data': 
				#	dataset.data = np.reshape(np.memmap.tolist(state.datasets[key].data), np.product(np.shape(state.datasets[key].data)))
				#	title_y = key
				#elif np.shape(dataset.data)[0] != len(dataset_x):
				#	dataset.data = np.transpose(dataset.data)
				#print(dataset_x)#, dataset_y[:10], np.shape(dataset.data))
				row = int(info[i]['row']) - 1
				column = int(info[i]['col']) - 1
				if info[i]['style'] == '-': style = 'lines'
				elif info[i]['style'] == 'o': style = 'markers'
				elif info[i]['style'] == '.': style = 'markers'
				else: style = '2d'

				if (row >= 0) and (column >= 0):
					
					figure['data'].append({'colorbar': {'len': 0.6/num_of_rows,
									   'thickness': 0.025/num_of_cols,
									   'thicknessmode': 'fraction',
									   'x': (column + 0.8)/num_of_cols,
									   'y': (row + 0.5)/num_of_rows},
							  'type': 'heatmap' if style == '2d' else 'scatter',
							  'colorscale': 'Viridis',
							  'mode': style,
							  'marker': {'size': 5 if info[i]['style'] == 'o' else 2},
							  'color': info[i]['color'],
							  'xaxis': 'x' + str(row + 1) + str(column + 1),
							  'yaxis': 'y' + str(row + 1) + str(column + 1),
							  # 'x': np.memmap.tolist(dataset_x),
							  # 'y': np.memmap.tolist(dataset_data) if info[i]['style'] != '2d' else np.memmap.tolist(dataset_y),
							  # 'z': np.memmap.tolist(data_to_plot)})
							  'x': x,#np.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
							  'y': y, #np.memmap.tolist(data_to_plot) if info[i]['style'] != '2d' else np.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
							  'z': data_to_plot}) #np.memmap.tolist(data_to_plot)})
					# already_plotted = {}
					# for itteration in layout['annotations']:
						# if ((column + 0.5)/num_of_cols == itteration['x']) and ((row + 0.8)/num_of_rows == itteration['y']):
							# already_plotted = itteration
					# if already_plotted != {}:
					layout['annotations'].append({'font': {'size': 16},
								'showarrow': False,
								'text': str(info[i]['id']) + ': ' + info[i]['op'] + '(' + key + ')',
								'x': (column + 0.5)/num_of_cols,
								'xanchor': 'center',
								'xref': 'paper',
								'y': (row + 0.8)/num_of_rows,
								'yanchor': 'bottom', 'yref': 'paper'})  
					# else:
						# layout['annotations'].append(already_plotted)
						# layout['annotations'][len(layout['annotations'])-1]['text'] += str(info[i]['id']) + ': ' + info[i]['op'] + '(' + key + ')',
					layout['xaxis' + str(row + 1) + str(column + 1)].update({'title': title_x})
					layout['yaxis' + str(row + 1) + str(column + 1)].update({'title': title_y})
					#layout['xaxis' + str(row + 1) + str(column + 1)].update({'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit})
					#layout['yaxis' + str(row + 1) + str(column + 1)].update({'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit})
					##layout['xaxis' + str(row + 1) + str(column + 1)].update({'titlefont': {'family': 'Cailbri', 'size': str(int(20 - max(num_of_cols, num_of_rows)))}})
					##layout['yaxis' + str(row + 1) + str(column + 1)].update({'titlefont': {'family': 'Cailbri', 'size': str(int(20 - max(num_of_cols, num_of_cols)))}})
					print(row, column, 'x' + str(str(row + 1) + str(column + 1)), 'y' + str(str(row + 1) + str(column + 1)))
		# for qubit_index, qubit in enumerate(measurement_to_plot.keys()):
			# state = measurement_to_plot[qubit]
			# for i, key in enumerate(state.datasets.keys()):
				# number_of_datasets = len(state.datasets.keys())
				# #print(state.datasets[key].parameters[0].values, state.datasets[key].parameters[1].values, state.datasets[key].data)
				# if (number_of_datasets == 1) and (number_of_qubits < 3): layout['width'] = 1000
				# number = number_of_qubits*number_of_datasets
				# index = i + qubit_index
				# layout['xaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [index/number, (index + 0.8)/number], 
									# 'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #RE
				# layout['yaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [0, 0.45], 
									# 'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
				# layout['xaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [index/number, (index + 0.8)/number],
									# 'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #IM
				# layout['yaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [0.55, 1], 
									# 'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
				# dataset = state.datasets[key]
				# figure['data'].append({'colorbar': {'len': 0.4,
									   # 'thickness': 0.025,
									   # 'thicknessmode': 'fraction',
									   # 'x': (index + 0.8)/number,
									   # 'y': 0.2},
						  # 'type': type,
						  # 'mode': 'markers' if not dim_2 else '',
						  # 'uid': '',
						  # 'xaxis': 'x' + str((index + 1)*2),
						  # 'yaxis': 'y' + str((index + 1)*2),
						  # 'x': np.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
						  # 'y': np.memmap.tolist(np.imag(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
						  # 'z': np.memmap.tolist(np.imag(dataset.data))})
				# layout['annotations'].append({'font': {'size': 16},
									# 'showarrow': False,
									# 'text': str(state.id) + ': Re(' + key + ')',
									# 'x': (index + 0.4)/number,
									# 'xanchor': 'center',
									# 'xref': 'paper',
									# 'y': 1,
									# 'yanchor': 'bottom', 'yref': 'paper'})
				# figure['data'].append({'colorbar': {'len': 0.4,
									   # 'thickness': 0.025,
									   # 'thicknessmode': 'fraction',
									   # 'x': (index + 0.8)/number,
									   # 'y': 0.8},
						  # 'type': type,
						  # 'mode': 'markers' if not dim_2 else '',
						  # 'uid': '',
						  # 'xaxis': 'x' + str((index + 1)*2 + 1),
						  # 'yaxis': 'y' + str((index + 1)*2 + 1),
						  # 'x': np.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
						  # 'y': np.memmap.tolist(np.real(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
						  # 'z': np.memmap.tolist(np.real(dataset.data))})
				# layout['annotations'].append({'font': {'size': 16},
									# 'showarrow': False,
									# 'text': str(state.id) + ': Im(' + key + ')',
									# 'x': (index + 0.4)/number,
									# 'xanchor': 'center',
									# 'xref': 'paper',
									# 'y': 0.45,
									# 'yanchor': 'bottom', 'yref': 'paper'})  
				# if (len(fit_ids_saved) > 0) and (qubit in measurement_to_fit.keys()):
					# fit_state = measurement_to_fit[qubit]
					# for key in fit_state.datasets.keys(): 
						# figure['data'].append({'colorbar': {'len': 0.4,
										   # 'thickness': 0.025,
										   # 'thicknessmode': 'fraction',
											# 'x': (index + 0.8)/number,
										   # 'y': 0.2},
							  # 'type': type,
							  # 'mode': 'lines' if not dim_2 else '',
							  # 'uid': '',
							  # 'xaxis': 'x' + str((index + 1)*2),
							  # 'yaxis': 'y' + str((index + 1)*2),
							  # 'x': np.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
							  # 'y': np.memmap.tolist(np.imag(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
							  # 'z': np.memmap.tolist(np.imag(dataset.data))})
						# figure['data'].append({'colorbar': {'len': 0.4,
										   # 'thickness': 0.025,
										   # 'thicknessmode': 'fraction',
										   # 'x': (index + 0.8)/number,
										   # 'y': 0.8},
							  # 'type': type,
							  # 'mode': 'lines' if not dim_2 else '',
							  # 'uid': '',
							  # 'xaxis': 'x' + str((index + 1)*2 + 1),
							  # 'yaxis': 'y' + str((index + 1)*2 + 1),
							  # 'x': np.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
							  # 'y': np.memmap.tolist(np.real(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
							  # 'z': np.memmap.tolist(np.real(dataset.data))})
		figure['layout'] = layout
		return figure

def query_results(query, max_rows=200):
	try:
		dataframe = psql.read_sql(query, direct_db)
		
#		rows = []
#		for i in range(min(len(dataframe), max_rows)):
#			cells = []
#			for col in dataframe.columns:
#				if col == 'id':
#					cell_className = 'id'
#					cell_contents = [dcc.Checklist(options=[{'label':str(dataframe.iloc[i][col]), 'value':str(dataframe.iloc[i][col])}], values=[])]
#				else:
#					cell_className = 'sql-result'
#					cell_contents = [str(dataframe.iloc[i][col])]
#				cell = html.Td(children=cell_contents, className=cell_className)
#				cells.append(cell)
#			rows.append(html.Tr(children=cells))
#		
#    id='table',
#    columns=[{"name": i, "id": i} for i in df.columns],
#    data=df.to_dict("rows"),
#)
		
		return [html.Div(className="query-results-scroll", 
			children=[dash_table.DataTable(
			# Header
				columns = [{"name":col, "id":col} for col in dataframe.columns],
				style_data_conditional=[{"if": {"column_id": 'id'},
										 'background-color': '#c0c0c0',
										 'color': 'white'}],
				data=dataframe.to_dict('rows'),
				row_selectable='multi',
				selected_rows=[], ### TODO: add selected rows from meas-id row
				id="query-results-table"
			)]
		)]
	except Exception as e:
		error = str(e)
		return html.Div(children=error)

def query_list():
	result = html.Ul([html.Li(children="BOMZ")])
	return result
	
def modal_content():
	return [html.Div(className="modal-content",
				children=[
					html.Div(className="modal-header", children=[
						html.Span(className="close", children="×", id="modal-select-measurements-close"),
						html.H1(children="Modal header"), 
					]),
					html.Div(className="modal-body", children = [
						html.Div(className="modal-left", children=[
							query_list()
						]),
						html.Div(className="modal-right", children=[
							html.Div(className="modal-right-content", children=[
								html.Div(children=[dcc.Textarea(id='query', value=default_query)]),
								html.Div(children=[html.Button('Execute', id='execute'), 
												   html.Button('Select all', id='select-all'), 
												   html.Button('Deselect all', id='deselect-all')]),
								html.Div(id='query-results', className='query-results', children=query_results(default_query)),
							]),
						])
					]),
					#html.Div(className="modal-footer", children=["Modal footer"])
			])]

#n_clicks_registered = 0
@app.callback(
	Output(component_id='query-results', component_property='children'),
	[Input(component_id='execute', component_property='n_clicks')],
	state=[State(component_id='query', component_property='value'),
	]
)
def update_query_result(n_clicks, query):
	#global n_clicks_registered
	#if n_clicks> n_clicks_registered:
	#n_clicks_registered = n_clicks
	return query_results(query)

@app.callback(
	Output(component_id='modal-select-measurements', component_property='style'),
	[Input(component_id='modal-select-measurements-open', component_property='n_clicks'),
	 Input(component_id='modal-select-measurements-close', component_property='n_clicks')]
)
def modal_select_measurements_open_close(n_clicks_open, n_clicks_close):
	if not n_clicks_open:
		n_clicks_open = 0
	if not n_clicks_close:
		n_clicks_close = 0
	return {'display': 'block' if (n_clicks_open - n_clicks_close) % 2 else 'none'};

if __name__ == '__main__':
	app.layout = app_layout()
	app.run_server(debug=False)
