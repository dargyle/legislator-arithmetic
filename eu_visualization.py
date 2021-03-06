import pandas as pd
import plotly.graph_objects as go

from constants import DATA_PATH
EU_PATH = DATA_PATH + '/eu/'

hue_map = {'PPE': '#3399FF',
           'S&D': '#FF0000',
           'RE': 'gold',
           'Verts/ALE': '#009900',
           'ID': '#2B3856',
           'CRE': '#0054A5',
           'GUE/NGL': '#990000',
           'NA': '#999999',
           'EFDD': '#24B9B9',
           }

dynamic_leg_data = pd.read_pickle(EU_PATH + "dynamic_leg_data.pkl")

leg_ids = dynamic_leg_data.loc[dynamic_leg_data["active"] == "active", "leg_id"].unique()
leg_id = leg_ids[3]
dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id]
dynamic_leg_data.loc[dynamic_leg_data["name_full"].str.contains('TOIA')]
dynamic_leg_data.loc[dynamic_leg_data["name_full"].str.contains('VAIDERE')]
dynamic_leg_data.loc[dynamic_leg_data["name_full"].str.contains('AUKEN')]
dynamic_leg_data.loc[dynamic_leg_data["name_full"].str.contains('SARY')]

fig = go.FigureWidget()
fig.layout.hovermode = 'closest'
fig.layout.hoverdistance = -1 #ensures no "gaps" for selecting sparse data
default_linewidth = 2
highlighted_linewidth_delta = 4
for leg_id in leg_ids:
    fig.add_trace(go.Scatter(x=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "ideal_1_time"],
                             y=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "ideal_2_time"],
                             mode='lines+markers',
                             line=dict(color=hue_map[dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "party_plot"].iloc[0]]),
                             showlegend=False,
                             opacity=1.0,
                             # hovertemplate=
                             # 'Dim 1: %{x}' +
                             # '<br>Dim 2: %{y}<br>' +
                             # '<br>Name: %{name}<br>',
                             # '<br>Party: %{party_plot}<br>',
                             name=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "name_full"].iloc[0],
                             # party_plot=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "party_plot"].iloc[0],
                             )
                  )
# fig.show()


def update_trace(trace, points, selector):
    i = points.trace_index
    for x in range(0, len(fig.data)):
        # fig.data[x]['line']['color'] = 'grey'
        fig.data[x]['opacity'] = 0.3
        # fig.data[x]['line']['width'] = default_linewidth
    #print('Correct Index: {}',format(i))
    # fig.data[i]['line']['color'] = 'red'
    fig.data[i]['opacity'] = 1
    # fig.data[i]['line']['width'] = highlighted_linewidth_delta


def update_trace(trace, points, selector):
    # this list stores the points which were clicked on
    # in all but one trace they are empty
    f.data[points.trace_index]['opacity'] = 1.0
    f.data[points.trace_index]['color'] = 'red'




for x in range(0, len(fig.data)):
    fig.data[x].on_hover(update_trace)

fig.show()



fig.write_html(EU_PATH + "dynamic_viz.html")

import plotly_express as px

fig = px.line(dynamic_leg_data[dynamic_leg_data["active"] == "active"], x="ideal_1_time", y="ideal_2_time",
              color="party_plot", line_group="leg_id", color_discrete_map=hue_map, hover_name='name_full',
              hover_data={'home_country': True, "congress": True}, width=1000, height=800)
fig.update_traces(mode='lines+markers')
fig.update_layout(title_text='European Parliament: Dynamic Ideal Points')
# for x in range(0, len(fig.data)):
#     fig.data[x]['line']['color'] = 'grey'
#     fig.data[x]['opacity'] = 0.3
fig.show()

f = go.FigureWidget(fig)
f.layout.hovermode = 'closest'
f.layout.hoverdistance = -1
for x in range(0, len(f.data)):
    f.data[x].on_hover(update_trace)
f.show()

fig.write_html(EU_PATH + "dynamic_viz.html")
