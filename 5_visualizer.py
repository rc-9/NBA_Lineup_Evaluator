import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import cm, patches
from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch, Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.path import Path
from plotly.subplots import make_subplots
from plotnine import ggplot, aes, geom_jitter, scale_color_manual, theme, labs, theme_bw

raw_player_df = pd.read_csv('raw_comprehensive_stats.csv')
cln_player_df = pd.read_csv('cln_comprehensive_stats.csv')
lineup_df = pd.read_csv('cln_lineup_stats.csv')
train_df = pd.read_csv('cln_train.csv')
cls_df = pd.read_csv('cln_clusters.csv', index_col='PLAYER')

st.set_page_config(layout="wide")

st.title("""
Lineup Evaluator
This app is designed to aid NBA coaching staff in lineup selections and the front-office in trade/free-agency decisions.
""")

st.sidebar.header('Test out a lineup: ')

p1 = st.sidebar.selectbox('Point-Guard:', cln_player_df[cln_player_df.POS.str.contains('G')].PLAYER)
p2 = st.sidebar.selectbox('Shooting-Guard:', cln_player_df[cln_player_df.POS.str.contains('G')].PLAYER)
p3 = st.sidebar.selectbox('Small-Forward:', cln_player_df[cln_player_df.POS.str.contains('F')].PLAYER)
p4 = st.sidebar.selectbox('Power-Forward:', cln_player_df[cln_player_df.POS.str.contains('F')].PLAYER)
p5 = st.sidebar.selectbox('Center:', cln_player_df[cln_player_df.POS.isin(['C', 'F-C', 'F'])].PLAYER)
    
test_record = pd.DataFrame(cls_df.loc[[p1, p2, p3, p4, p5]].values.sum(axis=0).reshape(1, 15), columns=cls_df.columns)

raw_test_df = raw_player_df[raw_player_df.PLAYER.isin([p1, p2, p3, p4, p5])]
cln_test_df = cln_player_df[raw_player_df.PLAYER.isin([p1, p2, p3, p4, p5])]


####################

## Get player images

c1, c2, c3, c4, c5 = st.columns((1, 1, 1, 1, 1))


base_url = 'https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/'
player_id_df = pd.read_csv('id.csv')
players = [p1, p2, p3, p4, p5]
cols = [c1, c2, c3, c4, c5]

ids = []
for i in range(5):
    ids.append(player_id_df[player_id_df.name == players[i]]['player_id'].values[0])
# ids = player_id_df[player_id_df.name.isin([p1, p2, p3, p4, p5])]['player_id'].values.tolist()

for i in range(5):
    url = base_url + str(ids[i]) + '.png'
    cols[i].image(url)


####################

c1, c2, c3, c4 = st.columns((1, 1, 1, 3))

X = train_df[cls_df.columns]
y1 = train_df.OffRtg
y2 = train_df.DefRtg
train_df['plus_rtg'] = train_df.NetRtg.map(lambda x: 0 if x < 0 else 1)
y3 = train_df.plus_rtg


clf = RandomForestRegressor(random_state=42, n_estimators=25, max_depth=10)
clf.fit(X, y1)
o_pred = round(clf.predict(test_record)[0], 1)
clf.fit(X, y2)
d_pred = round(clf.predict(test_record)[0], 1)
net_pred = round(o_pred - d_pred, 1)
if net_pred > 0:
    net_pred = '+' + str(net_pred)

   
    
#####################

    
# st.subheader('How Will This Lineup Perform?')
# c1.write(f"""
# ### Projected Offensive Rating: **{o_pred}**
# ### Projected Defensive Rating: **{d_pred}**
# ### Estimated Net Rating: **{net_pred}**
# """)
c1, c2, c3, c4 = st.columns((1, 1, 1, 3))

c1.metric('Projected Offensive Rating', o_pred)
c2.metric('Projected Defensive Rating', d_pred)
c3.metric('Estimated Net Rating', net_pred)


#####################

c1, c2 = st.columns((1, 1))

ldf = pd.read_csv('lineup_agg_stats.csv')
pdf = pd.read_csv('cln_comprehensive_stats.csv')
cols = ['%RA_FGA', '%PT_nonRA_FGA', '%MR_FGA', '%cns_2FGA', '%pullup_2FGA', '%Corner3_FGA', '%ATB3_FGA', '%cns_3PA', '%pullup_3PA', '%trsn_FGA', 
        '%iso_FGA', '%pnrbh_FGA', '%pnrrm_FGA', '%postup_FGA', '%spotup_FGA', '%handoff_FGA', '%cuts_FGA', '%offscrn_FGA', '%putbk_FGA']

comp_df = pd.DataFrame(ldf[cols].mean(), columns=['league_avg'])
comp_df['plyr_avg'] = pd.Series(pdf[pdf.PLAYER.isin([p1, p2, p3, p4, p5])][cols].mean())
comp_df['delta'] = round((comp_df.plyr_avg - comp_df.league_avg) / comp_df.league_avg * 100, 1)
comp_df['-'] = [0 for i in range(len(comp_df))]
comp_df['--'] = [0 for i in range(len(comp_df))]

c1.bar_chart(comp_df[['-', '--', 'delta']])

cols = ['Opp2P%', 'opp_RA_FG%', 'opp_PT_nonRA_FG%', 'opp_MR_FG%', 'Opp3P%', 'opp_Corner3_FG%', 'opp_ATB3_FG%', 'opp_iso_FG%',
        'opp_pnrbh_FG%', 'opp_pnrrm_FG%', 'opp_postup_FG%', 'opp_spotup_FG%', 'opp_handoff_FG%', 'opp_offscrn_FG%']
comp_df = pd.DataFrame(ldf[cols].mean(), columns=['league_avg'])
comp_df['plyr_avg'] = pd.Series(pdf[pdf.PLAYER.isin([p1, p2, p3, p4, p5])][cols].mean())
comp_df['delta'] = round((comp_df.plyr_avg - comp_df.league_avg) / comp_df.league_avg * 100, 1)
comp_df['-'] = [0 for i in range(len(comp_df))]
comp_df['--'] = [0 for i in range(len(comp_df))]

c1.bar_chart(comp_df[['delta']])


#####################

## SHOT CHART FOR INPUT LINEUP

shot_profiles_df = pd.read_csv('shot_profiles.csv')
filtered_shots = shot_profiles_df[shot_profiles_df.PLAYER_NAME.isin([p1, p2, p3, p4, p5])]


def plot_halfcourt(ax, ver):
    """Creates half-court visual on given input axes object. Methodology followed from: nbashots module"""
    
    if ver == 1:
        clr = 'White'
        ax.set_facecolor('Black')
    else:
        clr = 'Black'
    
    # Plot basket-area elements
    hoop = Circle((0, 60), radius=15, linewidth=2, color=clr, fill=False)
    backboard = Rectangle((-30, 40), 60, 0, linewidth=2, color=clr)
    ra_arc = Arc((0, 60), 80, 80, theta1=0, theta2=180, linewidth=1, color=clr)

    # Plot paint-area elements
    paint_o = Rectangle((-80, -10), 160, 190, linewidth=2, color=clr, fill=False)
    paint_i = Rectangle((-60, -10), 120, 190, linewidth=2, color=clr, fill=False)
    ft_arc = Arc((0, 180), 120, 120, theta1=0, theta2=180, linewidth=2, color=clr, fill=False)
    ft_arc2 = Arc((0, 180), 120, 120, theta1=180, theta2=0, linewidth=2, color=clr, linestyle='dashed')
    
    # Plot 3-pt area elements
    lc = Rectangle((-220, -10), 0, 160, linewidth=2, color=clr)
    rc = Rectangle((220, -10), 0, 160, linewidth=2, color=clr)
    arc = Arc((0, 140), 440, 315, theta1=0, theta2=180, linewidth=2, color=clr)
    hc_arc = Arc((0, 482.5), 120, 120, theta1=180, theta2=0, linewidth=2, color=clr)
    
    # Add each element as a patch to axis
    objects = [hoop, backboard, ra_arc, paint_o, paint_i, ft_arc, ft_arc2, lc, rc, arc, hc_arc]
    for i in objects:
        # ax.add_patch(i)
        ax.add_artist(i)
    
    # Remove tick labels and set viewpoint
    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 470)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.hexbin(filtered_shots['LOC_X'], filtered_shots['LOC_Y']+60, gridsize=(100, 100), bins='log', cmap='seismic')  
ax = plot_halfcourt(ax, 1)
c2.pyplot(fig)

plt.tight_layout()
plt.show()


#######################################

# IN COMMAND LINE, NAVIGATE TO PROJECT DIRECTORY AND EXECUTE:
# streamlit run 5_visualizer.py