import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.misc.pilutil import imread
import matplotlib.cm as cm
import math

#导入部分数据
deaths1 = pd.read_csv("deaths/kill_match_stats_final_0.csv")
deaths2 = pd.read_csv("deaths/kill_match_stats_final_1.csv")

deaths = pd.concat([deaths1, deaths2])

#打印前5列，理解变量
print (deaths.head(),'\n',len(deaths))

#两种地图
miramar = deaths[deaths["map"] == "MIRAMAR"]
erangel = deaths[deaths["map"] == "ERANGEL"]

#开局前100秒死亡热力图
position_data = ["killer_position_x","killer_position_y","victim_position_x","victim_position_y"]
for position in position_data:
    miramar[position] = miramar[position].apply(lambda x: x*1000/800000)
    miramar = miramar[miramar[position] != 0]

    erangel[position] = erangel[position].apply(lambda x: x*4096/800000)
    erangel = erangel[erangel[position] != 0]

n = 50000
mira_sample = miramar[miramar["time"] < 100].sample(n)
eran_sample = erangel[erangel["time"] < 100].sample(n)

# miramar热力图
bg = imread("miramar.jpg")
fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.imshow(bg)
sns.kdeplot(mira_sample["victim_position_x"], mira_sample["victim_position_y"],n_levels=100, cmap=cm.Reds, alpha=0.9)
plt.show()
fig.savefig('01.jpg')
# erangel热力图
bg = imread("erangel.jpg")
fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.imshow(bg)
sns.kdeplot(eran_sample["victim_position_x"], eran_sample["victim_position_y"], n_levels=100,cmap=cm.Reds, alpha=0.9)
plt.show()
fig.savefig('02.jpg')

#杀人武器排名
death_causes = deaths['killed_by'].value_counts()

sns.set_context('talk')
fig = plt.figure(figsize=(30, 10))
ax = sns.barplot(x=death_causes.index, y=[v / sum(death_causes) for v in death_causes.values])
ax.set_title('Rate of Death Causes')
ax.set_xticklabels(death_causes.index, rotation=90)
plt.show()
fig.savefig('03.jpg')
#排名前20的武器
rank = 20
fig = plt.figure(figsize=(20, 10))
ax = sns.barplot(x=death_causes[:rank].index, y=[v / sum(death_causes) for v in death_causes[:rank].values])
ax.set_title('Rate of Death Causes')
ax.set_xticklabels(death_causes.index, rotation=90)
plt.show()
fig.savefig('04.jpg')
#两个地图分开取
f, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].set_title('Death Causes Rate: Erangel (Top {})'.format(rank))
axes[1].set_title('Death Causes Rate: Miramar (Top {})'.format(rank))

counts_er = erangel['killed_by'].value_counts()
counts_mr = miramar['killed_by'].value_counts()

sns.barplot(x=counts_er[:rank].index, y=[v / sum(counts_er) for v in counts_er.values][:rank], ax=axes[0] )
sns.barplot(x=counts_mr[:rank].index, y=[v / sum(counts_mr) for v in counts_mr.values][:rank], ax=axes[1] )
axes[0].set_ylim((0, 0.20))
axes[0].set_xticklabels(counts_er.index, rotation=90)
axes[1].set_ylim((0, 0.20))
axes[1].set_xticklabels(counts_mr.index, rotation=90)
plt.show()
fig.savefig('05.jpg')
#吃鸡和武器的关系
win = deaths[deaths["killer_placement"] == 1.0]
win_causes = win['killed_by'].value_counts()

sns.set_context('talk')
fig = plt.figure(figsize=(20, 10))
ax = sns.barplot(x=win_causes[:20].index, y=[v / sum(win_causes) for v in win_causes[:20].values])
ax.set_title('Rate of Death Causes of Win')
ax.set_xticklabels(win_causes.index, rotation=90)
plt.show()
fig.savefig('06.jpg')


# python代码：杀人和距离的关系

def get_dist(df): #距离函数
    dist = []
    for row in df.itertuples():
        subset = (row.killer_position_x - row.victim_position_x)**2 + (row.killer_position_y - row.victim_position_y)**2
        if subset > 0:
            dist.append(math.sqrt(subset) / 100)
        else:
            dist.append(0)
    return dist

df_dist = pd.DataFrame.from_dict({'dist(m)': get_dist(erangel)})
df_dist.index = erangel.index

erangel_dist = pd.concat([erangel,df_dist], axis=1)

df_dist = pd.DataFrame.from_dict({'dist(m)': get_dist(miramar)})
df_dist.index = miramar.index

miramar_dist = pd.concat([miramar,df_dist], axis=1)

f, axes = plt.subplots(1, 2, figsize=(30, 10))
plot_dist = 150

axes[0].set_title('Engagement Dist. : Erangel')
axes[1].set_title('Engagement Dist.: Miramar')

plot_dist_er = erangel_dist[erangel_dist['dist(m)'] <= plot_dist]
plot_dist_mr = miramar_dist[miramar_dist['dist(m)'] <= plot_dist]

sns.distplot(plot_dist_er['dist(m)'], ax=axes[0])
sns.distplot(plot_dist_mr['dist(m)'], ax=axes[1])
plt.show()
fig.savefig('07.jpg')

#最后毒圈位置

#导入部分数据
deaths = pd.read_csv("deaths/kill_match_stats_final_0.csv")
#导入aggregate数据
aggregate = pd.read_csv("aggregate/agg_match_stats_0.csv")
print(aggregate.head())

#找出最后三人死亡的位置
team_win = aggregate[aggregate["team_placement"]==1] #排名第一的队伍


#找出每次比赛第一名队伍活的最久的那个player
grouped = team_win.groupby('match_id').apply(lambda t: t[t.player_survive_time==t.player_survive_time.max()])

deaths_solo = deaths[deaths['match_id'].isin(grouped['match_id'].values)]
deaths_solo_er = deaths_solo[deaths_solo['map'] == 'ERANGEL']
deaths_solo_mr = deaths_solo[deaths_solo['map'] == 'MIRAMAR']

df_second_er = deaths_solo_er[(deaths_solo_er['victim_placement'] == 2)].dropna()
df_second_mr = deaths_solo_mr[(deaths_solo_mr['victim_placement'] == 2)].dropna()
print (df_second_er)
position_data = ["killer_position_x","killer_position_y","victim_position_x","victim_position_y"]
for position in position_data:
    df_second_mr[position] = df_second_mr[position].apply(lambda x: x*1000/800000)
    df_second_mr = df_second_mr[df_second_mr[position] != 0]

    df_second_er[position] = df_second_er[position].apply(lambda x: x*4096/800000)
    df_second_er = df_second_er[df_second_er[position] != 0]

df_second_er=df_second_er


# erangel热力图
sns.set_context('talk')
bg = imread("erangel.jpg")
fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.imshow(bg)
sns.kdeplot(df_second_er["victim_position_x"], df_second_er["victim_position_y"], cmap=cm.Blues, alpha=0.7,shade=True)
plt.show()
fig.savefig('08.jpg')

# miramar热力图
bg = imread("miramar.jpg")
fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.imshow(bg)
sns.kdeplot(df_second_mr["victim_position_x"], df_second_mr["victim_position_y"], cmap=cm.Blues,alpha=0.8,shade=True)
plt.show()
fig.savefig('09.jpg')
