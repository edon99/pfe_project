import streamlit as st
import numpy as np
import pandas as pd
from mplsoccer import Sbopen,add_image
from player_viz import passe,shot,pass_cross,transition, persure_juego, pressure_heatmap,mistake,defensive_actions,passnetwork,assists,player
from PIL import Image

parser = Sbopen()
df_competition = parser.competition()
matches = parser.match(competition_id=1267, season_id=107)
#une colone pour les mathes
matches['match'] = matches['home_team_name'] + ' vs. ' + matches['away_team_name']

@st.cache_data
def load_data(team_choice):
    mask=((matches.home_team_name==teams_choice)|(matches.away_team_name==teams_choice))
    games_selected = matches.loc[mask,["match",'match_date','kick_off','home_score','away_score','competition_stage_name','stadium_name','stadium_country_name','referee_name','referee_country_name']]
    return games_selected

teams=list(matches['home_team_name'].drop_duplicates())



st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            font-size: 20px;
            color: blue;
        }
    </style>
    """,
    unsafe_allow_html=True
)


teams_choice = st.sidebar.selectbox('Team', teams)



games = load_data(teams_choice)
game=games[["match",'match_date','kick_off','home_score','away_score','competition_stage_name','stadium_name','stadium_country_name','referee_name','referee_country_name']].reset_index(drop=True)
if teams_choice:
    header_color = '#F5F5DC'  




    st.markdown('<h1 class="custom-title">Match Information </h1>', unsafe_allow_html=True)

    text = "This table presents details of the matches played by the selected team in the 2023 Africa Cup of Nations."

# Utilisez le balisage HTML et CSS pour définir la couleur beige
    beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'

# Affichez le texte stylisé dans Streamlit
    st.markdown(beige_text, unsafe_allow_html=True)
    st.write(game)    

else: 
    st.header(' ')

mask_game=((matches.home_team_name==teams_choice)|(matches.away_team_name==teams_choice))
matches=matches.loc[mask_game]
match_list=list(matches['match'])


match_choice=st.sidebar.selectbox('Match',match_list,index=None)


plot_options=["Passing Network",'Passes','Pressure heat map','Pressure heat map Juego de Posición','Shots',"Forward passes",'Crosses','Mistakes','Defensive actions',"Assists","Player performance"]
selected_plot = st.sidebar.selectbox('VizZ type:', plot_options)
st.markdown('<h1 class="custom-title">Visualizations </h1>', unsafe_allow_html=True)






if match_choice:
    mask_2=matches[matches['match']==match_choice]
    match_id=mask_2.match_id.unique()
    def event_data(match_choice):
        mask_2=matches[matches['match']==match_choice]
        events=pd.DataFrame()
        for i in mask_2['match_id']:
            events =parser.event(i)[0]
        return events
    df=event_data(match_choice)
    selected_team = teams_choice
    df_team_selected=df[df['team_name']==selected_team]
    
    




    if selected_plot == 'Passing Network':
       #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
       st.markdown('<h1 class="custom-subheader">Passing Network: </h1>', unsafe_allow_html=True)
       text="A [passing network](https://statsbomb.com/articles/soccer/explaining-xgchain-passing-networks/) is  the application of network theory and social network analysis to passing data in football. Each player is a node, and the passes between them are connections."
       beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
       st.markdown(beige_text, unsafe_allow_html=True)
       passnetwork(match_id,selected_team)
       \
  

    if selected_plot == 'Passes':
       #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
       st.markdown('<h1 class="custom-subheader">Ball Pass: </h1>', unsafe_allow_html=True)
       text="Ball is passed between teammates."
       beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
       st.markdown(beige_text, unsafe_allow_html=True)
       st.markdown('<h1 class="custom-subheader">Ball Receipt: </h1>', unsafe_allow_html=True)
       text="The receipt or intended receipt of a pass."
       beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
       st.markdown(beige_text, unsafe_allow_html=True)
       passe(df,selected_team)

       

    elif selected_plot == 'Pressure heat map':
         #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Pressure: </h1>', unsafe_allow_html=True)
         text="Applying pressure to an opposing player who’s receiving, carrying or releasing the ball."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         pressure_heatmap(df,selected_team)

    elif selected_plot == 'Pressure heat map Juego de Posición':
         #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Pressure Juego de Posición: </h1>', unsafe_allow_html=True)
         text="Percentage of pressure applied by the selected team according to [Juego de Posición](https://breakingthelines.com/tactical-analysis/what-is-juego-de-posicion/)."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)

         persure_juego(df,selected_team) 


    elif selected_plot == 'Assists':
         #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
         #st.markdown('<h1 class="custom-subheader">Assists: </h1>', unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Shot assist: </h1>', unsafe_allow_html=True)
         text="The pass was an assist to a shot (that did not score a goal)."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Goal assist: </h1>', unsafe_allow_html=True)
         text="The pass was an assist to a goal."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)

         assists(df_team_selected,selected_team)
         \


    elif selected_plot == 'Shots':
          #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
          st.markdown('<h1 class="custom-subheader">Shot: </h1>', unsafe_allow_html=True)
          text="An attempt to score a goal, made with any (legal) part of the body."
          beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
          st.markdown(beige_text, unsafe_allow_html=True)

          shot(df_team_selected)



    elif selected_plot == 'Forward passes':
         #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Forward pass: </h1>', unsafe_allow_html=True)
         text='All passes into the final third of the pitch.'
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         transition(df_team_selected)
  
    elif selected_plot == 'Crosses':
         #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Cross: </h1>', unsafe_allow_html=True)

         text='A [cross](https://www.soccerhelp.com/terms/soccer-cross.shtml) is a "square pass" to the area in front of the goal.'
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         pass_cross(df_team_selected)
        
    elif selected_plot == 'Mistakes':
          #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
          st.markdown('<h1 class="custom-subheader">Dispossessed: </h1>', unsafe_allow_html=True)
          text="Player loses ball to an opponent as a result of being tackled by a defender without attempting a dribble."
          beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
          st.markdown(beige_text, unsafe_allow_html=True)
          st.markdown('<h1 class="custom-subheader">Miscontrol: </h1>', unsafe_allow_html=True)
          text="Player loses ball due to bad touch."
          beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
          st.markdown(beige_text, unsafe_allow_html=True)
          st.markdown('<h1 class="custom-subheader">Foul Committed: </h1>', unsafe_allow_html=True)
          text="Any infringement that is penalised as foul play by a referee. Offside are not tagged as afoul committed.."
          beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
          st.markdown(beige_text, unsafe_allow_html=True)
          st.markdown('<h1 class="custom-subheader">Error: </h1>', unsafe_allow_html=True)
          text="When a player is judged to make an on-the-ball mistake that leads to a shot on goal."
          beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
          st.markdown(beige_text, unsafe_allow_html=True)
          mistake(df_team_selected)


    

    elif selected_plot == 'Defensive actions':
         #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)

         st.markdown('<h1 class="custom-subheader">Clearance: </h1>', unsafe_allow_html=True)
         text="Action by a defending player to clear the danger without an intention to deliver it to a teammate."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Block: </h1>', unsafe_allow_html=True)
         text="Blocking the ball by standing in its path."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)

         st.markdown('<h1 class="custom-subheader">Interception: </h1>', unsafe_allow_html=True)
         text="Preventing an opponent's pass from reaching their teammates by moving to the passing lane/reacting to intercept it."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Ball Recovery: </h1>', unsafe_allow_html=True)
         text="An attempt to recover a loose ball."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         defensive_actions(df_team_selected)
         \
         \
         \
         \
         \
         \
     
    elif selected_plot == 'Player performance':
         #st.markdown('<h1 class="custom-header">Description </h1>', unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Pass: </h1>', unsafe_allow_html=True)
         text="The Ball is passed by the player selected."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Carry: </h1>', unsafe_allow_html=True)
         text="The player controls the ball at their feet while moving or standing still."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)

         st.markdown('<h1 class="custom-subheader">Under pressure: </h1>', unsafe_allow_html=True)
         text='The action was performed while being pressured by an opponent.'
  
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         st.markdown('<h1 class="custom-subheader">Counterpress: </h1>', unsafe_allow_html=True)
         text="Pressing actions within 5 seconds of an open play turnover."
         beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
         st.markdown(beige_text, unsafe_allow_html=True)
         player(df_team_selected)

else:
    text="Please select a match from 'Match' and the event type you would like to display on the screen from 'VizZ type' dropdown menu on the left."
    beige_text = f'<span style="color: #F5F5DC;font-size: 20px">{text}</span>'
    st.markdown(beige_text, unsafe_allow_html=True)


   









