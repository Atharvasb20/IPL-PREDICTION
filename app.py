import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Must be the first streamlit command
st.set_page_config(page_title="IPL Match Analysis & Predictor", page_icon="🏏", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    try:
        points = pd.read_csv("points_table.csv")
    except Exception:
        points = None
    return matches, points

@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/winner_model.pkl')
        le_venue = joblib.load('models/le_venue.pkl')
        le_team = joblib.load('models/le_team.pkl')
        le_toss_decision = joblib.load('models/le_toss_decision.pkl')
        team_stats = joblib.load('models/team_stats.pkl')
        return model, le_venue, le_team, le_toss_decision, team_stats
    except Exception as e:
        return None, None, None, None, None

matches_df, points_df = load_data()
model, le_venue, le_team, le_toss_decision, team_stats = load_models()

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        transform: translateY(-2px);
        box-shadow: 0px 8px 15px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🏏 IPL Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard & EDA", "Match Winner Predictor"])



if page == "Dashboard & EDA":

    st.title("📋 IPL Points Table Dashboard")
    st.markdown("View the latest IPL points table, team standings, and key stats.")

    if points_df is not None:
        # Top Metrics from points table
        top_team = points_df.iloc[0]['team']
        most_points = points_df.iloc[0]['points']
        total_matches = points_df['matches'].sum()
        total_teams = points_df['team'].nunique()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>Top Team</h3><h2>{top_team}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Most Points</h3><h2>{most_points}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>Total Teams</h3><h2>{total_teams}</h2></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Points Table
        st.subheader("Current Points Table")
        st.dataframe(points_df, use_container_width=True, hide_index=True)

        # Visualizations based on points table
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.subheader("Points by Team")
            fig1 = px.bar(points_df, x='team', y='points', color='team', template="plotly_dark", 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig1, use_container_width=True)
        with col_v2:
            st.subheader("Net Run Rate (NRR) by Team")
            fig2 = px.bar(points_df, x='team', y='nrr', color='team', template="plotly_dark", 
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig2, use_container_width=True)

        # Additional Visuals: Pie and Donut Charts
        st.markdown("<br>", unsafe_allow_html=True)
        col_v3, col_v4 = st.columns(2)
        with col_v3:
            st.subheader("Win Distribution (Pie Chart)")
            if 'wins' in points_df.columns:
                fig_pie = px.pie(points_df, names='team', values='wins', title='Win Distribution',
                                 color_discrete_sequence=px.colors.qualitative.Set3, hole=0)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("'wins' column not found in points table.")
        with col_v4:
            st.subheader("Matches Played (Donut Chart)")
            if 'matches' in points_df.columns:
                fig_donut = px.pie(points_df, names='team', values='matches', title='Matches Played by Team',
                                   color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("'matches' column not found in points table.")
    else:
        st.warning("Points table data not found.")

    # Match Insights from match data
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 General Match Insights")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        # Toss Decision Donut Chart
        toss_dec = matches_df['toss_decision'].value_counts().reset_index()
        toss_dec.columns = ['Decision', 'Count']
        fig_toss = px.pie(toss_dec, names='Decision', values='Count', title='Overall Toss Decision (Donut Chart)',
                          color_discrete_sequence=['#ff9999','#66b3ff'], hole=0.5)
        fig_toss.update_layout(template="plotly_dark")
        st.plotly_chart(fig_toss, use_container_width=True)
        
    with col_m2:
        # Match Winners Pie Chart
        winner_counts = matches_df['match_winner'].value_counts().reset_index()
        winner_counts.columns = ['Team', 'Wins']
        fig_winners = px.pie(winner_counts, names='Team', values='Wins', title='Match Wins by Team (Pie Chart)',
                             color_discrete_sequence=px.colors.qualitative.Set3)
        fig_winners.update_layout(template="plotly_dark")
        st.plotly_chart(fig_winners, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_m3, col_m4 = st.columns(2)

    with col_m3:
        # Top 5 Player of the Match Bar Chart
        pom_counts = matches_df['player_of_the_match'].value_counts().head(5).reset_index()
        pom_counts.columns = ['Player', 'Awards']
        fig_pom = px.bar(pom_counts, x='Player', y='Awards', title='Top 5 Players of the Match',
                         color='Player', template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_pom, use_container_width=True)
        
    with col_m4:
        # Super Over Match Pie Chart
        super_over_counts = matches_df['super_over_match'].value_counts().reset_index()
        super_over_counts.columns = ['Super Over', 'Count']
        fig_super = px.pie(super_over_counts, names='Super Over', values='Count', title='Matches with Super Over (Pie Chart)',
                           color_discrete_sequence=['#77dd77', '#ffb347'])
        fig_super.update_layout(template="plotly_dark")
        st.plotly_chart(fig_super, use_container_width=True)

elif page == "Match Winner Predictor":
    st.title("🔮 Match Winner Predictor")
    st.markdown("Use Machine Learning to predict the winner of a match based on pre-match conditions.")
    
    if model is None:
        st.error("Model files not found! Please run `python train_models.py` first.")
    else:
        st.markdown("### 📝 Enter Match Details")
        
        col1, col2 = st.columns(2)
        
        # Get list of classes from LabelEncoders
        venues = le_venue.classes_
        teams = le_team.classes_
        toss_decisions = le_toss_decision.classes_
        
        with col1:
            selected_venue = st.selectbox("Select Venue", venues)
            team1 = st.selectbox("Team 1", teams, index=0)
            
        with col2:
            st.write("") # Spacing
            team2 = st.selectbox("Team 2", teams, index=1 if len(teams)>1 else 0)
            toss_decision = st.selectbox("Toss Decision", toss_decisions)
            
        st.markdown("<br>", unsafe_allow_html=True)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
            
        if team1 == team2:
            st.warning("Team 1 and Team 2 cannot be the same!")
        else:
            if st.button("Predict Winner 🚀"):
                # Encode inputs
                try:
                    v_encoded = le_venue.transform([selected_venue])[0]
                    t1_encoded = le_team.transform([team1])[0]
                    t2_encoded = le_team.transform([team2])[0]
                    tw_encoded = le_team.transform([toss_winner])[0]
                    td_encoded = le_toss_decision.transform([toss_decision])[0]
                    
                    input_data = pd.DataFrame({
                        'venue': [v_encoded],
                        'team1': [t1_encoded],
                        'team2': [t2_encoded],
                        'toss_winner': [tw_encoded],
                        'toss_decision': [td_encoded],
                        'team1_points': [team_stats.get(team1, {}).get('points', 0)],
                        'team1_nrr': [team_stats.get(team1, {}).get('nrr', 0)],
                        'team1_batting_sr': [team_stats.get(team1, {}).get('batting_sr', 130)],
                        'team1_bowling_eco': [team_stats.get(team1, {}).get('bowling_eco', 8.5)],
                        'team2_points': [team_stats.get(team2, {}).get('points', 0)],
                        'team2_nrr': [team_stats.get(team2, {}).get('nrr', 0)],
                        'team2_batting_sr': [team_stats.get(team2, {}).get('batting_sr', 130)],
                        'team2_bowling_eco': [team_stats.get(team2, {}).get('bowling_eco', 8.5)]
                    })
                    
                    # Predict probabilities
                    probabilities = model.predict_proba(input_data)[0]
                    
                    # Create probability DataFrame
                    prob_df = pd.DataFrame({
                        'Team_Encoded': model.classes_,
                        'Probability': probabilities * 100
                    })
                    prob_df['Team'] = le_team.inverse_transform(prob_df['Team_Encoded'])
                    
                    # Filter only the two playing teams
                    playing_teams_prob = prob_df[prob_df['Team'].isin([team1, team2])].copy()
                    
                    # Normalize probabilities so they add up to 100% for the playing teams
                    total_prob = playing_teams_prob['Probability'].sum()
                    if total_prob > 0:
                        playing_teams_prob['Probability'] = (playing_teams_prob['Probability'] / total_prob) * 100
                    
                    # Get the playing team with the highest probability
                    best_team_row = playing_teams_prob.loc[playing_teams_prob['Probability'].idxmax()]
                    predicted_winner = best_team_row['Team']
                    
                    st.markdown("---")
                    st.success(f"### 🎉 Predicted Winner: **{predicted_winner}**")
                    
                    # Show Probabilities
                    st.markdown("#### Win Probability")
                    
                    fig_prob = px.bar(playing_teams_prob, x='Team', y='Probability', color='Team', text='Probability', template="plotly_dark")
                    fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_prob.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# Add floating initials
st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 20px; background-color: #1e2130; padding: 10px 15px; border-radius: 5px; color: #ff4b4b; font-weight: bold; font-size: 18px; z-index: 1000; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #ff4b4b;">
        Project by AB
    </div>
""", unsafe_allow_html=True)
