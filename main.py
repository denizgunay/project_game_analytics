import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as colors
from plotly.subplots import make_subplots
import joblib


###############################
# CONFIGURATION
###############################


# Functions
def get_model():
    return joblib.load("model/catboost_model.pkl")


def get_scaler():
    return joblib.load("model/scaler.pkl")


@st.cache_data
def get_graph22_2():
    return pd.read_pickle("data/graph22_2.pkl")


@st.cache_data
def get_graph28_2():
    return pd.read_pickle("data/graph28_2.pkl")


@st.cache_data
def get_graph31_2():
    return pd.read_pickle("data/graph31_2.pkl")


@st.cache_data
def get_graph2():
    return pd.read_pickle("data/graph2.pkl")


# Layout
st.set_page_config(
    layout="wide",
    page_title="Game Analysis - Deniz Gunay",
    page_icon="https://cdn.prod.website-files.com/668bb5411585cd90fd5046d6/66ad484c30a48b9125740f82_favicon-32x32.webp",
)


# Banner
custom_html = """
<div class="banner">
    <img src="https://f5adce358e479954a9c5-8426d9a3ad512832042d342ce93c88f9.ssl.cf3.rackcdn.com/asset-graphics/gameanalytics-gameanalytics-fxhs964e3ktxrbv2ctkp.png" alt="Banner Image">
</div>
<style>
    .banner {
        width: 50%;
        height: 700px;
        overflow: hidden;
        
    }
    .banner img {
        width: 80%;
        object-fit: contain;
    }
</style>
"""

st.components.v1.html(custom_html)

# Header
st.header(":blue[Game] Data Analysis")

# Audio
st.audio("audio/gangsta_paradise.mp3", format="audio/mpeg", loop=True, autoplay=True)


# Tabs
part1, part2, part3, part4 = st.tabs(
    [
        "Part I: Analysis",
        "Part II: A/B Testing",
        "Part III: Modelling",
        "Part IV: Prediction",
    ]
)

# Tab configuration
st.markdown(
    """
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 2px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #FFFFFF;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
        transition: background-color 0.3s ease; 
        color: #000000; 
        margin-right: 10px;
    }

    
	.stTabs [data-baseweb="tab"]:hover {
  		background-color: #8585ad; 
        color: #FFFFFF; 
    }

    
	.stTabs [aria-selected="true"] {
  		background-color: #4d4dff; 
        color: #FFFFFF; 
	}

</style>""",
    unsafe_allow_html=True,
)


# Back to Top Button
scroll_to_top = """
    <script>
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
    </script>
"""

st.markdown(
    """<a href="#game-data-analysis" style="text-decoration:none;">
                <button style="position:fixed;bottom:60px;right:10px;padding:10px 20px;font-size:16px;">
                    Back to Top
                </button>
               </a>""",
    unsafe_allow_html=True,
)


###############################
# PART I: ANALIZ
###############################

with part1:
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In this section, we will delve into the current state of the game, leveraging various metrics and cohort analyses to gain deeper insights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 1
    st.subheader(":blue[1) Retention Rate]")
    st.image("images/part_i/retention_rate.png")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Looking at the chart, we observe that the retention rate is approximately 55% for D1, 39% for D7, and 30% for D28. According to GameAnalytics’ *Player Retention Report* published in 2019, the top-performing games have average retention rates of 40% for D1, 15% for D7, and 6.5% for D28. In light of this comparison, we can conclude that the game has a significantly higher retention rate than the industry average. To further enhance retention, strategies such as segmenting players and offering free items to loyal player groups could be implemented. Additionally, introducing a medal system based on the time players spend in the game could strengthen player engagement and boost retention rates.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 2
    st.subheader(":blue[2) Age distribution of players by country]")

    df2 = get_graph2()
    country_age = st.selectbox(
        "Please select the country variable:",
        [
            "Zephyra",
            "Thalassia",
            "Sunridge",
            "Amaryllis",
            "Brighthaven",
            "Luminara",
            "Gleamwood",
            "Azurelia",
            "Eldoria",
            "Windemere",
            "Rosewyn",
            "Floravia",
            "Glimmerdell",
            "Emberlyn",
            "Frostford",
            "Crystalbrook",
            "Seraphina",
            "Silvermist",
            "Moonvale",
            "Starcliff",
        ],
        key="selectbox1",
    )

    df_age = df2[df2["country"] == country_age]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df_age["age"].astype(float),
            nbinsx=10,
            histfunc="count",
            marker_color="skyblue",
            opacity=0.75,
            name="Age Distribution",
        )
    )

    fig.update_layout(
        title=f"Age Distribution Histogram for {country_age}",
        xaxis_title="Age",
        yaxis_title="Frequency",
        template="plotly_white",
    )

    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we examine the age distribution of players by country, we encounter a fascinating finding: the age distribution is remarkably similar across almost all countries, with an average age of 47. This suggests that the typical player profile for the given dataset is an adult in their 40s, likely engaged in professional life. This insight will be invaluable in making sense of the analyses that follow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 3
    st.subheader(":blue[3) Average daily hours spent in the game by users]")

    df3 = pd.read_pickle("data/graph3.pkl")
    grouped = (
        df3.groupby(["group_type", "group_id"])
        .agg({"avg_hours_spent_per_user": "mean"})
        .reset_index()
    )
    grouped["group_label"] = grouped.groupby("group_type").cumcount() + 1
    grouped["group_label"] = (
        grouped["group_type"] + " " + grouped["group_label"].astype(str)
    )

    fig = go.Figure()
    weekday_data = df3[df3["is_weekend"] == 0]
    fig.add_trace(
        go.Scatter(
            x=weekday_data["event_date"],
            y=weekday_data["avg_hours_spent_per_user"],
            mode="markers+lines",
            marker=dict(
                size=10, symbol="x", color="blue", line=dict(width=2, color="black")
            ),
            text=weekday_data["avg_hours_spent_per_user"],
            textposition="top center",
            name="Weekday",
        )
    )

    weekend_data = df3[df3["is_weekend"] == 1]
    fig.add_trace(
        go.Scatter(
            x=weekend_data["event_date"],
            y=weekend_data["avg_hours_spent_per_user"],
            mode="markers+lines",
            marker=dict(
                size=10, symbol="x", color="red", line=dict(width=2, color="black")
            ),
            text=weekend_data["avg_hours_spent_per_user"],
            textposition="top center",
            name="Weekend",
        )
    )

    fig.update_xaxes(tickformat="%b %d", dtick="D1", tickangle=45)

    fig.update_layout(
        title="Average hours spent per user in weekends and weekdays (May 2021)",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis_title="Date",
        yaxis_title="Average Hours Spent Per User",
        xaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        yaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        xaxis=dict(showgrid=False, gridcolor="white"),
        yaxis=dict(showgrid=False, gridcolor="white"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_x=0.05,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    for _, row in grouped.iterrows():
        fig.add_annotation(
            x=df3[df3["group_id"] == row["group_id"]]["event_date"].mean(),
            y=row["avg_hours_spent_per_user"],
            text=row["group_label"],
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
        )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            The chart above reveals that players generally spend around 1 hour per day in the game. However, when we analyze this data separately for weekdays and weekends, we find that players tend to spend more time in the game on weekdays. To understand the reasons behind this, it’s crucial to have a clear understanding of the player profile. Assuming the average player is 47 years old and actively working, it’s likely that they play the game during their commute, in the metro, or during breaks at work. To encourage players to spend more time in the game on weekends, special weekend events could be designed to enhance engagement.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 4
    st.subheader(":blue[4) Total time spent in the game by age groups]")
    df_age_stat = pd.read_pickle("data/graph4.pkl")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_age_stat["age_bins"],
            y=df_age_stat["time_spend"],
            marker=dict(color="orange"),
            text=df_age_stat["time_spend"].round(),
            textposition="inside",
            textfont=dict(size=20, color="white"),
            name="Time Spent",
            width=0.3,
        )
    )

    fig.update_layout(
        title="Time Spent by Age Group",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis_title="Age Group",
        yaxis_title="Time Spent",
        xaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        yaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, range=[0, 7650]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When players are divided into age groups of 16-28, 28-41, 41-53, 53-66, and 66-78, we observe that the time spent in the game is nearly the same across all age groups.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 5
    st.subheader(":blue[5) Average time spent by levels]")
    df_level_time = pd.read_pickle("data/graph5.pkl")
    fig = go.Figure(
        data=[
            go.Bar(
                x=df_level_time["level_group"],
                y=df_level_time["avg_time_per_user"],
                marker=dict(color="royalblue"),
                text=df_level_time["avg_time_per_user"].round(1),
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Average Time Spent per User by Level Group",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis_title="Level Group",
        yaxis_title="Average Time Spent per User (seconds)",
        xaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        yaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        xaxis=dict(showgrid=False, gridcolor="white"),
        yaxis=dict(showgrid=True, gridcolor="lightgrey"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_x=0.01,
        margin=dict(l=40, r=40, t=40, b=40),
        bargap=0.15,
        bargroupgap=0.1,
    )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            By grouping the levels and calculating the average time spent per user, we arrive at the chart above. The chart shows that players spend nearly the same amount of time between levels 200 and 700, but after level 700, the time spent begins to decline. This could be due to the increasing difficulty of higher levels, or it might indicate that players are losing interest in the game. Looking at our database, there is very little data available for players above level 950, so it's difficult to draw definitive conclusions at this point. However, for levels 700 and beyond, new, unique, and non-repetitive tasks could be designed to keep players engaged.
        <p></p>
            Additionally, when we examine the levels individually using the IQR method, we notice that players spend more time on Level 199, Level 209, Level 339, and Level 349 compared to other levels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 6
    st.subheader(":blue[6) Win, fail, and quit rates by level]")
    df_level_status = pd.read_pickle("data/graph6.pkl")
    trace1 = go.Bar(
        x=df_level_status["level_group"],
        y=df_level_status["avg_wins_per_user"],
        name="Wins",
        marker=dict(color="limegreen", opacity=0.7),
    )

    trace2 = go.Bar(
        x=df_level_status["level_group"],
        y=df_level_status["avg_quits_per_user"],
        name="Quits",
        marker=dict(color="orange", opacity=0.7),
        yaxis="y2",
    )

    trace3 = go.Bar(
        x=df_level_status["level_group"],
        y=df_level_status["avg_fails_per_user"],
        name="Fails",
        marker=dict(color="darkblue", opacity=0.7),
    )

    layout = go.Layout(
        title="Average Wins, Quits, and Fails per User by Level Group",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis=dict(
            title="Level Group",
            title_font=dict(size=18, family="Arial, sans-serif"),
            showgrid=False,
        ),
        yaxis=dict(
            title="Avg Wins/Fails per User",
            title_font=dict(size=18, family="Arial, sans-serif"),
            showgrid=False,
        ),
        yaxis2=dict(
            title="Avg Quits per User",
            title_font=dict(size=18, family="Arial, sans-serif"),
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        barmode="group",
        showlegend=True,
        legend=dict(x=0.8, y=1.1, orientation="h"),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the chart above, the green bars represent the win rate, the blue bars represent the fail rate, and the orange bars represent the quit rate. The y-axis for the green and blue bars is on the left, while the y-axis for the orange bars is on the right side of the graph. The chart shows that up until level 150, the win rate is either equal to or higher than the fail rate. However, after level 150, the game becomes progressively harder, and the fail rate surpasses the win rate. Similarly, from level 150 onwards, the quit rate also steadily increases. The rise in quit rates supports our earlier observation from the previous chart that the game is becoming more difficult, or that players may be losing interest.
        <p></p>
            Additionally, when we sort all the levels by their fail rates, we observe that levels ending in 9 tend to have higher fail rates, with Level 199 having the highest fail rate overall.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 7
    st.subheader(":blue[7) Average moves made and moves left by level]")
    trace1 = go.Scatter(
        x=df_level_status["level_group"],
        y=df_level_status["avg_movesmade_per_user"],
        mode="lines+markers",
        name="Moves Made",
        line=dict(color="blue", width=3),
        marker=dict(size=8, symbol="circle", color="blue"),
        yaxis="y1",
    )

    trace2 = go.Bar(
        x=df_level_status["level_group"],
        y=df_level_status["avg_movesleft_per_user"],
        name="Moves Left",
        marker=dict(color="orange"),
        yaxis="y2",
        opacity=0.7,
    )

    layout = go.Layout(
        title="Average Moves Made vs. Moves Left per User by Level Group",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis=dict(
            title="Level Group",
            title_font=dict(size=18, family="Arial, sans-serif"),
            showgrid=False,
        ),
        yaxis=dict(
            title="Avg Moves Made per User",
            title_font=dict(size=18, family="Arial, sans-serif"),
            showgrid=False,
        ),
        yaxis2=dict(
            title="Avg Moves Left per User",
            title_font=dict(size=18, family="Arial, sans-serif"),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.7, y=1.1, orientation="h"),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the chart above, the blue line represents the average number of moves made by players, while the orange bar indicates the average number of moves left. Similar to the previous chart on win, fail, and quit rates, we can see that after level 150, the number of moves made exceeds the number of moves left. However, after level 700, the number of moves made begins to decline. This trend is similar to what we observed in the chart for average time spent by levels (chart #5). There is insufficient data for players at higher levels in the database, making it difficult to draw a definitive conclusion about the cause of this decline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 8
    st.subheader(":blue[8) Average coin expenditure by age group]")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_age_stat["age_bins"],
            y=df_age_stat["coin_spend"],
            mode="markers+lines",
            marker=dict(size=12, color="royalblue", line=dict(width=2, color="black")),
            line=dict(width=2, color="royalblue"),
            text=df_age_stat["coin_spend"],
            textposition="top center",
            name="Coin Spend",
        )
    )

    fig.update_layout(
        title="Coin Spend by Age Group",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis_title="Age Group",
        yaxis_title="Coin Spend",
        xaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        yaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, range=[15000, 21500]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we look at the average coin expenditure by age group, we observe that there is no significant difference between the groups. In the following pages, we will build a model to predict whether users will make a purchase or not. In this model, we will see that the amount of coins spent and earned has a significant impact on the decision to buy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 9
    st.subheader(":blue[9) Average booster expenditure by age group]")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_age_stat["age_bins"],
            y=df_age_stat["booster_spend"],
            mode="markers+lines",
            marker=dict(size=12, color="royalblue", line=dict(width=2, color="black")),
            line=dict(width=2, color="royalblue"),
            text=df_age_stat["booster_spend"],
            textposition="top center",
            name="Booster Spend",
        )
    )

    fig.update_layout(
        title="Booster Spend by Age Group",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis_title="Age Group",
        yaxis_title="Booster Spend",
        xaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        yaxis_title_font=dict(size=20, family="Arial, sans-serif"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, range=[10, 22]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In both the chart above and the previous coin expenditure chart, we can see that there is no significant difference between the groups. 
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 10
    st.subheader(":blue[10) Number of users by country]")
    df2_2 = pd.read_pickle("data/graph2_2.pkl")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df2_2["country"],
                values=df2_2["num_users"],
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                marker=dict(
                    colors=colors.qualitative.Pastel, line=dict(color="white", width=2)
                ),
            )
        ]
    )

    fig.update_layout(
        title="User Distribution by Country",
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1000,
        height=800,
    )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we look at the users by country, we can see that the largest proportion, 28%, is from Zephyra. Following Zephyra, Emberlyn accounts for 14.3%, and Moonvale follows with 8.71%.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 11
    st.subheader(":blue[11)	Revenue distribution by country]")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df2_2["country"],
                values=df2_2["sum_revenue"],
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                marker=dict(
                    colors=colors.qualitative.Pastel, line=dict(color="white", width=2)
                ),
            )
        ]
    )

    fig.update_layout(
        title="Revenue Distribution by Country",
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1000,
        height=800,
    )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When examining the revenue distribution by country, we see that 68% of the total revenue comes from Zephyra. Following Zephyra, Amaryllis accounts for 7%, and Gleamwood follows with 4.85%. On the other hand, although 23% of the users are from Emberlyn and Moonvale, these countries contribute only 2.4% of the total revenue. Therefore, in-game activities should particularly consider the schedules and special days of users from Zephyra. Additionally, marketing efforts should be increased in Amaryllis and Gleamwood.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 12
    st.subheader(
        ":blue[12)	Revenue distribution and average time spent based on event participation]"
    )
    df12 = pd.read_pickle("data/graph12.pkl")

    fig1 = go.Figure(
        data=[
            go.Pie(
                labels=df12["event_participate"],
                values=df12["avg_rev"],
                textinfo="label+percent",
                insidetextorientation="radial",
                hole=0.3,
                marker=dict(colors=["darkblue", "orange"]),
            )
        ]
    )

    fig1.update_layout(
        title="Average Revenue by Event Participation",
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=df12["event_participate"],
                values=df12["avg_time_spend"],
                textinfo="label+percent",
                insidetextorientation="radial",
                hole=0.3,
                marker=dict(colors=["darkblue"]),
            )
        ]
    )

    fig2.update_layout(
        title="Average Time Spent by Event Participation",
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # buraya dikkat with part1: vardi...
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the first chart above, we can see that 75% of the revenue comes from users who participate in events. Therefore, diversifying the event calendar for Zephyra, as mentioned in the previous chart, could lead to increased revenue.
            In the second chart, we observe that users who participate in events spend twice as much time in the game compared to those who do not. From previous charts, we noted that users spend more time in the game during weekdays, and that planning events for weekends could increase the time spent in the game. The second chart supports this observation. In other words, adding more events on weekends will likely increase the amount of time players spend in the game.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 13
    st.subheader(":blue[13)	Network and Installation]")
    df13 = pd.read_pickle("data/graph13.pkl")
    fig = px.bar(
        df13.sort_values(by="total_installments", ascending=False),
        x="total_installments",
        y="network",
        orientation="h",  # Horizontal
        color="network",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Number of Installs by Network",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Number of Installs",
        yaxis_title="Network",
        margin=dict(l=100, r=40, t=40, b=40),
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside")
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the chart above, we can see that the majority of downloads came from Buzz. Before making any comments on this, let's first take a look at the spending for these channels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 14
    st.subheader(":blue[14)	Network and Cost]")
    fig = px.bar(
        df13.sort_values(by="total_cost", ascending=False),
        x="total_cost",
        y="network",
        orientation="h",
        color="network",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Total cost by network",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Total Cost",
        yaxis_title="Network",
        margin=dict(l=100, r=40, t=40, b=40),
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside")
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the chart above, we see the same ranking as in the previous one regarding the number of users. There is no cost associated with users who download the game organically. Therefore, organic growth is the most cost-effective strategy for the company. But how efficient are the other channels? To answer that, we need to look at the "Network and Number of Installations per $" chart below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 15
    st.subheader(":blue[15) Network and Number of Installations per $]")
    fig = px.bar(
        df13,
        x="installs_per_cost_unit",
        y="network",
        orientation="h",
        color="network",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Number of Installation per $ by Network",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Number of Installation per $",
        yaxis_title="Network",
        margin=dict(l=100, r=40, t=40, b=40),
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside")
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Looking at the graph above, we can see that the most efficient marketing investments for the company are in the Buzz, Sid, Woody, and Jessie channels, in that order. This indicates that the company is managing its marketing expenses effectively. Of course, the Organic channel is the hidden leader in the graph, but it appears last because it doesn't incur any cost. To attract more users through the Organic channel, features like "Invite a Friend" could be added to the game, with various reward mechanisms for both the inviter and the users who download the game via the invite.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 16
    st.subheader(":blue[16)	Number of Installations by Country]")
    df16 = pd.read_pickle("data/graph16.pkl")
    fig = px.bar(
        df16.sort_values(by="total_installments", ascending=False),
        x="total_installments",
        y="country",
        orientation="h",
        color="country",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Number of Installs by Country",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Number of Installs",
        yaxis_title="Country",
        margin=dict(l=100, r=40, t=40, b=40),
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside")
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the chart above, we can see that the highest number of installations occurred in Mercury, Pluton, and Venus, in that order. Before making any conclusions, let's take a look at the marketing expenditures in these countries. We will examine this in the next chart.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 17
    st.subheader(":blue[17)	Marketing Costs by Country]")
    fig = px.bar(
        df16.sort_values(by="total_cost", ascending=False),
        x="total_cost",
        y="country",
        orientation="h",
        color="country",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Total cost by country",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Total Cost",
        yaxis_title="Country",
        margin=dict(l=100, r=40, t=40, b=40),
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside")
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the chart above, we can see the spending for each country. The ranking is quite similar to the previous chart showing the number of installations. Both charts show the same ranking for Mercury, Venus, Pluton, Saturn, and Uranus. However, to determine which country’s spending resulted in the most installations, we will divide the number of installations for each country by the total spending for that country. We will answer this question in the next chart.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 18
    st.subheader(":blue[18) Number of Installations per $ by Country]")
    fig = px.bar(
        df16,
        x="installs_per_cost_unit",
        y="country",
        orientation="h",
        color="country",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Number of Installation per $ by Country",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Number of Installation per $",
        yaxis_title="Country",
        margin=dict(l=100, r=40, t=40, b=40),
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside")
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            The chart above is quite important for us, as we see that Saturn and Uranus countries yield the highest number of users per $1 spent. On the other hand, in Mercury and Venus, where the highest marketing expenditures are made, this figure is relatively low. Therefore, we can dive deeper and analyze the revenue per user in each country.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 19
    st.subheader(":blue[19) Revenue per User by Country]")
    df19 = pd.read_pickle("data/graph19.pkl")
    fig = px.bar(
        df19,
        x="total_rev_per_user",
        y="country",
        orientation="h",
        color="country",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Revenue per user by Country",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Revenue per user",
        yaxis_title="country",
        margin=dict(l=100, r=40, t=40, b=40),
    )

    fig.update_traces(texttemplate="%{x}", textposition="outside")
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Looking at the graph above, we can see that the highest revenue per user comes from Venus and Mercury, while the lowest revenue is generated from Uranus and Saturn. Although the marketing cost per user in Saturn and Uranus is lower, this could be influenced by exchange rates. However, the average revenue per user from these countries is still lower compared to others. As a result, the total spending in Venus and Mercury is higher, since these two countries generate the most revenue.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 20
    st.subheader(":blue[20)	Revenue, Costs, and Cost Per Revenue by Platform]")
    df20 = pd.read_pickle("data/graph20.pkl")
    trace1 = go.Bar(
        x=df20["platform"],
        y=df20["revenue"],
        name="Revenue",
        marker=dict(color="limegreen", opacity=0.7),
        width=0.1,
        offsetgroup=0,
    )

    trace2 = go.Bar(
        x=df20["platform"],
        y=df20["cost"],
        name="Cost",
        marker=dict(color="darkblue", opacity=0.7),
        width=0.1,
        offsetgroup=1,
    )

    trace3 = go.Bar(
        x=df20["platform"],
        y=df20["rev_to_cost"],
        name="Rev to Cost",
        marker=dict(color="orange", opacity=0.7),
        yaxis="y2",
        width=0.1,
        offsetgroup=2,
    )

    layout = go.Layout(
        title="Revenue, Cost and Revenue/Cost by Platform (Android & iOS)",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis=dict(
            title="Platform",
            title_font=dict(size=18, family="Arial, sans-serif"),
            showgrid=False,
        ),
        yaxis=dict(
            title="$",
            title_font=dict(size=22, family="Arial, sans-serif"),
            showgrid=False,
        ),
        yaxis2=dict(
            title="Revenue to Cost",
            title_font=dict(size=18, family="Arial, sans-serif"),
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        barmode="group",
        showlegend=True,
        legend=dict(x=0.8, y=1.1, orientation="h"),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Marketing expenses for the game are higher than the revenue generated. The above chart visualizes this situation for both Android and iOS users. It shows that the revenue per unit cost is higher for iOS users compared to Android users. The likely reason for this higher revenue from iOS users could be their higher economic well-being. In any case, targeting iOS users in marketing efforts could potentially increase the company's revenue.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 21
    st.subheader(":blue[21) Android vs IOS]")
    df21_1 = pd.read_pickle("data/graph21_1.pkl")
    df21_2 = pd.read_pickle("data/graph21_2.pkl")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Time Spent",
            "User Distribution",
        ),
        specs=[[{"type": "domain"}, {"type": "domain"}]],
    )

    # (Average Time Spent per User by Platform)
    fig.add_trace(
        go.Pie(
            labels=df21_1["platform"],
            values=df21_1["avg_time_spent_per_user"],
            textinfo="label+percent",
            insidetextorientation="radial",
            hole=0.3,
            marker=dict(colors=["limegreen", "lightgrey"]),
        ),
        row=1,
        col=1,
    )

    # (Number of Users by Platform)
    fig.add_trace(
        go.Pie(
            labels=df21_2["platform"],
            values=df21_2["user_count"],
            textinfo="label+percent",
            insidetextorientation="radial",
            hole=0.3,
            marker=dict(colors=["limegreen", "lightgrey"]),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            It appears that iOS users spend the most time in the game and are the most frequent downloaders. If we assume that in-app ads are generating revenue, the table above might reflect a negative situation for us. This is because, if revenue is being generated through in-app ads, the best strategy would be to reach as many users as possible. However, the user acquisition cost for the App Store is higher compared to the Play Store, and globally, there are more Android users than iOS users. Therefore, assuming that revenue is generated through in-app ads, having the majority of users as Android users would be a more profitable position.
            <p></p>
            However, if revenue is generated through subscriptions or in-app purchases, the table above may actually reflect a positive situation for us. According to a 2014 Comcast study, the median annual income of iOS users is $85,000, while Android users earn $61,000 annually. In other words, iOS users earn 40% more than Android users. We can also observe the impact of this socioeconomic difference when we examine the subscription behaviors of iOS and Android users. For instance, in 2021, iOS users spent $13.5 billion on subscriptions, whereas Android users spent only $4.8 billion.
            <p></p>
            According to future projections, iOS users are expected to spend increasingly more on the App Store.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part1, cent_part1, last_part1 = st.columns([0.5, 1, 2])
    cent_part1.image("images/part_i/android_vs_ios.png", width=1000)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Therefore, the marketing department should focus specifically on the App Store and develop strategies to further increase the number of iOS users.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 22
    st.subheader(":blue[22) ROAS]")
    df22 = pd.read_pickle("data/graph22.pkl")
    avg_roas = df22["ROAS"].mean()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df22["date"],
            y=df22["ROAS"],
            mode="lines+markers",
            name="ROAS",
            line=dict(color="black"),
            marker=dict(color="red"),
        )
    )

    fig.update_layout(
        title="Daily ROAS",
        xaxis_title="Date",
        yaxis_title="ROAS",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"ROAS: {avg_roas:.2f}",
        xref="paper",
        yref="paper",
        x=0.90,
        y=1.1,
        showarrow=False,
        font=dict(size=12, color="darkblue"),
        align="right",
    )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Although the average ROAS for the game is 0.26, when we examine the ROAS values on a daily basis, we can observe an increasing trend. At the beginning of May, ROAS values were around 6%, but by the end of the month, they had reached 42%, which is a very promising development.
        <p></p>
            Additionally, we can take a closer look at the daily ROAS graph by country and examine the network breakdowns for each country:
        </div>
        """,
        unsafe_allow_html=True,
    )

    country_roas = st.selectbox(
        "Please select a country:",
        ["Mercury", "Venus", "Pluton", "Saturn", "Uranus"],
        key="selectbox2",
    )

    df22_2 = get_graph22_2()
    df22_2 = df22_2[df22_2["country"] == country_roas]
    networks = df22_2.network.unique()
    fig = go.Figure()
    network_roas = dict()
    annot_loc = 1
    for network in networks:
        temp_df = df22_2[df22_2["network"] == network]
        fig.add_trace(
            go.Scatter(
                x=temp_df["date"],
                y=temp_df["daily_roas"],
                mode="lines+markers",
                name=network,
            )
        )

        avg_roas = temp_df["daily_roas"].mean()
        fig.add_annotation(
            text=f"Average {network} ROAS: {avg_roas:.2f}",
            xref="paper",
            yref="paper",
            x=0.05,
            y=annot_loc,
            showarrow=False,
            font=dict(size=12, color="darkblue"),
            align="right",
        )
        annot_loc -= 0.1

    fig.update_layout(
        title=f"ROAS for {country_roas}",
        xaxis_title="Date",
        yaxis_title="ROAS",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            For example, in the Mercury country, the Buzz, Sid, and Woody channels are steadily increasing, while the Jessie channel shows a high level of variance. On the other hand, in the Venus country, the Buzz and Sid channels remain stable, but there is a significant variance observed in the Jessie and Woody channels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 23
    st.subheader(":blue[23) Daily Installations]")
    df23 = pd.read_pickle("data/graph23.pkl")
    avg_installs = df23["daily_installs"].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df23["install_date"],
            y=df23["daily_installs"],
            mode="lines+markers",
            name="Daily Install",
            line=dict(color="black"),
            marker=dict(color="darkblue"),
        )
    )

    fig.update_layout(
        title="Daily Install",
        xaxis_title="Install Date",
        yaxis_title="Daily Install",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"Average Daily Install: {avg_installs:.2f}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.99,
        showarrow=False,
        font=dict(size=12, color="darkblue"),
        align="right",
    )
    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we look at the daily installations, we can see some fluctuations until the last week of May, but there is a noticeable increase in installs during the final week. On average, we can observe that around 7,000 installs occur daily.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 24
    st.subheader(":blue[24) DAU and Daily Session Count]")
    df24 = pd.read_pickle("data/graph24.pkl")
    avg_dau = df24["DAU"].mean()
    avg_daily_sessions = df24["daily_sessions"].mean()
    avg_sessions_per_dau = df24["sessions_per_DAU"].mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df24["event_date"],
            y=df24["daily_sessions"],
            mode="lines+markers",
            name="Daily Sessions",
            line=dict(color="black"),
            marker=dict(color="black"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df24["event_date"],
            y=df24["DAU"],
            mode="lines+markers",
            name="DAU",
            line=dict(color="orange"),
            marker=dict(color="orange"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Daily Sessions and DAU",
        xaxis_title="Event Date",
        yaxis_title="Daily Sessions",
        yaxis2=dict(
            title="DAU", overlaying="y", side="right", showgrid=False, showline=True
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"Average DAU: {avg_dau:.2f}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        showarrow=False,
        font=dict(size=12, color="orange"),
        align="right",
    )

    fig.add_annotation(
        text=f"Average Daily Sessions: {avg_daily_sessions:.2f}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.90,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Although both DAU and daily session numbers show an increasing trend in May, we observe a decline in both DAU and daily sessions in June. To make a comment on the rate of increase and decrease in DAU and daily sessions, we will examine the SessionDAU graph.

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 25
    st.subheader(":blue[25) SessionDAU]")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df24["event_date"],
            y=df24["sessions_per_DAU"],
            mode="lines+markers",
            name="Sessions per DAU",
            line=dict(color="black"),
            marker=dict(color="darkorange"),
        )
    )

    fig.update_layout(
        title="Sessions per DAU",
        xaxis_title="Event Date",
        yaxis_title="Sessions per DAU",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"Average Sessions per DAU: {avg_sessions_per_dau:.2f}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        showarrow=False,
        font=dict(size=12, color="darkblue"),
        align="right",
    )

    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Looking at the graph above, we can observe a decreasing trend. Therefore, when combining this information with the previous graph, we can conclude that throughout May, the growth rate of DAU was higher than the growth rate of session numbers. However, in June, the decline rate of DAU was lower than the decline rate of session numbers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 26
    st.subheader(":blue[26) ARPU]")
    left_part1, cent_part1, last_part1 = st.columns([4, 6, 7])
    cent_part1.image("images/part_i/arpu.png", width=750)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            For our dataset, the total revenue is 413,520 and the total number of players is 214,888, so the ARPU (Average Revenue Per User) is calculated as 1.92.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 27
    st.subheader(":blue[27) ARPPU]")
    left_part1, cent_part1, last_part1 = st.columns([4, 6, 7])
    cent_part1.image("images/part_i/arppu.png", width=750)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            For the dataset, the total revenue is 413,520, and the total number of paying players is 8,130, so the ARPPU (Average Revenue Per Paying User) is calculated as 50.86. A large gap between ARPPU and ARPU is not ideal. Therefore, it's crucial to thoroughly analyze non-paying players and develop strategies to encourage them to make purchases. At the same time, extra attention should be given to ensuring that paying players continue to make repeated purchases.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 28
    st.subheader(":blue[28) ARPDAU]")
    df28 = pd.read_pickle("data/graph28.pkl")
    avg_arpdau = df28["ARPDAU"].mean()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df28["date"],
            y=df28["ARPDAU"],
            mode="lines+markers",
            name="ARPDAU",
            line=dict(color="black"),
            marker=dict(color="magenta"),
        )
    )

    fig.update_layout(
        title="ARPDAU",
        xaxis_title="Date",
        yaxis_title="ARPDAU",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"Average ARPDAU: {avg_arpdau:.2f}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=1.1,
        showarrow=False,
        font=dict(size=12, color="darkblue"),
        align="right",
    )

    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Above, you can see the 45-day ARPDAU values. The graph shows significant fluctuations. An interesting point in the graph is that, with the exception of May 2nd, every Sunday has a higher ARPDAU than the following Monday. Additionally, the Friday-Saturday-Sunday trio consistently shows higher ARPDAU values than the rest of the week, except for the first week.
        <p></p>
            As we observed earlier, players tend to spend less time in the game on weekends compared to weekdays. However, when looking at the above graph, we can see that they spend more during weekends despite spending less time. This could be due to different player segments being active on weekends versus weekdays. For instance, if whale players, who tend to spend large amounts, are more active on weekends, this could explain the trend. Another possibility is that there are more special content and events over the weekend that encourage players to spend more. If players are spending more to participate in these events or during the events themselves, this could also contribute to the observed pattern.
        <p></p>
            Additionally, we can analyze the daily ARPDAU values on a country-by-country basis:
        </div>
        """,
        unsafe_allow_html=True,
    )

    country_arpdau = st.selectbox(
        "Please select a country:",
        ["Mercury", "Venus", "Pluton", "Saturn", "Uranus"],
        key="selectbox3",
    )
    df28_2 = get_graph28_2()
    a = df28_2[df28_2["country"] == country_arpdau]
    avg_arpdau_country = a["ARPDAU"].mean()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=a["date"],
            y=a["ARPDAU"],
            mode="lines+markers",
            name="ARPDAU",
            line=dict(color="black"),
            marker=dict(color="magenta"),
        )
    )

    fig.update_layout(
        title=f"ARPDAU for {country_arpdau}",
        xaxis_title="Date",
        yaxis_title="ARPDAU",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"Average ARPDAU: {avg_arpdau_country:.2f}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=1.1,
        showarrow=False,
        font=dict(size=12, color="darkblue"),
        align="right",
    )

    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Looking at the chart above, we can observe an increasing trend in the countries of Mercury, Venus, and Pluto, while no such trend is seen in Saturn and Uranus. The highest ARPDAU values are generated from Mercury and Venus countries.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 29
    st.subheader(":blue[29) PlaytimeDAU]")
    df29 = pd.read_pickle("data/graph29.pkl")
    avg_playtime_dau = df29["playtime_dau"].mean()
    avg_total_timespent = df29["total_time_spent"].mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df29["date"],
            y=df29["total_time_spent"],
            mode="lines+markers",
            name="Playtime",
            line=dict(color="black"),
            marker=dict(color="black"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df29["date"],
            y=df29["playtime_dau"],
            mode="lines+markers",
            name="PlaytimeDAU",
            line=dict(color="orange"),
            marker=dict(color="orange"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Playtime & PlaytimeDAU",
        xaxis_title="Date",
        yaxis_title="Playtime",
        yaxis2=dict(
            title="PlaytimeDAU",
            overlaying="y",
            side="right",
            showgrid=False,
            showline=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"PlaytimeDAU: {avg_playtime_dau:.2f}",
        xref="paper",
        yref="paper",
        x=0.95,
        y=1.1,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
    )

    fig.add_annotation(
        text=f"Playtime: {avg_total_timespent:.2f}",
        xref="paper",
        yref="paper",
        x=0.96,
        y=1.2,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
    )

    st.plotly_chart(fig)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In the chart above, you can see both the daily total Playtime and the PlaytimeDAU trends. Although Playtime increased throughout May, it decreased in June. On the other hand, the PlaytimeDAU line shows a declining trend. Previously, when we analyzed the DAU graph, we found a similar pattern where it increased during May but decreased in June. Therefore, the declining trend in the PlaytimeDAU graph suggests that the increase in DAU was higher than the increase in Playtime during May, while in June, the decrease in DAU was at a slower rate compared to the decrease in Playtime.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 30
    st.subheader(":blue[30) ARPInstall]")
    df30 = pd.read_pickle("data/graph30.pkl")
    avg_arpinstall = df30["arp_install"].mean()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df30["date"],
            y=df30["arp_install"],
            mode="lines+markers",
            name="ARPInstall",
            line=dict(color="black"),
            marker=dict(color="limegreen"),
        )
    )

    fig.update_layout(
        title="Daily ARPInstall",
        xaxis_title="Date",
        yaxis_title="ARPInstall",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"ARPInstall: {avg_arpinstall:.2f}",
        xref="paper",
        yref="paper",
        x=0.90,
        y=0.99,
        showarrow=False,
        font=dict(size=12, color="darkblue"),
        align="right",
    )

    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            The positive trend in the ARPInstall graph indicates that monetization strategies and marketing campaigns have been successful. The chart above shows that there is a higher number of whale players among new users, and these new players are making larger spending amounts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 31
    st.subheader(":blue[31) CPI]")
    df31 = pd.read_pickle("data/graph31.pkl")
    avg_cpi = df31.cpi.mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df31["date"],
            y=df31["cpi"],
            mode="lines+markers",
            name="CPI",
            line=dict(color="black"),
            marker=dict(color="orange"),
        )
    )

    fig.update_layout(
        title="Daily CPI",
        xaxis_title="Date",
        yaxis_title="CPI",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"CPI: {avg_cpi:.2f}",
        xref="paper",
        yref="paper",
        x=0.90,
        y=1.1,
        showarrow=False,
        font=dict(size=15, color="darkblue"),
        align="right",
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            Looking at the CPI graph, we can see that the average CPI value is 4.48. On the other hand, in the previous ARPInstall graph, the average ARPInstall value is 1.08. Therefore, we can once again conclude that marketing expenses are higher than revenue. Another observation from the graph is that the CPI value remained almost constant during the first three weeks of May, but showed a sharp drop in the last week of May. This could be due to the increase in download numbers during the last week of May. As we saw in the graph related to download numbers (Graph 24), there was a sudden spike in downloads during the last week of May. Therefore, the above graph aligns with our previous observations.
        <p></p>
            Now, let's check out the daily CPI values by country and take a look at the network breakdowns for each country:
        </div>
        """,
        unsafe_allow_html=True,
    )

    df31_2 = get_graph31_2()
    country_cpi = st.selectbox(
        "Please select a country:",
        ["Mercury", "Venus", "Pluton", "Saturn", "Uranus"],
        key="selectbox4",
    )
    a = df31_2[df31_2["country"] == country_cpi]
    networks = a.network.unique()
    fig = go.Figure()
    network_roas = dict()
    annot_loc = 1.05
    for network in networks:
        temp_df = a[a["network"] == network]
        fig.add_trace(
            go.Scatter(
                x=temp_df["date"],
                y=temp_df["cpi"],
                mode="lines+markers",
                name=network,
            )
        )

        avg_roas = temp_df["cpi"].mean()
        fig.add_annotation(
            text=f"Average {network} CPI: {avg_roas:.2f}",
            xref="paper",
            yref="paper",
            x=0.09,
            y=annot_loc,
            showarrow=False,
            font=dict(size=12, color="darkblue"),
            align="right",
        )
        annot_loc -= 0.1

    fig.update_layout(
        title=f"CPI for {country_cpi}",
        xaxis_title="Date",
        yaxis_title="CPI",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            For example, by looking at the graph above, we can see that for Mercury, Venus, and Saturn, the most efficient channel is Jessie, while for Pluto and Uranus, Sid proves to be the most effective channel.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 32
    st.subheader(":blue[32) Stickiness]")
    df32 = pd.read_pickle("data/graph32.pkl")
    avg_stickiness = df32.stickiness.mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df32["date"],
            y=df32["stickiness"],
            mode="lines+markers",
            name="Stickiness",
            line=dict(color="black"),
            marker=dict(color="orange"),
        )
    )

    fig.update_layout(
        title="Daily Stickiness",
        xaxis_title="Date",
        yaxis_title="Stickiness",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )

    fig.add_annotation(
        text=f"Stickiness: {avg_stickiness:.2f}",
        xref="paper",
        yref="paper",
        x=0.90,
        y=1.1,
        showarrow=False,
        font=dict(size=15, color="darkblue"),
        align="right",
    )

    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            The stickiness metric is an important measure related to retention and engagement. Therefore, it is desirable for stickiness to be as high as possible. Looking at the stickiness graph above, we can observe a positive trend, with stickiness steadily increasing.
        <p></p>
            However, when we calculate the average daily stickiness, we get a value of 3%, which is quite low. The first option to increase stickiness would be to enhance the personalization within the game. Many years ago, people were deeply engaged in MMORPGs because these games offered maximum levels of personalization. As long as you personalize a game correctly, the player will form a deeper connection with it, leading to greater loyalty. Therefore, the first strategy to boost stickiness would be to increase the level of personalization in the game.
        <p></p>
            Another approach would be to regularly release updates with reasonable frequency and introduce new features. For example, consistently fixing bugs through updates and addressing feedback from players. This way, the development process will align with players' expectations, and their engagement with the game will increase.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 33
    st.subheader(":blue[33) PLTV Segmentation]")
    df33 = pd.read_pickle("data/graph33.pkl")
    grouped_df = (
        df33.groupby("segment", observed=False)
        .agg({"total_payment": "sum", "total_transaction": "mean"})
        .reset_index()
    )

    pie1 = go.Pie(
        labels=grouped_df["segment"],
        values=grouped_df["total_payment"],
        textinfo="label+percent",
        insidetextorientation="radial",
        marker=dict(colors=colors.qualitative.Pastel),
        hole=0.4,
    )

    pie2 = go.Pie(
        labels=grouped_df["segment"],
        values=grouped_df["total_transaction"],
        textinfo="label+percent",
        insidetextorientation="radial",
        marker=dict(colors=colors.qualitative.Pastel),
        hole=0.4,
    )

    grouped_df = (
        df33.groupby("segment", observed=False)
        .agg({"average_order_value": ["mean", "count"]})
        .reset_index()
    )

    pie3 = go.Pie(
        labels=grouped_df["segment"],
        values=grouped_df.average_order_value["mean"],
        textinfo="label+percent",
        insidetextorientation="radial",
        marker=dict(colors=colors.qualitative.Pastel),
        hole=0.4,
    )

    pie4 = go.Pie(
        labels=grouped_df["segment"],
        values=grouped_df.average_order_value["count"],
        textinfo="label+percent",
        insidetextorientation="radial",
        marker=dict(colors=colors.qualitative.Pastel),
        hole=0.4,
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "domain"}, {"type": "domain"}],
            [{"type": "domain"}, {"type": "domain"}],
        ],
        subplot_titles=[
            "Total Payment Sum by Segment",
            "Average Transaction Count by Segment",
            "Average Order Value by Segment",
            "Distribution of Segments",
        ],
        horizontal_spacing=0,  # Yatay boşluk (default: 0.2)
        vertical_spacing=0.1,
    )

    fig.add_trace(pie1, row=1, col=1)
    fig.add_trace(pie2, row=1, col=2)
    fig.add_trace(pie3, row=2, col=1)
    fig.add_trace(pie4, row=2, col=2)

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=50),
        width=1000,
        height=750,
    )
    left_part1, right_part1 = st.columns([0.15, 0.7])
    right_part1.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            We can observe the Pareto Principle in these graphs. For example, players in Segment A make up only 20% of the total player base, yet they account for 80% of the revenue. Additionally, users in Segment A have an Average Order Value (AOV) that is twice as high as Segment B, three times higher than Segment C, and nearly five times higher than Segment D.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 33.2 Segmented DAU
    st.markdown(
        """
        <div class="justified-text">
        When we examine the daily DAU values of the segments we have identified, we arrive at the graph below. However, please do not confuse these DAU values with the ones we reviewed earlier. The values shown below represent the DAU of users who have made at least one purchase:
        </div>
        """,
        unsafe_allow_html=True,
    )
    df33_2 = pd.read_pickle("data/graph33_2.pkl")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "A"]["event_date"],
            y=df33_2[df33_2["segment"] == "A"]["dau"],
            mode="lines+markers",
            name="A segment DAU",
            line=dict(color="black"),
            marker=dict(color="black"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "B"]["event_date"],
            y=df33_2[df33_2["segment"] == "B"]["dau"],
            mode="lines+markers",
            name="B segment DAU",
            line=dict(color="orange"),
            marker=dict(color="orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "C"]["event_date"],
            y=df33_2[df33_2["segment"] == "C"]["dau"],
            mode="lines+markers",
            name="C segment DAU",
            line=dict(color="purple"),
            marker=dict(color="purple"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "D"]["event_date"],
            y=df33_2[df33_2["segment"] == "D"]["dau"],
            mode="lines+markers",
            name="D segment DAU",
            line=dict(color="darkblue"),
            marker=dict(color="darkblue"),
        )
    )

    fig.update_layout(
        title="DAU by Segments",
        xaxis_title="Date",
        yaxis_title="DAU",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        <div class="justified-text">
        The DAU graphs broken down by segment above show trends that are quite similar to the earlier DAU graph. In all of the graphs, we can observe a positive trend in DAU values throughout May, followed by a general decline in June.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 33.3 Segmented Total Payments
    st.markdown(
        """
        <div class="justified-text">
        Similarly, we can analyze the total daily payment amount made by these segments on a daily basis:
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "A"]["event_date"],
            y=df33_2[df33_2["segment"] == "A"]["total_payment"],
            mode="lines+markers",
            name="A segment Total Payment",
            line=dict(color="black"),
            marker=dict(color="black"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "B"]["event_date"],
            y=df33_2[df33_2["segment"] == "B"]["total_payment"],
            mode="lines+markers",
            name="B segment Total Payment",
            line=dict(color="orange"),
            marker=dict(color="orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "C"]["event_date"],
            y=df33_2[df33_2["segment"] == "C"]["total_payment"],
            mode="lines+markers",
            name="C segment Total Payment",
            line=dict(color="purple"),
            marker=dict(color="purple"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df33_2[df33_2["segment"] == "D"]["event_date"],
            y=df33_2[df33_2["segment"] == "D"]["total_payment"],
            mode="lines+markers",
            name="D segment Total Payment",
            line=dict(color="darkblue"),
            marker=dict(color="darkblue"),
        )
    )

    fig.update_layout(
        title="Total Payment by Segments",
        xaxis_title="Date",
        yaxis_title="Total Payment",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True),
        yaxis=dict(showgrid=False, showline=True),
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        <div class="justified-text">
        In the graphs above, just like in the DAU graphs, we can see that the total payment amount for all segments increased throughout May, but experienced a decline in June.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 34
    st.subheader(":blue[34) RFM Segmentation]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            We can further refine the segmentation we did with PLTV by using RFM Analysis on a user level. With RFM Analysis, we will assign each user a recency, frequency, and monetary score. Then, based on the recency and frequency scores, we will perform the segmentation according to the table below:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part1, right_part1 = st.columns([0.1, 0.4])
    right_part1.image("images/part_i/rfm.png", width=750)
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            The reason we don't use the monetary score in the segmentation process is that recency and frequency scores are more important for us. Additionally, a player with high recency and frequency scores typically also has a high monetary score. Therefore, we are using only the recency and frequency scores.
        <p></p>
            When we segment based on the RF scores we have, we obtain the tree map below:
        </div>
        """,
        unsafe_allow_html=True,
    )

    df34 = pd.read_pickle("data/graph34.pkl")
    fig = px.treemap(
        df34,
        path=["segments"],
        values="count",
        color="count",
        color_continuous_scale="Viridis",
        title="RFM Segments",
    )

    fig.update_layout(
        title_font=dict(size=15, family="Arial, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            The segments here are particularly valuable for the marketing department. For example, the IDs of users in the "can't_loose" segment can be provided to the marketing team, who can then work on re-engaging these players. Meanwhile, users in the "loyal_customers" and "potential_loyalist" segments could be targeted with various reward strategies.
        </div>
        """,
        unsafe_allow_html=True,
    )


###############################
# PART II: A/B TEST
###############################

with part2:

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In this section, we will analyze the control (A) and experiment (B) groups in terms of financial and engagement metrics. Since the company did not provide values such as the minimum detectable effect (MDE) or alpha, we will assume these values ourselves as we proceed. Additionally, before conducting any A/B tests, we will perform an A/A test for both the control and experiment groups. This will help ensure that if the test statistics are overly sensitive, the A/A test will fail, indicating an issue.
        <p></p>
            The methods we will use are the Z-test, t-test, and Mann-Whitney U test. For the A/A tests, we will use the Mann-Whitney U test. For the A/B test, we will use either the Z-test or the t-test, depending on the specific requirements of the test.
        <p></p>
            To make the process easier, we will use a function I’ve written called `ab_result()`. This function will take the A and B groups as input and sequentially perform the A/A test followed by the A/B test. If the A/A test fails, the A/B test will not be conducted. However, if the A/A test is successful, the A/B test will proceed, and a visual plot will be generated to help us better understand the results.
        <p></p>
            In the analysis we will perform shortly, I won’t mention the A/A test unless it fails. However, please keep in mind that an A/A test is conducted before every A/B test.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Test 1
    st.subheader(":blue[1) Total Time Spent]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (87.286, 89.342). When examined separately, the average total time spent by a user in group A is 99.964, while the average total time spent by a user in group B is 76.356. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method. 
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/1.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(1.4705639822420505e-113)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group spends less time in the game.**"
    )

    # Test 2
    st.subheader(":blue[2) Number of Sessions]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (3.026, 3.097). When examined separately, the average number of sessions for a user in group A is 3.456, while the average number of sessions for a user in group B is 2.656. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/2.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(4.4501436892908715e-109)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a lower number of sessions.**"
    )

    # Test 3
    st.subheader(":blue[3) Average Time Spent Per Session]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (26.91, 26.99). When examined separately, the average time spent per session for a user in group A is 26.98, while the average time spent per session for a user in group B is 26.93. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/3.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.15787022164233722)  >  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) CANNOT be rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we find that there is no statistically significant difference between the experiment (B) and control (A) groups in terms of average time spent per session.**"
    )

    # Test 4
    st.subheader(":blue[4) Level]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (217, 222). When examined separately, the average in-game level for a user in group A is 283, while the average in-game level for a user in group B is 154. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method: 
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/4.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.0)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a lower in-game level.**"
    )

    # Test 5
    st.subheader(":blue[5) Revenue per User]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (151, 177). When examined separately, the average total revenue generated by a user in group A is 158, while the average total revenue generated by a user in group B is 169. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/5.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.3739798636573419)  >  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) CANNOT be rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we find that there is no statistically significant difference in revenue per user between the experiment (B) and control (A) groups.**"
    )

    # Test 6
    st.subheader(":blue[6) Number of Transactions]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (11.78, 13.02). When examined separately, the average total number of transactions for a user in group A is 11.47, while the average total number of transactions for a user in group B is 13.20. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/6.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.005128313625524875)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a higher number of transactions.**"
    )

    # Test 7
    st.subheader(":blue[7) AOV]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (9.16, 9.67). When examined separately, the average AOV for a user in group A is 9.73, while the average AOV for a user in group B is 9.15. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/7.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.026560784774499985)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a lower AOV.**"
    )

    # Test 8
    st.subheader(":blue[8) Purchase Frequency]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            When we combine the A and B groups and calculate the confidence interval with an alpha of 0.05, we find it to be (0.001949, 0.002154). When examined separately, the average purchase frequency for a user in group A is 0.001897, while the average purchase frequency for a user in group B is 0.002184. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/8.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.00512831362552487)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a higher purchase frequency.**"
    )

    # Test 9
    st.subheader(":blue[9) DAU]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            We observe that the average DAU for group A is 1,780,185, while the average DAU for group B is 1,332,930. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method with an alpha of 0.05: 
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/9.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(1.563369276418578e-08)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a lower DAU value.**"
    )

    # Test 10
    st.subheader(":blue[10) ARPDAU]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            We observe that the average ARPDAU for group A is 0.5612, while the average ARPDAU for group B is 0.7696. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method with an alpha of 0.05: 
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/10.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(1.6141170753003553e-13)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a higher ARPDAU value.**"
    )

    # Test 11
    st.subheader(":blue[11) Daily Revenue]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            We observe that the daily average revenue for group A is 6,106, while the daily average revenue for group B is 7,679. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method with an alpha of 0.05:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/11.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.0005129920795266452)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a higher daily revenue.**"
    )

    # Test 12
    st.subheader(":blue[12) ARPInstall]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            We observe that the daily ARPInstall for group A is 4.55, while the daily ARPInstall for group B is 5.99. To determine if the difference is statistically significant, we will conduct an A/B test using the t-test method with an alpha of 0.05: 
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p>
            <strong><span style="font-size:20px;">NOTE</span></strong>: The reason we are using the t-test instead of the Z-test is that we only have data for 28 days. When the sample size is less than 30, it is recommended to use the t-test rather than the Z-test. In this case, if we were to use the Z-test, we would obtain a "H0 rejected" result. However, by using the t-test, we arrive at a "H0 NOT rejected" result. As the sample size increases, the t-test approaches the Z-test; but in this example, we have a small amount of data.
        </p>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/12.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- t test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(0.05229060907018408)  >  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) CANNOT be rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that there is no statistically significant difference in ARPInstall between the experiment (B) and control (A) groups.**"
    )

    # Test 13
    st.subheader(":blue[13) Number of Transactions per DAU]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            We observe that the transaction rate per DAU for group A is 0.0412, while the transaction rate per DAU for group B is 0.0602. To determine if the difference is statistically significant, we conduct an A/B test using the Z-test method with an alpha of 0.05:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part2, right_part2 = st.columns(2)
    left_part2.image("images/part_ii/13.png", width=750)
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown(" ")
    right_part2.markdown("- A/A Test passed successfully!")
    right_part2.markdown("- z test is being used...")
    right_part2.markdown(
        "- H0 : M1 = M2  (Control(A) and experimental(B) groups have the same distribution.)"
    )

    right_part2.markdown(
        "- H1 : M1 != M2  (Control(A) and experimental(B) groups DON'T have the same distribution.)"
    )

    right_part2.markdown("- p-value(1.7826052201525774e-33)  <=  alpha(0.05)")

    right_part2.markdown(
        "- The null hypothesis(H0) is rejected at a significance level of 0.05"
    )

    right_part2.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:4px;
        padding-top:10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    right_part2.markdown(
        "**As a result of the test, we observe that the experiment (B) group has a higher transaction rate per DAU.**"
    )

    # Sonuc
    st.markdown(
        """
        <p>
            <strong><span style="font-size:20px;">As a result:</span></strong>:
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <ul style="padding-left:4px; padding-top:1px; list-style-type:none;">
            <li style="margin-bottom: 10px;">
                <span style="color: green; font-weight: bold;">&#10004;</span>
                If the game is monetization-focused, meaning the main goal is to increase revenue and maximize revenue per user, then the experiment (B) group is performing better in this regard.
            </li>
            <li>
                <span style="color: green; font-weight: bold;">&#10004;</span>
                If the game is engagement-focused, meaning the priority is to increase user interaction and retention, then the control (A) group is yielding better results in this regard.
            </li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        
            Of course, the reliability of these results should also be tested. For example, users might be reacting simply to the change itself, which could be a case of what's known as the "*novelty effect*". Therefore, the tests we conducted here should be repeated over a longer time period, and the results should be checked to see if there are any differences.
        
        """,
        unsafe_allow_html=True,
    )


###############################
# PART III: MODEL
###############################
with part3:

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            In this section, we aim to build a binary classification model that predicts whether a player will make a purchase by the end of a 30-day period. The `d30_revenue` column in the SQL table is not a binary variable, so we create a new binary variable called `purchased` and use it as our target variable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # i) EDA
    st.markdown(
        """
        <p>
            <strong><span style="font-size:30px; color:DodgerBlue">i) EDA</span></strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .custom-bullet {
            padding-left: 4px;
            margin-top: -30px;
        }
        .custom-bullet ul {
            list-style-type: disc;  
            margin-top: -20px;  
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="custom-bullet">
        <ul>
        <li>First, we examine the target variable, `purchased`, and find that only 7.83% of users made a purchase. This indicates that we are dealing with an imbalanced dataset. This insight helps us determine which metrics to focus on when training our model. In imbalanced situations, metrics like ROC AUC and F1 score will be more meaningful for us.</li>
        <li>Next, we check for null values in the dataset. Fortunately, the dataset is well-structured and does not contain any null values.</li>
        <li>We use the `describe()` function to check for any anomalies in the variables. For example, if the `age` column contains a 0 value anywhere, we would be able to identify it using the `describe()` function. Fortunately, at this stage, there are no abnormalities in the data.</li>
        <li>We create a heatmap to visualize the correlation between the numeric columns. Here, we discover strong correlations between columns such as `time_spend`, `coin_spend`, `coin_earn`, `level_success`, `level_fail`, `level_start`, `booster_spend`, and `booster_earn`. We can use this information later when creating features:</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.15, 0.4])
    right_part3.image("images/part_iii/correlation.png", width=650)

    st.markdown(
        """
        <div class="custom-bullet">
        <ul>
        <li>We perform an outlier detection and find outliers in the columns `time_spend`, `coin_spend`, `coin_earn`, `level_success`, `level_fail`, `level_start`, `booster_spend`, `booster_earn`, `coin_amount`, `event_participate`, and `shop_open`. Outliers can negatively affect the performance of the model we are about to build, so we replace them using the IQR (Interquartile Range) method with thresholds at the 1st and 99th percentiles.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ii) Feature Engineering
    st.markdown(
        """
        <p>
            <strong><span style="font-size:30px; color:DodgerBlue">ii) Feature Engineering</span></strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="custom-bullet">
        <ul>
        <li>We create a categorical column based on users' ages, with values categorized as "young", "early_adult", "mid_adult", "late_adult", and "old".</li>
        <li>Next, we use one-hot encoding to convert all categorical variables into numerical features.</li>
        <li>We experiment with creating new variables based on the highly correlated features we identified. If any of the newly created variables improve the model's ROC AUC score, the `feature_creator()` function notifies us. Using this function, we discover that new features like `time_spend/age` and `coin_spend/coin_amount` could improve the model's ROC AUC score. We then add these new features to the `x` dataframe.</li>
        <li>Finally, we split the dataframe into training and test sets, with 70% of the data used for training and 30% for testing, to be used in the model selection phase.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # iii) Model Selection
    st.markdown(
        """
        <p>
            <strong><span style="font-size:30px; color:DodgerBlue">iii) Model Selection</span></strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            At this stage, we try several models. When we rank the models based on ROC AUC and F1 score criteria, we get the following results:
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = {
        "Model": [
            "CatBoost",
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "ShallowNN",
            "SVM",
            "LogisticRegression",
            "GBM",
        ],
        "ROC AUC": [0.87, 0.87, 0.87, 0.85, 0.85, 0.83, 0.83, 0.80],
        "F1": [0.32, 0.31, 0.28, 0.26, 0.22, 0.34, 0.26, 0.29],
    }

    models = pd.DataFrame(data).set_index("Model")
    left_part3, center_part3, right_part3 = st.columns([0.52, 0.53, 0.6])
    center_part3.table(models)

    st.markdown(
        """
        <div class="justified-text">
        <p>
            <strong><span style="font-size:20px;">NOTE</span></strong>: The reason we focus on the ROC AUC and F1 metrics is because the dataset is generally imbalanced. When dealing with imbalanced datasets, techniques like undersampling and oversampling can be used to balance the data. However, for this project, I will not be applying these methods.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
            .center-text {
                text-align: center;
            }
        </style>
        <div class="center-text">
        As a result, we decide to go with the CatBoost model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # iv) Hyperparameter Tuning
    st.markdown(
        """
        <p>
            <strong><span style="font-size:30px; color:DodgerBlue">iv) Hyperparameter Tuning</span></strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="justified-text">
        For hyperparameter optimization, I will use Optuna, an open-source library. Thanks to its tree-structured algorithm, Optuna performs a more efficient and intelligent search across hyperparameters, resulting in significant time and resource savings compared to methods like GridSearch or RandomSearch.
        <p></p>
        This time, I will split the dataset into a train_test set and a 10% validation set. I will train the model using 3-fold cross-validation on the train_test set, while keeping the 10% validation set completely separate and not showing it to the model during the training process.
        <p></p>
        After setting up the necessary configurations for Optuna with two functions, objective() and logging_callback(), I instruct Optuna to optimize the model in a way that maximizes the ROC AUC score. As a result, Optuna successfully improves the ROC AUC score by approximately 1% and provides the best parameters it found, as shown below:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([40, 60])
    right_part3.markdown(
        """
        <div class="justified-text">
        <br>
        {'objective': 'CrossEntropy', <br>
        'colsample_bylevel': 0.08140222490192758, <br>
        'depth': 11, <br>
        'boosting_type': 'Ordered', <br>
        'bootstrap_type': 'Bayesian', <br>
        'bagging_temperature': 0.1315082098008834}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # v) Final Model
    st.markdown(
        """
        <p>
            <strong><span style="font-size:30px; color:DodgerBlue">v) Final Model</span></strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="justified-text">
        After the hyperparameter tuning process, we can now test the model on the validation set, which it has never seen before:
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = {
        "0": [
            "Train",
            "Validation",
        ],
        "ROC AUC Scores": [0.88, 0.87],
    }

    scores = pd.DataFrame(data).set_index("0")
    left_part3, center_part3, right_part3 = st.columns([0.55, 0.22, 0.6])
    center_part3.table(scores)

    st.markdown(
        """
        <div class="center-text">
        As a result, we achieve a score of 87% on the validation set.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # vi) Feature Analysis
    st.markdown(
        """
        <p>
            <strong><span style="font-size:30px; color:DodgerBlue">vi) Feature Analysis</span></strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="justified-text">
        To explain that all of this process wasn't just some magic trick, we need to understand how the model works ourselves. At this point, we can use a library called SHAP (SHapley Additive exPlanations). This open-source Python package is extremely useful for visualizing which features are influencing the output and how they are doing so. With SHAP's BeeSwarm plot, we can identify the most important features and see how they relate to the output.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/beeswarm.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        For example, by looking at the BeeSwarm plot above, we can see that the top three most important variables are coin_earn, country_Zephyra, and the coin_spend/coin_amount feature we created. To examine the impact of these variables on the output in more detail, we can use the scatter function. For instance, there is a direct proportional relationship between the coin_earn variable and purchases, as shown in the graph below:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/coin_earn.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        Another example is the level_success variable. There is a negative relationship between level_success and purchases. In other words, as level_success increases, the likelihood of users making a purchase decreases:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/level_success.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        Now, let's take a look at how using iOS affects the likelihood of making a purchase. We have a binary variable called `platform_ios` that we can examine for this:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/platform_ios.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        As you can see, and as we expected, using iOS is one of the factors that increases the likelihood of making a purchase.
        </div>
        """,
        unsafe_allow_html=True,
    )


###############################
# PART IV: PREDICTION
###############################


scaler = get_scaler()
model = get_model()

with part4:
    st.markdown(
        """
        <div class="justified-text">
        In this section, you can view the model's prediction results by entering the appropriate values in the fields below.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_part4, right_part4 = st.columns(2)
    age = left_part4.number_input(
        "Please enter the age variable:", min_value=5, max_value=90, step=1, value=17
    )
    time_spend = left_part4.number_input(
        "Please enter the time_spend variable:",
        min_value=0,
        max_value=120000,
        step=1000,
        value=38890,
    )
    coin_spend = left_part4.number_input(
        "Please enter the coin_spend variable:",
        min_value=0,
        max_value=350000,
        step=5000,
        value=117500,
    )
    coin_earn = left_part4.number_input(
        "Please enter the coin_earn variable:",
        min_value=0,
        max_value=375000,
        step=5000,
        value=125640,
    )
    level_success = left_part4.number_input(
        "Please enter the level_success variable:",
        min_value=0,
        max_value=1000,
        step=1,
        value=255,
    )
    level_fail = left_part4.number_input(
        "Please enter the level_fail variable:",
        min_value=0,
        max_value=1000,
        step=1,
        value=0,
    )
    level_start = left_part4.number_input(
        "Please enter the level_start variable:",
        min_value=0,
        max_value=1000,
        step=1,
        value=278,
    )
    booster_spend = left_part4.number_input(
        "Please enter the booster_spend variable:",
        min_value=0,
        max_value=500,
        step=50,
        value=110,
    )
    booster_earn = right_part4.number_input(
        "Please enter the booster_earn variable:",
        min_value=0,
        max_value=500,
        step=50,
        value=205,
    )
    coin_amount = right_part4.number_input(
        "Please enter the coin_amount variable:",
        min_value=0,
        max_value=37500,
        step=1750,
        value=12262,
    )
    shop_open = right_part4.number_input(
        "Please enter the shop_open variable:",
        min_value=0,
        max_value=20,
        step=1,
        value=1,
    )
    event_participate = right_part4.selectbox(
        "Please select the event_participate variable:", ["Yes", "No"]
    )
    if event_participate == "Yes":
        event_participate = 1
    else:
        event_participate = 0
    platform = right_part4.selectbox(
        "Please select the platform variable:", ["ios", "android"]
    )
    network = right_part4.selectbox(
        "Please enter the network variable:",
        [
            "Oyster",
            "Piggy",
            "Cupboard",
            "Dynamite",
            "Bird",
            "Vase",
            "Owl",
            "Box",
            "Curtain",
            "Egg",
            "Mailbox",
            "Grass",
            "Honey",
            "Potion",
        ],
    )
    country = right_part4.selectbox(
        "Please select the country variable:",
        [
            "Zephyra",
            "Thalassia",
            "Sunridge",
            "Amaryllis",
            "Brighthaven",
            "Luminara",
            "Gleamwood",
            "Azurelia",
            "Eldoria",
            "Windemere",
            "Rosewyn",
            "Floravia",
            "Glimmerdell",
            "Emberlyn",
            "Frostford",
            "Crystalbrook",
            "Seraphina",
            "Silvermist",
            "Moonvale",
            "Starcliff",
        ],
    )

    user = pd.DataFrame(
        {
            "age": age,
            "time_spend": time_spend,
            "coin_spend": coin_spend,
            "coin_earn": coin_earn,
            "level_success": level_success,
            "level_fail": level_fail,
            "level_start": level_start,
            "booster_spend": booster_spend,
            "booster_earn": booster_earn,
            "coin_amount": coin_amount,
            "event_participate": event_participate,
            "shop_open": shop_open,
            "network": network,
            "country": country,
        },
        index=[0],
    )
    user.loc[(user["age"] <= 28), "age_cat"] = "young"
    user.loc[(user["age"] > 28) & (user["age"] <= 41), "age_cat"] = "early_adult"
    user.loc[(user["age"] > 41) & (user["age"] <= 53), "age_cat"] = "mid_adult"
    user.loc[(user["age"] > 53) & (user["age"] <= 66), "age_cat"] = "late_adult"
    user.loc[(user["age"] > 66), "age_cat"] = "old"

    user = pd.get_dummies(
        user,
        columns=[col for col in user.columns if user[col].dtypes == "O"],
    )

    model_input = pd.DataFrame(
        {
            "age": 0,
            "time_spend": 0,
            "coin_spend": 0,
            "coin_earn": 0,
            "level_success": 0,
            "level_fail": 0,
            "level_start": 0,
            "booster_spend": 0,
            "booster_earn": 0,
            "coin_amount": 0,
            "event_participate": 0,
            "shop_open": 0,
            "platform_android": 0,
            "platform_ios": 0,
            "network_Bird": 0,
            "network_Box": 0,
            "network_Cupboard": 0,
            "network_Curtain": 0,
            "network_Dynamite": 0,
            "network_Egg": 0,
            "network_Grass": 0,
            "network_Honey": 0,
            "network_Mailbox": 0,
            "network_Owl": 0,
            "network_Grass": 0,
            "network_Oyster": 0,
            "network_Piggy": 0,
            "network_Potion": 0,
            "network_Vase": 0,
            "country_Amaryllis": 0,
            "country_Azurelia": 0,
            "country_Brighthaven": 0,
            "country_Crystalbrook": 0,
            "country_Eldoria": 0,
            "country_Emberlyn": 0,
            "country_Floravia": 0,
            "country_Frostford": 0,
            "country_Gleamwood": 0,
            "country_Glimmerdell": 0,
            "country_Luminara": 0,
            "country_Moonvale": 0,
            "country_Rosewyn": 0,
            "country_Seraphina": 0,
            "country_Silvermist": 0,
            "country_Starcliff": 0,
            "country_Sunridge": 0,
            "country_Thalassia": 0,
            "country_Windemere": 0,
            "country_Zephyra": 0,
            "age_cat_early_adult": 0,
            "age_cat_late_adult": 0,
            "age_cat_mid_adult": 0,
            "age_cat_old": 0,
            "age_cat_young": 0,
        },
        index=[0],
    )

    for col in user.columns:
        model_input[col] = user[col]

    model_input = pd.DataFrame(
        scaler.transform(model_input), columns=model_input.columns
    )

    model_input["time_spend/age"] = model_input["time_spend"] / model_input["age"]
    model_input["coin_spend/coin_amount"] = (
        model_input["coin_spend"] / model_input["coin_amount"]
    )

    if st.button("Predict!"):
        prediction = model.predict(model_input)
        if prediction == 1:
            st.success(f"This player will purchase! :)")
        else:
            st.success(f"This player won't purchase! :)")
        st.balloons()
