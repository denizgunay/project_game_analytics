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


# Layout
st.set_page_config(layout="wide")

# Banner
custom_html = """
<div class="banner">
    <img src="https://s3-us-west-2.amazonaws.com/cbi-image-service-prd/original/17a1efd7-2ba4-462e-9f6e-b53c36fbfd7d.png" alt="Banner Image">
</div>
<style>
    .banner {
        width: 160%;
        height: 200px;
        overflow: hidden;
        background: linear-gradient(to right, white, blue);
    }
    .banner img {
        width: 23%;
        object-fit: contain;
    }
</style>
"""

st.components.v1.html(custom_html)

# Header
st.header(":blue[Dream] Games Data Scientist Case Study")

# Audio
st.audio("audio/royal_match.mp3", format="audio/mpeg", loop=True, autoplay=True)


# Tabs
part1, part2, part3, part4 = st.tabs(
    ["Part I: Analiz", "Part II: A/B Test", "Part III: Model", "Part IV: Tahmin"]
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
    """<a href="#dream-games-data-scientist-case-study" style="text-decoration:none;">
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
            Bu bölümde, Row Match oyunundaki mevcut durumu çeşitli metrikler ve kohort analizleriyle anlamaya çalışacağız.
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
            Grafiği incelediğimizde, D1 için retention rate’in %55, D7 için %39, ve D28 için ise %30 civarında olduğunu gözlemlemekteyiz. GameAnalytics’in 2019 yılında yayımladığı Player Retention Report’a göre, en iyi performans gösteren oyunların retention rate’lerinin D1 için ortalama %40, D7 için %15 ve D28 için %6.5 olduğu bilinmektedir. Bu bilgiler ışığında, Row Match’in endüstriye kıyasla çok daha iyi bir retention rate’e sahip olduğunu söyleyebiliriz. Retention rate’i daha da geliştirmek için, oyunculara segmentasyon yapıp loyal oyuncu gruplarına ücretsiz öğeler verilebilir ya da oyuncuların oyunda geçirdikleri zamana göre madalya sistemi geliştirilebilir. Bu şekilde, oyuncuların oyuna olan bağlılıkları pekiştirilip retention rate artırılabilir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 2
    st.subheader(":blue[2) Ülkelere göre oyuncuların ortalama yaşı]")

    df2 = pd.read_pickle("data/graph2.pkl")
    fig = go.Figure(
        data=[
            go.Bar(
                x=df2["country"],
                y=df2["avg_age"],
                marker=dict(color="darkblue"),
                text=df2["avg_age"].round(),
                textposition="inside",
            )
        ]
    )

    fig.update_layout(
        title="Average Age by Country",
        title_font=dict(size=15, family="Arial, sans-serif"),
        xaxis_title="Country",
        yaxis_title="Average Age",
        xaxis_title_font=dict(size=18, family="Arial, sans-serif"),
        yaxis_title_font=dict(size=18, family="Arial, sans-serif"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50),
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
            Ülkelere göre oyuncuların ortalama yaşlarına baktığımızda çok ilginç bir sonuçla karşılaşıyoruz. Hemen hemen tüm ülkelerde oyuncuların ortalama yaşı 47. Dolayısıyla, genel oyuncu profilinin 40'lı yaşlarda ve muhtemelen iş hayatında yer alan bir yetişkin olduğunu söyleyebiliriz. Bu bilgi, ilerleyen analizleri anlamlandırmamızı kolaylaştıracak.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 3
    st.subheader(
        ":blue[3) Kullanıcılar tarafından oyunda geçirilen ortalama günlük saat]"
    )

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
            Yukarıdaki grafikte, oyuncuların genel olarak günlük 1 saat civarında zaman geçirdiklerini görüyoruz. Ancak, bu grafiği hafta sonu ve hafta içi olarak ayrı ayrı incelediğimizde, hafta içi oyuncuların daha uzun vakit geçirdiği sonucuna ulaşıyoruz. Bunun sebeplerini anlayabilmek için oyuncu profilini iyi anlamak gerekiyor. Ortalama bir oyuncunun 47 yaşında ve çalışma hayatında olan bir birey olduğunu varsayarsak, muhtemelen bu kişiler işe giderken veya dönerken metroda ya da iş yerlerinde molalarda boş vakitlerini Row Match ile değerlendiriyor olabilirler. Oyuncuların hafta sonları harcadıkları zamanı artırmak için hafta sonlarına özel bazı etkinlikler tasarlanabilir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 4
    st.subheader(":blue[4) Yaş gruplarına göre oyunda geçirilen toplam zaman]")
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
        yaxis=dict(showgrid=False, range=[7450, 7650]),
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
            Oyuncuları yaşlarına göre 16-28, 28-41, 41-53, 53-66 ve 66-78 şeklinde yaş gruplarına ayırdığımızda, oyunda en çok vakit geçiren grubun 41-53 yaş aralığındaki oyuncular olduğunu görmekteyiz.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 5
    st.subheader(":blue[5) Seviyelere göre harcanan ortalama zaman]")
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
            Level'ları gruplayıp kullanıcı başına harcanan ortalama zamanları hesapladığımızda yukarıdaki grafiğe ulaşıyoruz. Grafikte, level 200 ile 700 arasında oyuncuların hemen hemen benzer süreler geçirdiğini, ancak level 700 sonrasında bu sürenin zamanla düştüğünü görmekteyiz. Bu durum, yüksek level'ların zorluk derecesinden kaynaklanıyor olabilir. Bir diğer olasılık ise, oyuncuların oyundan sıkılmış olması olabilir. Elimizdeki veritabanına baktığımızda, level 950'den sonra çok az sayıda oyuncuya ait veri bulunmakta, bu yüzden bu noktada net bir şey söylemek zor. Ancak, yine de level 700 ve sonrası için oyuncuların o level’e kadar hiç karşılaşmadığı ve tekrarlanmayan daha farklı görevler tasarlanabilir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 6
    st.subheader(":blue[6) Seviyelere göre win, fail ve quit oranları]")
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
            Yukarıdaki grafikte, yeşil barlar win oranını, mavi barlar fail oranını ve turuncu barlar ise quit oranını temsil etmektedir. Yeşil ve mavi barlar için y ekseni solda yer alırken, turuncu bar için y ekseni grafiğin sağında yer almaktadır. Grafikte, level 150’ye kadar win oranının eşit veya daha yüksek olduğunu, level 150’den sonra ise oyunun giderek zorlaştığını ve fail oranının win oranını geçtiğini görmekteyiz. Benzer şekilde, level 150’den itibaren quit oranının da giderek arttığını görüyoruz. Quit oranının artmış olması, bir önceki grafik için yaptığımız “oyunun zorlaşması” veya “oyuncuların sıkılması” yorumunu destekler niteliktedir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 7
    st.subheader(":blue[7) Seviyelere göre ortalama moves made ve moves left]")
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
            Yukarıdaki grafikte, mavi renkteki çizgi oyuncular tarafından yapılan ortalama move sayısını, turuncu bar ise ortalama moves left sayısını ifade etmektedir. Bir önceki win, fail ve quit oranlarına ilişkin grafiğe paralel olarak, level 150’den sonra moves made sayısının moves left sayısından daha fazla olduğunu görmekteyiz. Ancak, level 700’den sonra moves made sayısının azaldığı gözlemleniyor. Bu durum, seviyelere göre harcanan ortalama zaman grafiğinde (5 numaralı grafik) karşılaştığımız duruma benziyor. Yüksek seviyedeki kullanıcılarla ilgili yeterli miktarda veri veritabanında yer almıyor. Dolayısıyla, grafikte gördüğümüz azalmanın sebebi ile ilgili net bir yorum yapmak bizim için zorlaşıyor.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 8
    st.subheader(":blue[8) Yaş gruplarına harcanan ortalama coin miktarı]")
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
        yaxis=dict(
            showgrid=False,
        ),
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
            Yaş gruplarına göre harcanan ortalama coin miktarına baktığımızda, 41-53 yaş aralığının en çok harcamayı yaptığını görmekteyiz. İlerleyen sayfalarda, kullanıcıların satın alıp almamasını tahmin etmek üzere bir model kuracağız. Bu modelde, harcanan ve kazanılan coin miktarının satın alıp almama kararını oldukça etkilediğini göreceğiz.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 9
    st.subheader(":blue[9) Yaş gruplarına göre harcanan ortalama booster miktarı]")
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
        yaxis=dict(
            showgrid=False,
        ),
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
            Hem yukarıdaki grafikte hem de bir önceki coin harcama grafiğinde, en az coin ve booster harcayan grubun 28-41 yaş aralığındaki kullanıcılar olduğunu görmekteyiz.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 10
    st.subheader(":blue[10) Ülkelere göre kullanıcı sayısı]")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df2["country"],
                values=df2["num_users"],
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
            Ülkelere göre kullanıcıları incelediğimizde, kullanıcıların %28 oranıyla en çok Zephyra ülkesinde yer aldığı görülüyor. Zephyra ülkesini %14.3 oranıyla Emberlyn ve %8.71 oranıyla Moonvale ülkesi takip ediyor.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 11
    st.subheader(":blue[11)	Ülkelere göre gelir dağılımı]")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df2["country"],
                values=df2["sum_revenue"],
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
            Ülkelere göre gelir dağılımını incelediğimizde, toplam gelirin %68’inin Zephyra ülkesinden geldiğini görmekteyiz. Zephyra ülkesini sırasıyla %7 ile Amaryllis ve %4.85 ile Gleamwood takip ediyor. Diğer taraftan, kullanıcıların %23’ü Emberlyn ve Moonvale ülkelerinde yer almasalarına rağmen, bu ülkeler toplam gelirin sadece %2.4’ünü oluşturmaktadır. Dolayısıyla, oyun içi etkinliklerde özellikle Zephyra ülkesindeki kullanıcıların takvimleri ve özel günleri dikkate alınmalıdır. Ayrıca, Amaryllis ve Gleamwood ülkelerindeki pazarlama çalışmaları artırılmalıdır.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 12
    st.subheader(
        ":blue[12)	Etkinlik katılımına göre gelir dağılımı ve harcanan ortalama zaman miktarı]"
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
            Yukaridaki ilk grafikte, gelirin %75’lik kısmının etkinliklere katılan kullanıcılar tarafından geldiğini görmekteyiz. Dolayısıyla, bir önceki grafikte bahsettiğimiz Zephyra ülkesinin etkinlik takvimini çeşitlendirmek, gelir artışı sağlayabilir.
            İkinci grafikte ise etkinliğe katılan kullanıcıların, katılmayan kullanıcılara kıyasla oyun içinde iki kat daha uzun süre geçirdiği görülüyor. Daha önceki grafiklerden, kullanıcıların hafta içi oyun içinde daha fazla vakit geçirdiğini ve hafta sonu için etkinlik planlamanın oyun içi geçirilen süreyi artırabileceğini ifade etmiştik. Ikinci grafik bu ifademizi destekler nitelikte. Yani, hafta sonlarına daha fazla sayıda etkinlik eklemek, oyuncuların oyun içinde geçirdiği süreyi artıracaktır.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 13
    st.subheader(":blue[13)	Network ve Installation İlişkisi]")
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
            Yukarıdaki grafikte, en çok indirmenin Buzz sayesinde elde edildiğini görüyoruz. Bununla ilgili bir yorum yapmadan önce, bu kanallar için ne kadar harcama yapıldığını inceleyelim.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 14
    st.subheader(":blue[14)	Network ve Cost İlişkisi]")
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
            Yukarıdaki grafikte de, bir önceki kullanıcı sayısı grafiğiyle aynı sıralamayı görmekteyiz. Oyunu kendiliğinden, organik bir şekilde indiren kullanıcılar için herhangi bir maliyet meydana gelmiyor. Dolayısıyla, şirket açısından en verimli yöntem organik büyüme. Ancak diğer kanallar ne kadar verimli? Bunu öğrenmek için aşağıdaki "Network ve Number of Installation per $" grafiğine bakmamız gerekiyor.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 15
    st.subheader(":blue[15) Network ve Number of Installation per $ İlişkisi]")
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
            Yukarıdaki grafiğe bakarak, şirket açısından en verimli pazarlamanın sırasıyla Buzz, Sid, Woody ve Jessie kanallarına yapılan yatırım olduğunu görmekteyiz. Dolayısıyla, şirketin pazarlama harcamalarını doğru şekilde yönettiğini anlıyoruz. Elbette, yukarıdaki grafikte gizli birinci olarak görülen Organic kanal, fakat herhangi bir maliyet harcaması gerektirmediği için en son sırada görünmekte. Organic kanal ile daha fazla kullanıcı çekmek için, "Arkadaş Davetiyesi" gibi uygulamalar oyun içine eklenebilir ve davetiye yollayan ile bu davetiyeyle oyunu indiren kullanıcılar için çeşitli ödüllendirme yöntemlerine başvurulabilir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 16
    st.subheader(":blue[16)	Ülkelere Göre Installation Sayısı]")
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
            Yukarıdaki grafikte, en çok indirmenin sırasıyla Mercury, Pluton ve Venus ülkelerinde gerçekleştiğini görüyoruz. Bir yorum yapmadan önce, bu ülkelerdeki pazarlama harcamalarına bakalım. Bir sonraki grafikte buna göz atacağız.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 17
    st.subheader(":blue[17)	Ülkelere Göre Marketing Cost]")
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
            Yukarıdaki grafikte, her ülkeye yapılan harcamaları görmekteyiz. Bir önceki grafik olan kullanıcı sayısı grafiğiyle yukarıdaki grafik benzer sıralamaya sahip. Her iki grafikte de Mercury, Venus, Pluton, Saturn ve Uranus sıralaması aynı. Ancak, hangi ülkeye yapılan harcamanın en çok installation sağladığını belirlemek için, her ülkenin installation sayısını o ülke için yapılan toplam harcamaya bölerek bu sorunun cevabını bulacağız. Bir sonraki grafikte bu sorunun yanıtını vereceğiz.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 18
    st.subheader(":blue[18) Ülkelere Göre Birim Maliyet Başına İndirme Sayısı]")
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
            Yukarıdaki grafik bizim için oldukça önemli, zira harcanan 1$ karşılığında en çok kullanıcıyı Saturn ve Uranus ülkelerinden elde ediyoruz. Diğer taraftan, en çok pazarlama harcaması yapılan Mercury ve Venus ülkelerinde bu rakam oldukça düşük. Dolayısıyla, biraz daha derine inip her ülkedeki kullanıcı başına elde edilen gelire bakabiliriz.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 19
    st.subheader(":blue[19) Ülkelere Göre Kullanıcı Başına Elde Edilen Gelir]")
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
            Yukarıdaki grafiğe baktığımızda, kullanıcı başına en çok gelirin Venus ve Mercury ülkelerinden, en az gelirin ise Uranus ve Saturn ülkelerinden geldiğini görüyoruz. Böylece, Saturn ve Uranus ülkelerindeki bir kullanıcıyı kazanmak için gereken pazarlama maliyeti her ne kadar az olsa da, bu maliyetin düşük görünmesinde dolar kuru etkili olabilir. Ancak, bu ülkelerin kullanıcılarından elde edilen ortalama gelir diğer ülkelere kıyasla daha düşük. Bu sebeple, Venus ve Mercury ülkelerine yapılan toplam harcama daha yüksek çünkü en çok geliri yine bu iki ülkeden elde ediyoruz.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 20
    st.subheader(":blue[20)	Platformlara Göre Gelir, Maliyet ve Maliyet Başına Gelir]")
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
            Row Match için yapılan pazarlama harcamaları elde edilen gelirden daha fazla. Yukarıdaki grafikte, Android ve iOS kullanıcıları üzerinden bu durum görselleştirilmiş. Grafikte, iOS kullanıcılarından birim maliyet başına elde edilen gelirin Android kullanıcılarına kıyasla daha yüksek olduğu anlaşılmaktadır. iOS kullanıcılarından daha çok gelir elde edilmesinin olası sebebi, iOS kullanıcılarının daha yüksek ekonomik refah düzeyine sahip olmalarından kaynaklanıyor olabilir. Her halükarda, pazarlama tarafında iOS kullanıcılarının hedeflenmesi şirketin gelirlerini artırabilir.
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
            Oyunda en çok zaman geçiren ve oyunu en çok indiren kullanıcıların iOS kullanıcıları olduğu görülmekte. Row Match'te in-app reklamların yer aldığını varsayarsak, yukarıdaki tablo bizim için olumsuz bir durumu yansıtıyor olabilir. Çünkü in-app reklamlarla gelir sağlanıyorsa, mümkün olduğunca çok sayıda kullanıcıya ulaşmak en doğru strateji olacaktır. Ancak, App Store için kullanıcı edinim maliyeti (user acquisition cost) Play Store'a kıyasla daha yüksek ve dünya genelinde Android kullanıcı sayısı iOS kullanıcı sayısından daha fazla. Dolayısıyla, in-app reklamlarla gelir sağlandığı varsayımında, kullanıcıların çoğunun Android kullanıcısı olması daha karlı bir pozisyon sağlar.
            <p></p>
            Fakat, Row Match'te abonelik (subscription) veya in-app satın alımla gelir elde edildiğini varsayarsak, yukarıdaki tablo bizim için olumlu bir durumu yansıtıyor olabilir. 2014 yılında yapılan bir Comcast araştırmasına göre, iOS kullanıcılarının medyan yıllık geliri $85,000 iken, Android kullanıcılarının yıllık geliri $61,000 seviyesinde. Yani, iOS kullanıcıları, Android kullanıcılarına kıyasla %40 daha fazla kazanmakta. Bu sosyoekonomik farkın etkilerini, iOS ve Android kullanıcılarının abonelik davranışlarını incelediğimizde de görebiliyoruz. Örneğin, 2021 yılında iOS kullanıcıları abonelik için $13.5 milyar harcamışken, Android kullanıcıları sadece $4.8 milyar harcamış.
            <p></p>
            Gelecek projeksiyonlarına göre, iOS kullanıcılarının App Store'da giderek daha fazla harcama yapması beklenmekte:
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
            Dolayısıyla, pazarlama departmanı App Store ile özel olarak ilgilenmeli ve iOS kullanıcı sayısını daha da artırmak için stratejiler geliştirmelidir.
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
            Row Match için ortalama ROAS 0.26 olsa da, ROAS değerlerini günlük bazda incelediğimizde giderek artan bir trend gözlemliyoruz. Mayıs ayının başında ROAS değerleri %6 civarındayken, ayın sonlarına doğru %42'lere ulaşmış durumda; bu, oldukça umut verici bir gelişme.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 23
    st.subheader(":blue[23) Günlük İndirme Miktarları]")
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
            Günlük indirme miktarlarını incelediğimizde, Mayıs ayının son haftasına kadar inişli çıkışlı bir durum söz konusu olsa da, son hafta indirme miktarlarında gözle görülür bir artış dikkat çekiyor. Ortalamaya baktığımızda ise, günlük yaklaşık 7000 indirmenin gerçekleştiği anlaşılmakta.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graph 24
    st.subheader(":blue[24) DAU ve Günlük Session Sayısı]")
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
        title="Daily Sessions ve DAU",
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
            Yukarıdaki grafikle bir önceki grafiği kıyasladığımızda, DAU’nun günlük indirme miktarından daha az olduğunu görüyoruz. Bu durumun birkaç sebebi olabilir:

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("- Yapılan reklamlar, gameplay’i doğru yansıtmıyor olabilir.")
    st.markdown(
        "- Oyunun ilk seviyeleri çok kolay veya çok zor olabilir, ya da tutoriallar yetersiz olabilir."
    )
    st.markdown(
        "- Oyuncuları oyundan soğutacak kadar fazla sayıda oyun içi reklam çıkıyor olabilir."
    )

    st.markdown(
        """
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:400px;
        padding-top:20px;
    }
    </style>
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
            Ayrıca, DAU ve günlük session sayısı Mayıs ayı için artan bir trende sahip olsa da, Haziran ayında hem DAU hem de günlük session sayısının azaldığını görüyoruz. DAU ve günlük session sayısının artış ve azalış hızları hakkında bir yorum yapabilmek için SessionDAU grafiğine bakacağız.

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
            Yukarıdaki grafiğe baktığımızda, azalan bir trend olduğunu söyleyebiliriz. Dolayısıyla, bu bilgiyi bir önceki grafikle birleştirirsek, Mayıs ayı boyunca DAU artış hızının session sayısı artış hızından daha yüksek olduğunu, ancak Haziran ayında DAU azalış hızının session sayısı azalış hızından daha düşük olduğunu yorumlayabiliriz.
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
            Elimizdeki veri seti için toplam gelir 413520 ve toplam oyuncu sayısı 214888 olduğu için ARPU değerini 1.92 olarak hesaplıyoruz.
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
            Elimizdeki veri seti için toplam gelir 413520 ve gelir sağlayan toplam oyuncu sayısı ise 8130 olduğu için ARPPU değerini 50.86 olarak hesaplıyoruz. ARPPU ve ARPU arasında çok büyük bir fark olması istenilen bir durum değildir. Burada, satın alma gerçekleştirmeyen oyuncuları iyi analiz edip onları satın almaya ikna edecek stratejiler geliştirmek, diğer taraftan satın alma gerçekleştiren oyuncuların bunu tekrar tekrar yapabilmesini sağlamak için ekstra dikkat ve özen göstermek gerekiyor.
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
            Yukarıda 45 günlük ARPDAU değerlerini görüyorsunuz. Grafikte oldukça dalgalanma mevcut. Grafikle ilgili ilginç olabilecek bir nokta ise, 2 Mayıs hariç tüm Pazar günlerinin kendisini takip eden Pazartesi gününden daha yüksek ARPDAU değerine sahip olması. Ayrıca, Cuma-Cumartesi-Pazar üçlüsü, ilk hafta hariç tüm haftalarda genel olarak haftanın geri kalanından daha yüksek ARPDAU değerlerine sahip.
        <p></p>
            Hatırlarsanız, daha önce oyuncuların oyunda hafta sonları hafta içlerine kıyasla daha az zaman geçirdiğini gözlemlemiştik. Yukarıdaki grafikle birlikte, hafta sonları daha az zaman geçirmelerine rağmen daha çok harcama yaptıklarını görüyoruz. Bunun sebebi, hafta sonları ve hafta içleri farklı oyuncu segmentlerinin aktif olmasından kaynaklanıyor olabilir. Örneğin, hafta sonları büyük harcamalar yapan whale oyuncular daha aktif oluyorsa, böyle bir tabloyla karşılaşabiliriz. Bir diğer seçenek ise, hafta sonları oyuncuları satın almaya teşvik edecek özel içerik ve etkinliklerin daha fazla gerçekleşiyor olmasıdır. Oyuncular bu etkinlikler sırasında veya bu etkinliklere katılmak için daha fazla harcama yapıyorsa, yine bu tabloyla karşı karşıya kalabiliriz.
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
            Yukarıdaki grafikte hem günlük toplam Playtime'ı hem de PlaytimeDAU çizgilerini görüyorsunuz. Playtime Mayıs ayı boyunca artış gösterse de Haziran ayında azaldığı görülmekte. Diğer taraftan, PlaytimeDAU çizgisinde azalan bir trend dikkat çekiyor. Daha önce DAU ile ilgili grafiğe baktığımızda da, yukarıdaki Playtime grafiğine benzer şekilde Mayıs ayı boyunca arttığını ancak Haziran ayında azaldığını tespit etmiştik. Dolayısıyla, PlaytimeDAU grafiğindeki azalan bir trend, Mayıs ayı boyunca DAU’daki artışın Playtime’daki artıştan daha yüksek olduğunu; Haziran ayında ise DAU’daki azalmanın, Playtime’daki azalmadan daha düşük oranda gerçekleştiğini gösteriyor.
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
            ARPInstall grafiğindeki pozitif trend, monetizasyon stratejilerinin ve pazarlama kampanyalarının başarılı olduğuna işaret ediyor. Zira yukarıdaki grafik, yeni kullanıcılar arasında daha fazla sayıda whale oyuncu olduğunu ve yeni oyuncuların daha yüksek miktarlarda harcama yaptığını gösteriyor.
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
            CPI grafiğine baktığımızda, ortalama CPI değerinin 4.48 olduğunu görüyoruz. Diğer taraftan, bir önceki grafik olan ARPInstall grafiğinde ortalama ARPInstall değerinin 1.08 olduğunu görmekteyiz. Dolayısıyla, buradan bir kez daha pazarlama harcamalarının gelirden daha yüksek olduğu sonucuna ulaşabiliriz. Grafikle ilgili yapabileceğimiz bir diğer yorum ise, CPI değerinin Mayıs ayının ilk üç haftasında hemen hemen sabitken, Mayıs ayının son haftasında ani bir düşüş göstermesidir. Bunun sebebi, Mayıs ayının son haftasında indirme sayısındaki artış olabilir. Hatırlarsanız, indirme sayısı ile ilgili grafikte (24 numaralı grafik) Mayıs ayının son haftasında indirme sayısında ani bir yükseliş gözlemlemiştik. Dolayısıyla, yukarıdaki grafik daha önceki gözlemlerimizle örtüşmektedir.
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
            Stickiness metriği, retention ve engagement ile ilgili olan önemli bir metriktir. Bu sebeple, stickiness’in olabildiğince yüksek olması arzulanan bir durumdur. Yukarıdaki stickiness grafiğine baktığımızda, pozitif bir trend olduğunu ve stickiness’in giderek arttığını görüyoruz.
        <p></p>
            Ancak, ortalama günlük stickiness’i hesapladığımızda %3 gibi bir değer elde ediyoruz, ki bu oldukça düşük bir değerdir. Stickiness’i artırmak için yapılabilecek birinci seçenek, oyundaki kişiselleştirmeyi artırmaktır. Yıllar önce insanların deliler gibi MMORPG oynamalarının nedeni, MMORPG oyunlarındaki kişiselleştirmenin maksimum seviyelerde olmasıdır. Bir oyunu doğru bir şekilde kişiselleştirdiğiniz sürece, oyuncu oynadığı oyunla daha derin bir bağ kuracak ve oyuna olan bağlılığı da artacaktır. Bu nedenle, stickiness’i artırmak için yapılabilecek ilk seçenek, oyundaki kişiselleştirmeyi artırmaktır.
        <p></p>
            İkinci bir seçenek ise oyuna düzenli ve makul sayıda güncellemeler getirmek ve yeni özellikler eklemektir. Örneğin, tespit edilen bug’ları düzenli olarak güncellemelerle gidermek ve oyunculardan gelen geri bildirimleri takip etmek. Böylece geliştirme süreci oyuncuların beklentileriyle paralel bir şekilde gerçekleşecek ve oyuncuların oyuna olan bağlılığı artacaktır.
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
            Bu grafiklerde Pareto İlkesini gözlemleyebiliriz. Örneğin, A segmentine ait oyuncular toplam oyuncu havuzunun sadece %20’sini oluştururken, elde edilen gelirlerin %80’i A segmentinden geliyor. Ayrıca, A segmenti kullanıcıları B segmentinden 2 kat, C segmentinden 3 kat, D segmentinden ise yaklaşık 5 kat daha yüksek bir AOV (Average Order Value) değerine sahip.
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
            PLTV ile yaptığımız segmentasyonun daha detaylısını RFM Analizi ile gerçekleştirebiliriz. RFM Analizi ile her kullanıcıya birer recency, frequency ve monetary skoru atayacağız. Ardından aşağıdaki tabloya göre recency ve frequency skorundan yola çıkarak segmentasyon işlemini gerçekleştireceğiz:
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
            Monetary skorunu segmentasyon sırasında kullanmıyor olmamızın nedeni, recency ve frequency skorlarının bizim için daha önemli olması ve aynı zamanda yüksek recency ve frequency skorlarına sahip bir oyuncunun genellikle yüksek bir monetary skoruna sahip olmasıdır. Bu yüzden yalnızca recency ve frequency skorunu kullanıyoruz.
        <p></p>
            Elimizdeki RF skorlarına göre segmente ettiğimizde aşağıdaki tree map'i elde ediyoruz:
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
        color_continuous_scale="Viridis",  # Renk paleti
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
            Buradaki segmentler, özellikle pazarlama departmanı için oldukça değerli. Örneğin, can't_loose segmentine ait kullanıcıların ID'leri pazarlama departmanına verilebilir ve bu kullanıcıları yeniden oyuna çekmek için çalışmalar yapılabilir. loyal_customers ve potential_loyalist segmentindeki kullanıcılara ise çeşitli ödüllendirme stratejileri geliştirilebilir.
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
            Bu bölümde, bize verilen kontrol (A) ve deney (B) gruplarını, parasal ve etkileşim metrikleri açısından inceleyeceğiz. Şirket tarafından minimum algılanabilir etki (minimum detectable effect), alfa gibi değerler verilmediği için bu değerleri kendimiz varsayarak ilerleyeceğiz. Aynı zamanda, yaptığımız her A/B testi öncesinde hem kontrol hem de deney grupları için A/A testi gerçekleştireceğiz. Böylece, test istatistiği gereğinden fazla hassas ise A/A testi başarısız olacak.
        <p></p>
            Kullanacağımız yöntemler ise Z testi, t testi ve Mann-Whitney U testi olacak. A/A testlerini Mann-Whitney U testi ile yapacağız. A/B testini ise gerektiğinde Z testi, gerektiğinde t testi ile gerçekleştireceğiz.
        <p></p>
            Tüm bu sürecin daha kolay olması açısından, ab_result() isimli yazmış olduğum bir fonksiyonu kullanacağız. Bu fonksiyon, input olarak A ve B gruplarını alacak ve bu gruplar üzerinde sırasıyla A/A testi ve A/B testi gerçekleştirecek. Eğer A/A testinde bir başarısızlık gerçekleşirse, A/B testine devam edilmeyecek. Eğer A/A testi başarılı bir şekilde gerçekleşirse, A/B testi ile devam edilecek ve daha kolay anlayabilmemiz için bir görsel plot edilecek.
        <p></p>
            Ben biraz sonra yapacağımız analizlerde, A/A testinde bir başarısızlık gerçekleşmediği sürece A/A testinden bahsetmeyeceğim. Ancak yapmış olduğumuz her A/B testi öncesinde A/A testinin de gerçekleştirildiğini aklınızda bulundurun.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Test 1
    st.subheader(":blue[1) Toplam Harcanan Zaman]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (87.286, 89.342) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının oyunda geçirdiği ortalama toplam zaman 99.964 iken, B grubuna ait bir kullanıcının geçirdiği ortalama toplam zaman 76.356 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun oyunda daha az zaman geçirdiğini görüyoruz.**"
    )

    # Test 2
    st.subheader(":blue[2) Session Sayısı]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (3.026, 3.097) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının ortalama session sayısı 3.456 iken, B grubuna ait bir kullanıcının ortalama session sayısının 2.656 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun session sayısının daha az olduğunu görüyoruz.**"
    )

    # Test 3
    st.subheader(":blue[3) Session başına geçirilen ortalama zaman]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (26.91, 26.99) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının session başına geçirdiği ortalama zaman 26.98 iken, B grubuna ait bir kullanıcının session başına geçirdiği ortalama zaman 26.93 olarak bulunuyor. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) ve kontrol (A) grubu arasında session başına geçirilen ortalama zaman açısından istatistiksel olarak anlamlı bir fark olmadığını görüyoruz.**"
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
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (217, 222) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının oyundaki seviyesinin ortalama 283 iken, B grubuna ait bir kullanıcının oyundaki seviyesinin 154 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun oyundaki seviyesinin daha düşük olduğunu görüyoruz.**"
    )

    # Test 5
    st.subheader(":blue[5) Kullanıcı başına toplam gelir (revenue)]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (151, 177) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının getirdiği toplam gelir ortalama olarak 158 iken, B grubuna ait bir kullanıcının getirdiği toplam gelirin ortalaması 169 olarak bulunuyor. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) ve kontrol (A) grubu arasında kişi başına toplam gelir açısından istatistiksel olarak anlamlı bir fark olmadığını görüyoruz.**"
    )

    # Test 6
    st.subheader(":blue[6) Transaction Sayısı]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (11.78, 13.02) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının gerçekleştirdiği toplam transaction sayısının ortalama 11.47, B grubuna ait bir kullanıcının gerçekleştirdiği toplam transaction sayısının ortalamasının ise 13.20 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun transaction sayısının daha yüksek olduğunu görüyoruz.**"
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
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (9.16, 9.67) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının ortalama AOV değerinin 9.73, B grubuna ait bir kullanıcının ortalama AOV değerinin ise 9.15 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun AOV değerinin daha düşük olduğunu görüyoruz.**"
    )

    # Test 8
    st.subheader(":blue[8) Satın Alma Frekansı (Purchase Frequency)]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A ve B gruplarını birleştirip alfa = 0.05 iken güven aralığını hesapladığımızda (0.001949, 0.002154) olduğunu tespit ediyoruz. Ayrı ayrı incelediğimizde ise, A grubuna ait bir kullanıcının ortalama satın alma frekansının 0.001897, B grubuna ait bir kullanıcının ortalama satın alma frekansının ise 0.002184 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun satın alma frekansının daha yüksek olduğunu görüyoruz.**"
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
            A grubu için ortalama DAU değerinin 1780185, B grubu için ortalama DAU değerinin ise 1332930 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için alfa = 0.05 iken Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun DAU değerinin daha düşük olduğunu görüyoruz.**"
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
            A grubu için ortalama ARPDAU değerinin 0.5612, B grubu için ortalama ARPDAU değerinin ise 0.7696 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için alfa = 0.05 iken Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun ARPDAU değerinin daha yüksek olduğunu görüyoruz.**"
    )

    # Test 11
    st.subheader(":blue[11) Günlük Gelir (Revenue)]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A grubu için günlük ortalama gelir 6106 iken, B grubu için günlük ortalama gelirinin 7679 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için alfa = 0.05 iken Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz: 
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
        "**Test sonucunda, deney (B) grubunun günlük gelir değerinin daha yüksek olduğunu görüyoruz.**"
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
            A grubu için günlük ARPInstall değeri 4.55 iken, B grubunun günlük ARPInstall değeri 5.99 olarak belirlenmiştir. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için alfa = 0.05 iken t testi yöntemini kullanarak A/B testini gerçekleştireceğiz: 
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p>
            <strong><span style="font-size:20px;">NOT</span></strong>: Z testi yerine t testi kullanmamızın nedeni, elimizde sadece 28 günlük verinin bulunmasıdır. Veri adedi 30’dan az ise, Z testi yerine t testi kullanmanız tavsiye edilir. Bu örnekte, t testi yerine Z testi kullandığımızda “H0 rejected” sonucuna ulaşıyoruz. Ancak biz t testini kullanacağız ve t testini kullandığımızda “H0 NOT rejected” sonucuna varıyoruz. Veri sayısı arttıkça t testi, Z testine yaklaşır; fakat bu örnekte elimizde az sayıda veri mevcut.
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
        "**Test sonucunda, deney (B) ve kontrol (A) grubu arasında ARPInstall açısından istatistiksel olarak anlamlı bir fark olmadığını görüyoruz.**"
    )

    # Test 13
    st.subheader(":blue[13) Transaction Sayısı/DAU]")
    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
            }
        </style>
        <div class="justified-text">
            A grubu için DAU başına transaction sayısı 0.0412 iken, B grubu için DAU başına transaction sayısının 0.0602 olduğunu görüyoruz. Aradaki farkın istatistiksel olarak anlamlı olup olmadığını görmek için alfa = 0.05 iken Z testi yöntemini kullanarak A/B testini gerçekleştiriyoruz:
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
        "**Test sonucunda, deney (B) grubunun DAU başına düşen transaction sayısının daha yüksek olduğunu görüyoruz.**"
    )

    # Sonuc
    st.markdown(
        """
        <p>
            <strong><span style="font-size:20px;">Sonuç olarak</span></strong>:
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <ul style="padding-left:4px; padding-top:1px; list-style-type:none;">
            <li style="margin-bottom: 10px;">
                <span style="color: green; font-weight: bold;">&#10004;</span>
                Eğer Row Match monetizasyon odaklıysa, yani ana hedef geliri artırmak ve kullanıcı başına düşen geliri maksimize etmekse, deney (B) grubu bu konuda daha iyi performans sergiliyor.
            </li>
            <li>
                <span style="color: green; font-weight: bold;">&#10004;</span>
                Eğer Row Match engagement odaklıysa, yani kullanıcı etkileşimini ve bağlılığını artırmak öncelikse, kontrol (A) grubu bu konuda daha iyi sonuçlar veriyor.
            </li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        
            Elbette, bu sonuçların güvenilirliğini de test etmek gerekir. Örneğin, kullanıcılar belki de sadece değişimin kendisine tepki veriyor olabilir; yani "*novelty effect*" dediğimiz durum söz konusu olabilir. Bu sebeple, burada yaptığımız testler daha geniş bir zaman aralığında tekrar edilmeli ve sonuçlarda bir farklılık olup olmadığı kontrol edilmelidir.
        
        """,
        unsafe_allow_html=True,
    )


###############################
# PART III: MODELLEME
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
            Bu bölümde, bir oyuncunun 30 günlük periyodun sonunda satın alma gerçekleştirip gerçekleştirmeyeceğini tahmin eden binary bir sınıflandırma modeli oluşturmayı amaçlayacağız. SQL tablosunda bulunan d30_revenue sütunu binary bir değişken değil; bu yüzden purchased isminde bir binary değişken oluşturuyoruz ve onu hedef değişkenimiz olarak belirliyoruz.
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
            list-style-type: disc;  /* Varsayılan nokta işaretli liste */
            margin-top: -20px;  /* Üst boşluğu azaltmak için */
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
        <li>İlk olarak, hedef değişkenimiz olan purchased değişkenini inceliyoruz ve kullanıcıların sadece %7.83’ünün satın alma gerçekleştirdiğini öğreniyoruz. Dolayısıyla, bir dengesizliğin (imbalanced dataset) söz konusu olduğunu anlıyoruz. Bu durum, ilerleyen kısımlarda modelimizi eğitirken hangi metriklere odaklanmamız gerektiği konusunda bize fikir veriyor. Dengesiz durumlarda ROC AUC ve F1 gibi skorlar bizim için daha anlamlı olacaktır.</li>
        <li>Daha sonra veri setindeki null değerleri kontrol ediyoruz. Neyse ki, veri seti oldukça düzenli ve içerisinde hiç null değer bulunmuyor.</li>
        <li>describe() fonksiyonuyla değişkenlerde herhangi bir anomali olup olmadığını kontrol ediyoruz. Örneğin, age sütununa herhangi bir yerde 0 yazılmışsa, bunu describe fonksiyonu sayesinde öğrenebiliriz. Neyse ki, bu aşamada da anormal bir durum gözükmüyor.</li>
        <li>Numeric column’lar arasındaki korelasyonu görmek için bir heatmap oluşturuyoruz. Burada, time_spend, coin_spend, coin_earn, level_success, level_fail, level_start, booster_spend, ve booster_earn sütunları arasında oldukça yüksek korelasyonlar keşfediyoruz. Bu bilgiyi daha sonra özellik türetirken kullanabiliriz:</li>
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
        <li>Outlier kontrolü yapıyoruz ve time_spend, coin_spend, coin_earn, level_success, level_fail, level_start, booster_spend, booster_earn, coin_amount, event_participate, ve shop_open sütunlarında outlier’lar olduğunu görüyoruz. Outlier’lar, biraz sonra oluşturacağımız modelin performansını olumsuz yönde etkileyebilir, bu yüzden outlier’ları IQR (Interquartile Range) kullanarak %1 ve %99’luk eşiklerle yer değiştiriyoruz.</li>
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
        <li>Kullanıcıların yaşlarından yola çıkarak young, early_adult, mid_adult, late_adult ve old şeklinde kategorik değerlerden oluşan bir kategorik sütun oluşturuyoruz.</li>
        <li>Daha sonra, one-hot encoding kullanarak bütün kategorik değişkenleri sayısal değişkenlere dönüştürüyoruz.</li>
        <li>Yüksek korelasyon olduğunu keşfettiğimiz değişkenler üzerinden son derece deneysel bir şekilde yeni değişkenler türetmeye çalışıyoruz. Türettiğimiz değişkenler arasında modelin ROC AUC skorunu geliştiren bir değişken varsa, feature_creator() fonksiyonu bunu bize bildiriyor. feature_creator() fonksiyonu sayesinde time_spend/age ve coin_spend/coin_amount şeklinde oluşturabileceğimiz yeni değişkenlerin modelin ROC AUC skorunu geliştirebileceğini öğreniyoruz. Bu yeni değişkenleri x dataframe’ine ekliyoruz.</li>
        <li>Son olarak, model seçim bölümünde kullanmak üzere dataframe’i %70 eğitim ve %30 test şeklinde ikiye ayırıyoruz.</li>
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
            Bu aşamada çeşitli modeller deniyoruz. Denediğimiz modelleri ROC AUC ve F1 skoru kriterlerine göre sıraladığımızda, aşağıdaki gibi bir sonuç elde ediyoruz:
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
            <strong><span style="font-size:20px;">NOT</span></strong>: ROC AUC ve F1 metriklerine bakmamızın sebebi, veri setinin genel olarak dengesiz (imbalanced) olmasıdır. Imbalanced veri setleriyle karşılaştığımızda, veri setini dengeli hale getirmek için undersampling ve oversampling gibi yöntemler kullanabiliriz. Ancak, geçmişte çalıştığım fraud detection modellerinde bu tekniklerden pek fayda elde edemedim. Genel populasyonda satın alma oranı %7.83 ise, modeli de aynı orana sahip bir veri kümesiyle eğitmek, validasyon performansı üzerinde daha faydalı oluyor. Bu sebeple, herhangi bir sampling işlemi yapmadan ROC AUC ve F1 metriklerini kullanarak devam edeceğim. 
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
        Sonuç olarak, CatBoost modelinde karar kılıyoruz.
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
        Hiperparametre optimizasyonu için Optuna isimli açık kaynak kodlu bir kütüphaneyi kullanacağım. Optuna, sahip olduğu ağaç benzeri algoritma sayesinde hiperparametreler arasında daha verimli ve akıllı bir şekilde arama yaptığı için GridSearch veya RandomSearch gibi yöntemlere kıyasla zaman ve kaynak açısından oldukça tasarruf sağlıyor.
        <p></p>
        Veri setini bu kez train_test ve %10’luk validation seti şeklinde ikiye ayırıyorum. Modeli train_test veri setiyle 3’lü k-fold cross validation kullanarak eğiteceğim; diğer %10’luk validation setini ise eğitim sürecinde modele hiç göstermeyeceğim.
        <p></p>
        objective() ve logging_callback() isimli iki fonksiyonla Optuna için gerekli ayarlamaları yaptıktan sonra, ROC AUC skorunu artıracak şekilde modeli eğitmesi için Optuna’ya talimat veriyorum. Böylece Optuna, ROC AUC skorunu yaklaşık %1 oranında artırmayı başarıyor ve bulduğu en iyi parametreleri aşağıdaki gibi aktarıyor:
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
        Hiperparameter tuning işleminden sonra artık modeli hiç görmediği validation seti üzerinde test edebiliriz:
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
        Böylece, validation kümesi üzerinde %87 oranında bir skor elde etmeyi başarıyoruz.
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
        Bütün bu sürecin bir hokus pokus şeklinde gerçekleşmediğini diğer iş birimlerine anlatabilmek için, modelin nasıl çalıştığını kendimiz anlayabilmeliyiz. Bu noktada, SHAP (SHapley Additive exPlanations) isimli bir kütüphaneyi kullanabiliriz. Bu kütüphane, çıktıya hangi özelliklerin nasıl etki ettiğini görselleştirmeye yarayan oldukça kullanışlı bir açık kaynaklı Python paketidir. SHAP’in BeeSwarm grafiği ile hangi özelliklerin en önemli olduğunu ve outputla aralarındaki ilişkileri görebiliriz:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/beeswarm.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        Örneğin, yukarıdaki BeeSwarm grafiğine bakarak en önemli ilk üç değişkenin coin_earn, country_Zephyra ve kendi türettiğimiz coin_spend/coin_amount değişkeni olduğunu görebiliyoruz. Bu değişkenlerin outputa etkisini daha detaylı görmek için scatter fonksiyonunu kullanabiliriz. Mesela, coin_earn değişkeni ile satın alma arasında aşağıdaki grafikte görüldüğü üzere doğru orantılı bir ilişki mevcut:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/coin_earn.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        Bir başka örnek olarak, level_success değişkenini inceleyelim. level_success değişkeni ile satın alma arasında negatif bir ilişki var. Yani level_success arttıkça kullanıcıların satın alma ihtiyacı azalıyor:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/level_success.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        Son olarak, bir de iOS kullanıp kullanmamanın satın almayı nasıl etkilediğine bakalım. Bununla ilgili platform_ios isminde bir binary değişkenimiz vardı, onu inceleyeceğiz:
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_part3, right_part3 = st.columns([0.2, 0.4])
    right_part3.image("images/part_iii/platform_ios.png", width=500)

    st.markdown(
        """
        <div class="justified-text">
        Gördüğünüz gibi ve tahmin ettiğimiz gibi, iOS kullanmak satın almayı artıran faktörlerden biriymiş.
        </div>
        """,
        unsafe_allow_html=True,
    )


###############################
# PART IV: TAHMIN
###############################


scaler = get_scaler()
model = get_model()

with part4:
    st.markdown(
        """
        <div class="justified-text">
        Bu bölümde aşağıda belirtilen yerlere uygun değerleri girerek modelin tahmin sonucunu görüntüleyebilirsiniz.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_part4, right_part4 = st.columns(2)
    age = left_part4.number_input(
        "age değişkenini giriniz:", min_value=5, max_value=90, step=1, value=17
    )
    time_spend = left_part4.number_input(
        "time_spend değişkenini giriniz:",
        min_value=0,
        max_value=120000,
        step=1000,
        value=38890,
    )
    coin_spend = left_part4.number_input(
        "coin_spend değişkenini giriniz:",
        min_value=0,
        max_value=350000,
        step=5000,
        value=117500,
    )
    coin_earn = left_part4.number_input(
        "coin_earn değişkenini giriniz:",
        min_value=0,
        max_value=375000,
        step=5000,
        value=125640,
    )
    level_success = left_part4.number_input(
        "level_success değişkenini giriniz:",
        min_value=0,
        max_value=1000,
        step=1,
        value=255,
    )
    level_fail = left_part4.number_input(
        "level_fail değişkenini giriniz:", min_value=0, max_value=1000, step=1, value=0
    )
    level_start = left_part4.number_input(
        "level_start değişkenini giriniz:",
        min_value=0,
        max_value=1000,
        step=1,
        value=278,
    )
    booster_spend = left_part4.number_input(
        "booster_spend değişkenini giriniz:",
        min_value=0,
        max_value=500,
        step=50,
        value=110,
    )
    booster_earn = right_part4.number_input(
        "booster_earn değişkenini giriniz:",
        min_value=0,
        max_value=500,
        step=50,
        value=205,
    )
    coin_amount = right_part4.number_input(
        "coin_amount değişkenini giriniz:",
        min_value=0,
        max_value=37500,
        step=1750,
        value=12262,
    )
    shop_open = right_part4.number_input(
        "shop_open değişkenini giriniz:", min_value=0, max_value=20, step=1, value=1
    )
    event_participate = right_part4.selectbox(
        "event_participate değişkenini seçiniz:", ["Evet", "Hayir"]
    )
    if event_participate == "Evet":
        event_participate = 1
    else:
        event_participate = 0
    platform = right_part4.selectbox(
        "platform değişkenini seçiniz:", ["ios", "android"]
    )
    network = right_part4.selectbox(
        "network değişkenini seçiniz:",
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
        "country değişkenini seçiniz:",
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

    if st.button("Tahmin et!"):
        prediction = model.predict(model_input)
        if prediction == 1:
            st.success(f"Kullanıcı satın alacak! :)")
        else:
            st.success(f"Kullanıcı satın almayacak! :)")
        st.balloons()
