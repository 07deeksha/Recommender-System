import streamlit
import streamlit as st
from PIL import Image
from predict import main as generate_hashtags
from musicTest import recommend_genre, recommend_songs

def main():
    st.title("TRENDPULSE")
    st.subheader("Content's Best Guide, Trendpulse by Your Side!")

    # Redirect buttons
    if st.button("HASHtheTAGS"):
        show_hashtags_page()

    if st.button("MUSIC-^-PULSE"):
        page = st.sidebar.radio("Navigation", ["Recommend Genre", "Recommend Songs"])

        if page == "Recommend Genre":
            recommend_genre_ui()
        elif page == "Recommend Songs":
            recommend_songs_ui()


def show_hashtags_page():
    st.header("HASHtheTAGS")
    st.write("Upload an image to generate recommended hashtags")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Generate Hashtags"):
            hashtags = generate_hashtags(image, r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\encoder-150-1.ckpt', r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\decoder-150-1.ckpt')
            streamlit.bootstrap.run(generate_hashtags(), '', image, flag_options={})
            st.subheader("Recommended Hashtags")
            st.write(", ".join(hashtags))

def recommend_genre_ui():
    st.header("MUSIC-^-PULSE")
    st.write("Select a genre to get song recommendations")
    genres = [
        "Trending Genres",
        "BollywoodDance",
        "BollywoodDanceRomantic",
        "BollywoodRomantic",
        "Bollywood",
        "BollywoodRomanticSad",
        "BollywoodDevotional",
        "BollywoodDanceSad",
        "BollywoodSad",
        "BollywoodMotivational",
        "BollywoodRomanticSadSensual",
        "BollywoodRomanticSensual",
        "BollywoodDancePatriotic",
        "BollywoodMotivationalPatriotic",
        "BollywoodDanceSensual",
        "BollywoodDevotionalSad",
        "BollywoodDanceMotivationalPatriotic",
        "BollywoodPatrioticSad"
    ]

    st.latex(genres)
    favorite_genre = st.text_input("FOR RECOMMENDATIONS BASED ON GENRE, ENTER YOUR FAVORITE GENRE")
    if st.button("Get Recommendations"):
        recommendations = recommend_genre(favorite_genre)
        st.write("Recommendations:", recommendations)
# def show_music_page():
#     st.header("MUSIC-^-PULSE")
#     st.write("Select a genre to get song recommendations")
#
#     genres = [
#         "BollywoodDance",
#         "BollywoodDanceRomantic",
#         "BollywoodRomantic",
#         "Bollywood",
#         "BollywoodRomanticSad",
#         "BollywoodDevotional",
#         "BollywoodDanceSad",
#         "BollywoodSad",
#         "BollywoodMotivational",
#         "BollywoodRomanticSadSensual",
#         "BollywoodRomanticSensual",
#         "BollywoodDancePatriotic",
#         "BollywoodMotivationalPatriotic",
#         "BollywoodDanceSensual",
#         "BollywoodDevotionalSad",
#         "BollywoodDanceMotivationalPatriotic",
#         "BollywoodPatrioticSad"
#     ]
#
#     selected_genre = st.selectbox("Select Genre", genres)
#     if st.button("Get Recommendations"):
#         recommendations = rec()
#         st.subheader(f"Songs Recommendations for {selected_genre}")
#         for song in recommendations:
#             st.write(song)
def recommend_songs_ui():
    st.title("Recommend Songs")
    st.write("Welcome to TrendPulse - MusicPulse !!!")
    favorite_song = st.text_input("FOR RECOMMENDATIONS BASED ON SONGS, SELECT YOUR FAVORITE SONG")
    if st.button("Get Recommendations"):
        recommendations = recommend_songs(favorite_song)
        st.write("Recommendations:", recommendations)

if __name__ == "__main__":
    main()
