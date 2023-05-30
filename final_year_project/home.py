import streamlit as st
#from final_recommender import recommender, add_item
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Swapsies App",
    page_icon="👋",
)

st.title("Swapsies Home Page")
st.write("Swapsies App: navigate to the recommend tab to find a recommendation for your item")
st.write("Then swap by contacting the email associated with your desired swap item")
st.sidebar.success("Select a page above.")
