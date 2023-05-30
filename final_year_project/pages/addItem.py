import streamlit as st
from pages.final_recommender import add_item
from PIL import Image
import os

st.title("Add an item:")

item_desc = st.text_input("Item Description", 'Enter Description', key=10000001)

st.write("Images can only be JPG type")

uploaded_file = st.file_uploader("", type=['jpg'])

dir_img = "/Users/rainietamakloe/CodingProjects/final_year_project/dataset_files/images/"

addedItem = st.button("Add")

if addedItem:
    if len(item_desc) == 0 :
        st.error("please add an item description")
    elif len(item_desc) < 10:
        st.error("Please add a longer description")
    else:
        img_id = add_item(item_desc)
        #print(f'image id is here {img_id}')
        st.write("Item added!")
        if uploaded_file is not None:
            with open(os.path.join(dir_img, uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())

        # img = Image.open(r'/Users/rainietamakloe/CodingProjects/final_year_project/dataset_files/images/')
        # img = img.save(dir_img + uploaded_file.name)
        # print(img.name)
        #print(img_id)
        os.rename(str(dir_img) + uploaded_file.name, str(dir_img) + str(img_id) + ".jpg" )
        #print(f' new rename image is {uploaded_file.name}')
        #Image.open(str(dir_img) + str(img_id)+ '.jpg')
        
