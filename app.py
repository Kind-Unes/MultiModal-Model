import streamlit as st

# Landing Page
def landing_page():
    st.title("AI Functionality Dashboard")
    
    # Define a 2x4 grid for tasks
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.image("assets/images/2659360.png", caption="Text Generation"):
            st.session_state.page = "Text Generation"

    with col2:
        if st.image("assets/images/4577216.png", caption="Image Generation"):
            st.session_state.page = "Image Generation"

    with col3:
        if st.image("assets/images/picture.png", caption="Image Segmentation 3"):
            st.session_state.page = "Task 3"

    with col4:
        if st.image("assets/images/zoom.png", caption="Object detection"):
            st.session_state.page = "Task 4"

# Main Page for Text Generation
def text_generation_page():
    st.title("Text Generation Task")
    st.write("This is the main page for Text Generation.")
    # Add your specific text generation functionality here

# Main Page for Image Generation
def image_generation_page():
    st.title("Image Generation Task")
    st.write("This is the main page for Image Generation.")
    # Add your specific image generation functionality here

# Main Page for Task 3
def task3_page():
    st.title("Task 3")
    st.write("This is the main page for Task 3.")
    # Add your functionality for Task 3 here

# Main Page for Task 4
def task4_page():
    st.title("Task 4")
    st.write("This is the main page for Task 4.")
    # Add your functionality for Task 4 here

# App Routing
if 'page' not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "Text Generation":
    text_generation_page()
elif st.session_state.page == "Image Generation":
    image_generation_page()
elif st.session_state.page == "Task 3":
    task3_page()
elif st.session_state.page == "Task 4":
    task4_page()
