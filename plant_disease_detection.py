import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"C:\Users\Vishwa\OneDrive\Desktop\plant_disease_detection\trained_plant_disease0007.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","User Guide","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = r"C:\Users\Vishwa\Downloads\bg(2).jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Introducing the Plant Disease Recognition System! 🌿

Our goal is to aid in the prompt identification of plant diseases. Simply upload an image of a plant, and our system will swiftly analyze it to pinpoint any signs of diseases. Let's work together to safeguard our crops and guarantee a bountiful harvest!

    
    """)

#About Project
elif(app_mode=="User Guide"):
    st.header("User Guide")
    st.markdown("""
   ### How It Works
    1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.
    3. Results: View the results and recommendations for further action.

    ### Why Choose Us?
    - Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - User-Friendly: Simple and intuitive interface for seamless user experience.
    - Fast and Efficient: Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the Disease Recognition page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the About page.

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
                    'Blueberry_healthy', 'Cherry(including_sour)__Powdery_mildew', 
                    'Cherry_(including_sour)_healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)_Common_rust', 'Corn(maize)_Northern_Leaf_Blight', 'Corn(maize)__healthy', 
                    'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 
                    'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot',
                    'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 
                    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 
                    'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 
                    'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 
                    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
                    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 
                    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Uploaded image has {}".format(class_name[result_index]))