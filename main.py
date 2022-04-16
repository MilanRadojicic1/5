import streamlit as st
from multiapp import MultiApp
from apps import home, model ,model2,model3,model4,model5,model6,model7,model8,model9,model10# import your app modules here

app = MultiApp()

st.markdown("""
# Movie Recommender App
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Recommend by Genre", model.app)
app.add_app("Recommend by Director", model2.app)
app.add_app("Recommend by Actor", model3.app)
app.add_app("Recommend by Movie Description and Metadata", model4.app)
app.add_app("Recommend by Movie Description", model5.app)
app.add_app("Recommend by Metadata", model6.app)
app.add_app("(IMPROVED) Recommend by Movie Description and Metadata", model7.app)
app.add_app("User-based collaborative filtering", model8.app)
app.add_app("Item-based collaborative filtering", model9.app)
app.add_app("Twitter based movie recommender", model10.app)

# The main app
app.run()