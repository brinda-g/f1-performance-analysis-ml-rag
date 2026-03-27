import streamlit as st

st.set_page_config(page_title="F1 Test", layout="wide")

st.title("🏎️ F1 Race Insights AI - Simple Test")

page = st.sidebar.radio("Navigation", ["Overview", "Test"])

if page == "Overview":
    st.header("Dashboard Overview")
    st.success("✅ Streamlit is working!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Races", "1")
    with col2:
        st.metric("Laps", "245")
    with col3:
        st.metric("Model", "31.9%")
        
else:
    st.header("Test Page")
    st.write("If you can see this, everything works!")
    
    if st.button("Click Me"):
        st.balloons()
        st.success("Success! Your dashboard is functional!")