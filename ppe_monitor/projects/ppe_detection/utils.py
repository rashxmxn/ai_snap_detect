from collections import defaultdict
import streamlit as st
import pandas as pd
import plotly.express as px

def display_violation_metrics():
    st.subheader("Violation Statistics")
    
    # Display total violations
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Violations", st.session_state.violation_stats['total_violations'])
    
    # Violations by PPE type
    if st.session_state.violation_stats['violations_by_ppe']:
        violations_df = pd.DataFrame(
            list(st.session_state.violation_stats['violations_by_ppe'].items()),
            columns=['PPE Type', 'Violations']
        )
        
        fig = px.bar(violations_df, x='PPE Type', y='Violations',
                    title='Violations by PPE Type',
                    color='PPE Type')
        st.plotly_chart(fig)
    
    # Display violation screenshots
    if st.session_state.violation_stats['violations_screenshots']:
        st.subheader("Violation Screenshots")
        for idx, violation in enumerate(st.session_state.violation_stats['violations_screenshots']):
            with st.expander(f"Violation {idx + 1} - {violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                st.image(violation['image'])
                st.write(f"Missing PPE: {', '.join(violation['missing_ppe'])}")

def reset_violation_stats():
    from pathlib import Path
    st.session_state.violation_stats = {
        'total_violations': 0,
        'violations_by_ppe': defaultdict(int),
        'violations_screenshots': []
    }
    # Clean up screenshot directory
    for file in Path("violation_screenshots").glob("*.jpg"):
        file.unlink()