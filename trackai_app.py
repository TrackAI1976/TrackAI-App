import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
from datetime import timedelta
import re
from io import BytesIO

st.set_page_config(page_title="TRACK.ai - Schedule Analyzer", layout="wide")

# --- Access Code Protection ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 TRACK.ai Access Required")
    password = st.text_input("Enter Access Code", type="password")
    if password == "track123":
        st.session_state.authenticated = True
    else:
        st.stop()

st.title("📊 TRACK.ai - Upload Your Schedule and Visualize It")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your project Excel file:", type=["xlsx"])

if uploaded_file:
    # --- Load Sheets ---
    project_start_df = pd.read_excel(uploaded_file, sheet_name='Sheet2')
    project_start_date = pd.to_datetime(project_start_df.columns[1])

    df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    df.columns = df.columns.str.strip()
    df['Duration'] = df['Duration'].astype(int)
    df['Predecessor Details'] = df['Predecessor Details'].fillna("").astype(str)

    # --- Helper Function ---
    def parse_pred_detail(detail):
        # Accept formats like "A1010:FS3", "A1010 SS-2", no d, no + required
        match = re.match(r"^\s*([A-Za-z0-9]+)[:\s]*(FS|SS|FF|SF)([-]?\d+)?\s*$", detail.strip())
        if match:
            pred, rel_type, lag = match.groups()
            lag_days = int(lag) if lag else 0
            return pred, rel_type, lag_days
        return None, None, 0

    # --- Build Graph ---
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row['Activity ID'], duration=row['Duration'], name=row['Activity Name'])
    for _, row in df.iterrows():
        details = row['Predecessor Details']
        for entry in details.split(','):
            pred_id, rel_type, lag = parse_pred_detail(entry)
            if pred_id:
                G.add_edge(pred_id, row['Activity ID'], rel_type=rel_type, lag=lag)

    # --- Forward Pass ---
    early_start, early_finish = {}, {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            es = project_start_date
        else:
            start_candidates = []
            for p in preds:
                edge = G.edges[p, node]
                lag = timedelta(days=edge['lag'])
                pred_es = early_start[p]
                pred_ef = early_finish[p]
                dur = timedelta(days=G.nodes[node]['duration'])
                if edge['rel_type'] == 'FS':
                    start_candidates.append(pred_ef + lag)
                elif edge['rel_type'] == 'SS':
                    start_candidates.append(pred_es + lag)
                elif edge['rel_type'] == 'FF':
                    start_candidates.append(pred_ef + lag - dur)
                elif edge['rel_type'] == 'SF':
                    start_candidates.append(pred_es + lag - dur)
            es = max(start_candidates)
        ef = es + timedelta(days=G.nodes[node]['duration'])
        early_start[node] = es
        early_finish[node] = ef

    # --- Backward Pass ---
    project_finish = max(early_finish.values())
    late_finish, late_start = {}, {}
    for node in G.nodes:
        late_finish[node] = project_finish
        late_start[node] = project_finish - timedelta(days=G.nodes[node]['duration'])

    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        if not succs:
            continue
        ls_candidates = []
        for s in succs:
            edge = G.edges[node, s]
            lag = timedelta(days=edge['lag'])
            succ_ls = late_start[s]
            succ_lf = late_finish[s]
            duration = timedelta(days=G.nodes[node]['duration'])
            if edge['rel_type'] == 'FS':
                ls_candidates.append(succ_ls - lag - duration)
            elif edge['rel_type'] == 'SS':
                ls_candidates.append(succ_ls - lag)
            elif edge['rel_type'] == 'FF':
                ls_candidates.append(succ_lf - lag - duration)
            elif edge['rel_type'] == 'SF':
                ls_candidates.append(succ_lf - lag)
        if ls_candidates:
            late_start[node] = min(ls_candidates)
            late_finish[node] = late_start[node] + timedelta(days=G.nodes[node]['duration'])

    # --- Merge Results ---
    df['ES'] = df['Activity ID'].map(early_start)
    df['EF'] = df['Activity ID'].map(early_finish)
    df['LS'] = df['Activity ID'].map(late_start)
    df['LF'] = df['Activity ID'].map(late_finish)
    df['Total Float'] = (df['LS'] - df['ES']).dt.days
    df['Is_Critical'] = df['Total Float'] == 0

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")
    critical_filter = st.sidebar.checkbox("Show only critical activities")
    activity_names = sorted(df['Activity Name'].unique())
    selected_activities = st.sidebar.multiselect("Select activities to show", activity_names, default=activity_names)

    filtered_df = df[df['Activity Name'].isin(selected_activities)]
    if critical_filter:
        filtered_df = filtered_df[filtered_df['Is_Critical']]

    # --- Interactive Gantt Chart ---
    st.subheader("📈 Gantt Chart")
    gantt_data = filtered_df.rename(columns={'Activity Name': 'Task', 'ES': 'Start', 'EF': 'Finish'})
    fig = px.timeline(gantt_data, x_start="Start", x_end="Finish", y="Task", color="Is_Critical",
                      color_discrete_map={True: 'red', False: 'gray'})
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Activities",
                      legend_title="Critical Path")
    st.plotly_chart(fig, use_container_width=True)

    # --- Download Updated Schedule ---
    st.subheader("👅 Download Updated Schedule")

    # Create the output DataFrame
    output = df[['Activity ID', 'Activity Name', 'Duration', 'ES', 'EF', 'LS', 'LF', 'Total Float', 'Is_Critical']]

    # Create in-memory Excel file
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        output.to_excel(writer, index=False, sheet_name='Schedule Output')
    excel_buffer.seek(0)

    # Streamlit download button
    st.download_button(
        label="Download Excel Output",
        data=excel_buffer,
        file_name="schedule_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_output"
    )
