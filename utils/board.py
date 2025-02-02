import os
import json
import argparse
import streamlit as st

def load_logs(base_dir):
    logs = dict()
    for root, _, files in sorted(os.walk(base_dir)):
        if files:
            files = [file for file in files if file.endswith('.json')]
            for file in sorted(files, key=lambda file: int(file.replace('.json', '').split('ID')[-1])):
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    logs[file_path] = json.load(f)
        
    return logs

def display_log(log_title, log):
    st.title(log_title)
    body = log["prompt_at_repair"]
    processed_body = body[body.find('## Description'):]  # Trim opening remarks on Scientific Debugging
    processed_body = processed_body.replace('\n', '  \n')  # Update newline to follow markdown spec
    st.write(processed_body)
    st.subheader(f'Passed: `{log["passed"]}`')

def on_log_change(logs, selected_log):
    st.session_state["explanation_rate"] = logs[selected_log].get("rate", 0)

def on_button_change(logs, selected_log, rate):
    if not rate:
        return
    logs[selected_log]["rate"] = rate
    with open(selected_log, 'w') as f:
        json.dump(logs[selected_log], f, indent=4)

def setup(logs):
    st.sidebar.header('Log List')
    
    selected_log = st.sidebar.selectbox(
        'log_selection',
        logs.keys(),
        label_visibility="collapsed",
        key="selected_log",
        on_change=lambda: on_log_change(logs, st.session_state.selected_log)
    )
    
    display_log(selected_log, logs[selected_log])
    
    st.sidebar.header('Rate explanation')
    st.sidebar.radio(
        'explanation_rate',
        [0, 1, 2, 3, 4, 5],
        index=logs[selected_log].get("rate", 0),
        horizontal=True,
        label_visibility="collapsed",
        key="explanation_rate",
        on_change=lambda: on_button_change(logs, st.session_state.selected_log, st.session_state.explanation_rate)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='asd')
    parser.add_argument('-b', '--base_directory', default='../asd/results')
    args = parser.parse_args()
    
    logs = load_logs(args.base_directory)
    setup(logs)
