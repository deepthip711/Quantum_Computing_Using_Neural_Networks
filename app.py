import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import time
import pennylane as qml
from pennylane import numpy as pnp
import tensorflow as tf
from tensorflow.keras import layers, Model
import io
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.set_page_config(page_title="Quantum Neural Network Simulator", layout="wide")
st.title("🔮 Quantum Neural Network (QNN) Simulator")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, step=50)
noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.2, step=0.05)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 30, step=10)
n_layers = st.sidebar.slider("Quantum Circuit Layers", 1, 5, 2)
model_choice = st.sidebar.radio("Model Type", ["Classical", "Quantum"])

st.markdown("""
### 🧠 What are Quantum Neural Networks?
Quantum Neural Networks (QNNs) are algorithms that merge quantum computing and classical machine learning. 
QNNs use quantum circuits to process data in superposition, potentially offering exponential speed-ups.
""")

# --- Dataset Generation ---
st.subheader("Generated Dataset")
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

fig, ax = plt.subplots()
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
ax.set_title("2D Quantum Dataset")
st.pyplot(fig)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


metrics_records = []
final_accuracies = {}

if model_choice == "Classical":
    st.subheader("Classical Neural Network Integration")
    model = tf.keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(10, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    losses, accuracies = [], []
    progress_bar = st.progress(0)
    loss_chart = st.line_chart()
    accuracy_chart = st.line_chart()
    log_output = st.empty()

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        acc = history.history['accuracy'][0]

        losses.append(loss)
        accuracies.append(acc)
        loss_chart.line_chart(losses)
        accuracy_chart.line_chart(accuracies)
        log_output.code(f"Epoch {epoch + 1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
        progress_bar.progress((epoch + 1) / epochs)
        time.sleep(0.05)

    st.success("Training Complete")
    final_accuracies["Classical"] = accuracies[-1]

    st.subheader("Confusion Matrix (Animated)")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(cm.astype(int), x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues')
    fig_cm.update_layout(title_text="Confusion Matrix Animation", title_x=0.5)
    st.plotly_chart(fig_cm)

else:
    if model_choice == "Quantum":
        st.subheader("Quantum Neural Network Integration")
        model = tf.keras.Sequential([
            layers.Input(shape=(2,)),
            layers.Dense(10, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        losses, accuracies = [], []
        progress_bar = st.progress(0)
        loss_chart = st.line_chart()
        accuracy_chart = st.line_chart()
        log_output = st.empty()

        for epoch in range(epochs):
            history = model.fit(X_train, y_train, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            acc = history.history['accuracy'][0]

            losses.append(loss)
            accuracies.append(acc)
            loss_chart.line_chart(losses)
            accuracy_chart.line_chart(accuracies)
            log_output.code(f"Epoch {epoch + 1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
            progress_bar.progress((epoch + 1) / epochs)
            time.sleep(0.05)

        st.success("Training Complete")
        final_accuracies["Quantum"] = accuracies[-1]

        st.subheader("Confusion Matrix (Animated)")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(cm.astype(int), x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues')
        fig_cm.update_layout(title_text="Confusion Matrix Animation", title_x=0.5)
        st.plotly_chart(fig_cm)
    


# --- Export CSV ---
if metrics_records:
    st.subheader("📊 Export Metrics")
    metrics_df = pd.DataFrame(metrics_records)
    csv = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button("📅 Download Metrics as CSV", data=csv, file_name="training_metrics.csv", mime='text/csv')

# --- Comparison Plot ---
if final_accuracies:
    st.subheader("📈 Model Accuracy Comparison")
    comparison_df = pd.DataFrame(final_accuracies.items(), columns=["Model", "Accuracy"])
    fig_comp, ax_comp = plt.subplots()
    sns.barplot(data=comparison_df, x="Model", y="Accuracy", ax=ax_comp)
    ax_comp.set_title("Final Accuracy by Model Type")
    st.pyplot(fig_comp)

# --- Explanation ---
st.subheader("📘 About Quantum Neural Networks")
with st.expander("Click to learn about QNNs"):
    st.markdown("""
    Quantum Neural Networks combine principles of quantum computing with neural networks.

    In this simulation:
    - A classical neural network is trained.
    - A PennyLane-based quantum circuit is shown.
    """)
