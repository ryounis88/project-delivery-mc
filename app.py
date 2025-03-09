import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from io import BytesIO

def monte_carlo_simulation(duration_mean, duration_std, cost_mean, cost_std, quality_mean, quality_std, num_simulations):
    """Performs Monte Carlo simulation."""
    duration = np.random.normal(duration_mean, duration_std, num_simulations)
    cost = np.random.normal(cost_mean, cost_std, num_simulations)
    quality = np.random.normal(quality_mean, quality_std, num_simulations)
    return duration, cost, quality

def calculate_weighted_score(duration, cost, quality, duration_weight, cost_weight, quality_weight):
    """Calculates a weighted score for each simulation."""
    duration_normalized = (duration - np.min(duration)) / (np.max(duration) - np.min(duration))
    cost_normalized = 1 - ((cost - np.min(cost)) / (np.max(cost) - np.min(cost))) #lower cost is better
    quality_normalized = (quality - np.min(quality)) / (np.max(quality) - np.min(quality))

    weighted_score = (duration_normalized * duration_weight) + (cost_normalized * cost_weight) + (quality_normalized * quality_weight)
    return weighted_score

def calculate_p_value(scores1, scores2):
    """Calculates the p-value using a t-test."""
    t_stat, p_value = stats.ttest_ind(scores1, scores2)
    return p_value

def generate_pdf_cdf(data, title):
    """Generates and displays PDF and CDF charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PDF
    axes[0].hist(data, bins=30, density=True, alpha=0.6, color='g')
    axes[0].set_title(f'PDF: {title}')

    # CDF
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1].plot(sorted_data, cdf)
    axes[1].set_title(f'CDF: {title}')

    return fig

def main():
    st.title("Project Delivery Method Monte Carlo Simulation")

    num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)

    st.sidebar.header("Parameter Weights")
    duration_weight = st.sidebar.slider("Duration Weight", 0.0, 1.0, 0.3)
    cost_weight = st.sidebar.slider("Cost Weight", 0.0, 1.0, 0.4)
    quality_weight = st.sidebar.slider("Quality Weight", 0.0, 1.0, 0.3)

    st.header("Project Delivery Method Inputs")

    dbb_duration_mean = st.number_input("DBB Duration Mean (months)", 1, 100, 24)
    dbb_duration_std = st.number_input("DBB Duration Std Dev (months)", 1, 20, 3)
    dbb_cost_mean = st.number_input("DBB Cost Mean ($)", 100000, 10000000, 1000000)
    dbb_cost_std = st.number_input("DBB Cost Std Dev ($)", 10000, 1000000, 100000)
    dbb_quality_mean = st.number_input("DBB Quality Mean (PQI)", 1, 10, 7)
    dbb_quality_std = st.number_input("DBB Quality Std Dev (PQI)", 0.1, 2.0, 0.5)

    db_duration_mean = st.number_input("Design-Build Duration Mean (months)", 1, 100, 20)
    db_duration_std = st.number_input("Design-Build Duration Std Dev (months)", 1, 20, 2.5)
    db_cost_mean = st.number_input("Design-Build Cost Mean ($)", 100000, 10000000, 900000)
    db_cost_std = st.number_input("Design-Build Cost Std Dev ($)", 10000, 1000000, 90000)
    db_quality_mean = st.number_input("Design-Build Quality Mean (PQI)", 1, 10, 8)
    db_quality_std = st.number_input("Design-Build Quality Std Dev (PQI)", 0.1, 2.0, 0.4)

    cmar_duration_mean = st.number_input("CMaR Duration Mean (months)", 1, 100, 18)
    cmar_duration_std = st.number_input("CMaR Duration Std Dev (months)", 1, 20, 2)
    cmar_cost_mean = st.number_input("CMaR Cost Mean ($)", 100000, 10000000, 1100000)
    cmar_cost_std = st.number_input("CMaR Cost Std Dev ($)", 10000, 1000000, 110000)
    cmar_quality_mean = st.number_input("CMaR Quality Mean (PQI)", 1, 10, 9)
    cmar_quality_std = st.number_input("CMaR Quality Std Dev (PQI)", 0.1, 2.0, 0.3)

    if st.button("Run Simulation"):
        dbb_duration, dbb_cost, dbb_quality = monte_carlo_simulation(dbb_duration_mean, dbb_duration_std, dbb_cost_mean, dbb_cost_std, dbb_quality_mean, dbb_quality_std, num_simulations)
        db_duration, db_cost, db_quality = monte_carlo_simulation(db_duration_mean, db_duration_std, db_cost_mean, db_cost_std, db_quality_mean, db_quality_std, num_simulations)
        cmar_duration, cmar_cost, cmar_quality = monte_carlo_simulation(cmar_duration_mean, cmar_duration_std, cmar_cost_mean, cmar_cost_std, cmar_quality_mean, cmar_quality_std, num_simulations)

        dbb_score = calculate_weighted_score(dbb_duration, dbb_cost, dbb_quality, duration_weight, cost_weight, quality_weight)
        db_score = calculate_weighted_score(db_duration, db_cost, db_quality, duration_weight, cost_weight, quality_weight)
        cmar_score = calculate_weighted_score(cmar_duration, cmar_cost, cmar_quality, duration_weight, cost_weight, quality_weight)

        st.subheader("Simulation Results")

        results_df = pd.DataFrame({
            "DBB Score": dbb_score,
            "Design-Build Score": db_score,
            "CMaR Score": cmar_score
        })
        st.dataframe(results_df)

        best_method = results_df