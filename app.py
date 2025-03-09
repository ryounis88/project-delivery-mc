import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import ttest_ind

# Set page configuration
st.set_page_config(
    page_title="Project Delivery Method Selection Tool",
    page_icon="🏗️",
    layout="wide"
)

# Application title and description
st.title("Project Delivery Method Selection Tool")
st.markdown("""
This application uses Monte Carlo simulation to help select the optimal project delivery method based on parameters 
including Duration, Cost, and Quality. Compare different delivery methods to make data-driven decisions.
""")

# Initialize session state if not already done
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, 100)

# Weights for criteria
st.sidebar.header("Criteria Weights")
st.sidebar.markdown("Set the importance of each criterion (must sum to 1.0)")

# Use columns for weights to display them side-by-side
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    weight_duration = st.number_input("Duration Weight", 0.0, 1.0, 0.33, 0.01)
with col2:
    weight_cost = st.number_input("Cost Weight", 0.0, 1.0, 0.33, 0.01)
with col3:
    weight_quality = st.number_input("Quality Weight", 0.0, 1.0, 0.34, 0.01)

# Check if weights sum to 1.0
total_weight = weight_duration + weight_cost + weight_quality
if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning(f"Weights sum to {total_weight:.2f}, but should sum to 1.0")

# Create tabs for different delivery methods
st.header("Project Delivery Methods")

tab1, tab2, tab3 = st.tabs(["Design-Bid-Build (DBB)", "Design-Build (DB)", "Construction Management at Risk (CMaR)"])

# Function to create input fields for each method
def create_method_inputs(tab, method_name):
    with tab:
        st.subheader(f"{method_name} Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Duration (months)**")
            min_duration = st.number_input(f"{method_name} Min Duration", 1.0, 100.0, 12.0, 0.5, key=f"{method_name}_min_duration")
            most_likely_duration = st.number_input(f"{method_name} Most Likely Duration", 1.0, 120.0, 18.0, 0.5, key=f"{method_name}_ml_duration")
            max_duration = st.number_input(f"{method_name} Max Duration", 1.0, 150.0, 24.0, 0.5, key=f"{method_name}_max_duration")
            
            st.markdown("**Cost ($ millions)**")
            min_cost = st.number_input(f"{method_name} Min Cost", 0.1, 1000.0, 5.0, 0.1, key=f"{method_name}_min_cost")
            most_likely_cost = st.number_input(f"{method_name} Most Likely Cost", 0.1, 1200.0, 7.5, 0.1, key=f"{method_name}_ml_cost")
            max_cost = st.number_input(f"{method_name} Max Cost", 0.1, 1500.0, 10.0, 0.1, key=f"{method_name}_max_cost")
        
        with col2:
            st.markdown("**Quality (PQI Score, 0-100)**")
            min_quality = st.number_input(f"{method_name} Min Quality", 0.0, 100.0, 60.0, 1.0, key=f"{method_name}_min_quality")
            most_likely_quality = st.number_input(f"{method_name} Most Likely Quality", 0.0, 100.0, 75.0, 1.0, key=f"{method_name}_ml_quality")
            max_quality = st.number_input(f"{method_name} Max Quality", 0.0, 100.0, 90.0, 1.0, key=f"{method_name}_max_quality")
        
        return {
            "duration": (min_duration, most_likely_duration, max_duration),
            "cost": (min_cost, most_likely_cost, max_cost),
            "quality": (min_quality, most_likely_quality, max_quality)
        }

# Get inputs for each method
dbb_params = create_method_inputs(tab1, "DBB")
db_params = create_method_inputs(tab2, "DB")
cmar_params = create_method_inputs(tab3, "CMaR")

# Run simulation button
if st.button("Run Monte Carlo Simulation", type="primary"):
    st.session_state.run_simulation = True

# Function to generate triangular distribution samples
def generate_triangular_samples(min_val, mode_val, max_val, num_samples):
    return np.random.triangular(min_val, mode_val, max_val, num_samples)

# Function to normalize scores (lower is better for duration and cost, higher is better for quality)
def normalize_scores(durations, costs, qualities, methods):
    # For duration and cost, lower is better
    # For quality, higher is better
    
    # Combine all methods' data
    all_durations = np.concatenate([durations[method] for method in methods])
    all_costs = np.concatenate([costs[method] for method in methods])
    all_qualities = np.concatenate([qualities[method] for method in methods])
    
    # Initialize normalized scores dictionaries
    norm_durations = {}
    norm_costs = {}
    norm_qualities = {}
    
    # Calculate min and max for each parameter across all methods
    min_duration, max_duration = np.min(all_durations), np.max(all_durations)
    min_cost, max_cost = np.min(all_costs), np.max(all_costs)
    min_quality, max_quality = np.min(all_qualities), np.max(all_qualities)
    
    # Normalize each method's scores
    for method in methods:
        # For duration and cost, 1 is best (shortest duration, lowest cost)
        # so we invert the normalization
        if max_duration != min_duration:
            norm_durations[method] = 1 - (durations[method] - min_duration) / (max_duration - min_duration)
        else:
            norm_durations[method] = np.ones_like(durations[method])
            
        if max_cost != min_cost:
            norm_costs[method] = 1 - (costs[method] - min_cost) / (max_cost - min_cost)
        else:
            norm_costs[method] = np.ones_like(costs[method])
        
        # For quality, higher is better
        if max_quality != min_quality:
            norm_qualities[method] = (qualities[method] - min_quality) / (max_quality - min_quality)
        else:
            norm_qualities[method] = np.ones_like(qualities[method])
    
    return norm_durations, norm_costs, norm_qualities

# Function to calculate weighted scores
def calculate_weighted_scores(norm_durations, norm_costs, norm_qualities, methods, weights):
    weighted_scores = {}
    
    for method in methods:
        weighted_scores[method] = (
            weights[0] * norm_durations[method] +
            weights[1] * norm_costs[method] +
            weights[2] * norm_qualities[method]
        )
    
    return weighted_scores

# Function to run statistical tests and interpret results
def run_statistical_analysis(weighted_scores, methods):
    results = {}
    
    # Find the best method based on mean weighted score
    mean_scores = {method: np.mean(weighted_scores[method]) for method in methods}
    best_method = max(mean_scores, key=mean_scores.get)
    
    # Compare best method with others using t-test
    for method in methods:
        if method != best_method:
            t_stat, p_value = ttest_ind(
                weighted_scores[best_method],
                weighted_scores[method],
                equal_var=False
            )
            
            results[method] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
    
    return best_method, results

# Function to display main simulation results
def display_simulation_results(durations, costs, qualities, weighted_scores, methods):
    # Calculate statistics for each method
    stats_data = []
    
    for method in methods:
        stats_data.append({
            "Method": method,
            "Mean Duration (months)": np.mean(durations[method]),
            "Mean Cost ($ millions)": np.mean(costs[method]),
            "Mean Quality (PQI)": np.mean(qualities[method]),
            "Mean Weighted Score": np.mean(weighted_scores[method]),
            "Score Std Dev": np.std(weighted_scores[method]),
            "95% CI Lower": np.percentile(weighted_scores[method], 2.5),
            "95% CI Upper": np.percentile(weighted_scores[method], 97.5)
        })
    
    # Convert to DataFrame for display
    stats_df = pd.DataFrame(stats_data)
    
    # Round numerical values for better display
    display_df = stats_df.round(2)
    
    # Highlight the best method (highest weighted score)
    best_method_idx = display_df["Mean Weighted Score"].idxmax()
    
    # Display the results
    st.subheader("Simulation Results Summary")
    
    # Use DataFrame styler to highlight best method
    def highlight_best(s):
        is_best = s.index == best_method_idx
        return ['background-color: #c6f6c6' if v else '' for v in is_best]
    
    st.dataframe(display_df.style.apply(highlight_best, axis=1))
    
    return stats_df

# Function to create probability density function (PDF) plot
def create_pdf_plot(weighted_scores, methods):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        sns.kdeplot(weighted_scores[method], label=method, ax=ax)
    
    plt.title("Probability Density Function of Weighted Scores")
    plt.xlabel("Weighted Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return fig

# Function to create cumulative distribution function (CDF) plot
def create_cdf_plot(weighted_scores, methods):
    fig = go.Figure()
    
    for method in methods:
        sorted_scores = np.sort(weighted_scores[method])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        fig.add_trace(go.Scatter(
            x=sorted_scores,
            y=cumulative,
            mode='lines',
            name=method
        ))
    
    fig.update_layout(
        title="Cumulative Distribution Function of Weighted Scores",
        xaxis_title="Weighted Score",
        yaxis_title="Cumulative Probability",
        legend_title="Method",
        height=500
    )
    
    return fig

# Function to interpret results for laypeople
def interpret_results(best_method, stats_df, statistical_results, methods, durations, costs):
    st.subheader("Interpretation of Results")
    
    # Calculate potential savings
    best_method_duration = np.mean(durations[best_method])
    best_method_cost = np.mean(costs[best_method])
    
    time_savings = {}
    cost_savings = {}
    
    for method in methods:
        if method != best_method:
            time_diff = np.mean(durations[method]) - best_method_duration
            cost_diff = np.mean(costs[method]) - best_method_cost
            
            time_savings[method] = time_diff
            cost_savings[method] = cost_diff
    
    # Create practical significance message
    practical_msg = f"Based on {num_simulations} simulations, **{best_method}** appears to be the optimal project delivery method.\n\n"
    
    practical_msg += "### Practical Significance\n\n"
    
    for method in methods:
        if method != best_method:
            time_diff = time_savings[method]
            cost_diff = cost_savings[method]
            
            if time_diff > 0:
                practical_msg += f"- Compared to {method}, {best_method} is expected to save approximately "
                practical_msg += f"**{time_diff:.1f} months** in duration"
            else:
                practical_msg += f"- {best_method} may take **{-time_diff:.1f} more months** than {method}"
                
            if cost_diff > 0:
                practical_msg += f" and **${cost_diff:.2f} million** in cost.\n"
            else:
                practical_msg += f" but may cost **${-cost_diff:.2f} million more**.\n"
    
    practical_msg += "\n### Statistical Significance\n\n"
    
    # Add statistical significance interpretation
    for method in methods:
        if method != best_method:
            p_value = statistical_results[method]["p_value"]
            
            if p_value < 0.001:
                sig_level = "strong statistical significance (p < 0.001)"
            elif p_value < 0.01:
                sig_level = f"high statistical significance (p = {p_value:.3f})"
            elif p_value < 0.05:
                sig_level = f"statistical significance (p = {p_value:.3f})"
            else:
                sig_level = f"no statistical significance (p = {p_value:.3f})"
            
            practical_msg += f"- The advantage of {best_method} over {method} shows {sig_level}.\n"
            
            if p_value < 0.05:
                practical_msg += f"  This means we can be confident that the observed differences are not due to random chance.\n\n"
            else:
                practical_msg += f"  This means the observed differences might be due to random variation, and both methods could perform similarly in practice.\n\n"
    
    st.markdown(practical_msg)
    
    # Add recommendation box
    st.success(f"**Recommendation:** Based on the weighted criteria and simulation results, **{best_method}** is recommended as the optimal project delivery method for this project.")

# Run the simulation if button is clicked
if st.session_state.run_simulation:
    with st.spinner("Running Monte Carlo simulation..."):
        # Setup methods list
        methods = ["DBB", "DB", "CMaR"]
        
        # Generate triangular distribution samples for each parameter and method
        durations = {
            "DBB": generate_triangular_samples(*dbb_params["duration"], num_simulations),
            "DB": generate_triangular_samples(*db_params["duration"], num_simulations),
            "CMaR": generate_triangular_samples(*cmar_params["duration"], num_simulations)
        }
        
        costs = {
            "DBB": generate_triangular_samples(*dbb_params["cost"], num_simulations),
            "DB": generate_triangular_samples(*db_params["cost"], num_simulations),
            "CMaR": generate_triangular_samples(*cmar_params["cost"], num_simulations)
        }
        
        qualities = {
            "DBB": generate_triangular_samples(*dbb_params["quality"], num_simulations),
            "DB": generate_triangular_samples(*db_params["quality"], num_simulations),
            "CMaR": generate_triangular_samples(*cmar_params["quality"], num_simulations)
        }
        
        # Normalize scores
        norm_durations, norm_costs, norm_qualities = normalize_scores(durations, costs, qualities, methods)
        
        # Calculate weighted scores
        weights = [weight_duration, weight_cost, weight_quality]
        weighted_scores = calculate_weighted_scores(norm_durations, norm_costs, norm_qualities, methods, weights)
        
        # Run statistical analysis
        best_method, statistical_results = run_statistical_analysis(weighted_scores, methods)
        
        # Display results in multiple columns
        st.markdown("## Simulation Results")
        
        # Display summary statistics
        stats_df = display_simulation_results(durations, costs, qualities, weighted_scores, methods)
        
        # Create visualizations
        st.markdown("## Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Probability Density Function (PDF)")
            pdf_fig = create_pdf_plot(weighted_scores, methods)
            st.pyplot(pdf_fig)
        
        with col2:
            st.subheader("Cumulative Distribution Function (CDF)")
            cdf_fig = create_cdf_plot(weighted_scores, methods)
            st.plotly_chart(cdf_fig, use_container_width=True)
        
        # Interpret results and make recommendations
        interpret_results(best_method, stats_df, statistical_results, methods, durations, costs)
        
        # Reset simulation flag for next run
        st.session_state.run_simulation = False