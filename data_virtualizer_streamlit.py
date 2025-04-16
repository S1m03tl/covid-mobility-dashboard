import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# --- Database Configuration ---
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'root_password'
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3307
MYSQL_DB = 'covid_db'
MYSQL_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(MYSQL_DATABASE_URL)
Base = declarative_base()

# --- Database Model ---
class Prediction(Base):
    __tablename__ = 'covid19_predictions'
    id = Column(Integer, primary_key=True)
    location = Column(String(255))
    date = Column(Date)
    new_cases_predicted = Column(Float)
    new_deaths_predicted = Column(Float)
    new_vaccinations_predicted = Column(Float)
    new_cases_actual = Column(Float)
    new_deaths_actual = Column(Float)
    new_vaccinations_actual = Column(Float)
    total_cases = Column(Float)
    total_deaths = Column(Float)
    life_expectancy = Column(Float)
    excess_mortality = Column(Float)
    model_version = Column(String(255))
    created_at = Column(Date)

    def __repr__(self):
        return f"<Prediction(location='{self.location}', date='{self.date}', new_cases_predicted={self.new_cases_predicted}, new_deaths_predicted={self.new_deaths_predicted}, new_vaccinations_predicted={self.new_vaccinations_predicted})>"

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- Data Loading ---
@st.cache_data
def load_data_from_db():
    """Loads data from the MySQL database with caching."""
    session = Session()
    try:
        predictions = session.query(Prediction).all()
        df = pd.DataFrame([
            {
                'location': p.location,
                'date': p.date,
                'new_cases_predicted': p.new_cases_predicted,
                'new_deaths_predicted': p.new_deaths_predicted,
                'new_vaccinations_predicted': p.new_vaccinations_predicted,
                'new_cases_actual': p.new_cases_actual,
                'new_deaths_actual': p.new_deaths_actual,
                'new_vaccinations_actual': p.new_vaccinations_actual,
                'total_cases': p.total_cases,
                'total_deaths': p.total_deaths,
                'life_expectancy': p.life_expectancy,
                'excess_mortality': p.excess_mortality,
                'model_version': p.model_version,
                'created_at': p.created_at
            } for p in predictions
        ])
        return df
    except Exception as e:
        st.error(f"Error loading data from the database: {e}")
        return pd.DataFrame()
    finally:
        session.close()

# --- Evaluation Metrics ---
def calculate_metrics(df, model_name, actual_col, predicted_col):
    """Calculates evaluation metrics (RMSE, MAE, R²) for a model."""
    df_model = df[df['model_version'] == model_name].dropna(subset=[actual_col, predicted_col])
    if df_model.empty:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
    y_true = df_model[actual_col]
    y_pred = df_model[predicted_col]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# --- Visualization Functions ---
def plot_predictions(df, location, actual_col, predicted_col, title):
    """Visualizes model predictions compared to actual data."""
    df_location = df[df['location'] == location].dropna(subset=[actual_col, predicted_col])
    if df_location.empty:
        st.warning(f"No data available to plot predictions for {location}.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_location['date'], df_location[actual_col], label='Actual', color='#1f77b4')  # Blue
    ax.plot(df_location['date'], df_location[predicted_col], label='Predicted', color='#ff7f0e', linestyle='--')  # Orange
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f'{title} for {location}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='-', alpha=0.6)
    fig.tight_layout()
    st.pyplot(fig)

def show_model_performance(df, location, actual_col, predicted_col, title):
    """Displays model performance (RMSE, MAE, R²) for a given location with improved styling."""
    st.subheader(f"Model Performance for {title} in {location}")
    prophet_metrics = calculate_metrics(df, 'prophet_v1', actual_col, predicted_col)
    xgboost_metrics = calculate_metrics(df, 'xgboost_v1', actual_col, predicted_col)

    st.markdown("**Prophet Model:**")
    if np.isnan(prophet_metrics['rmse']):
        st.info("Not enough data to calculate Prophet metrics.")
    else:
        st.metric("RMSE", f"{prophet_metrics['rmse']:.2f}")
        st.metric("MAE", f"{prophet_metrics['mae']:.2f}")
        st.metric("R²", f"{prophet_metrics['r2']:.2f}")

    st.markdown("**XGBoost Model:**")
    if np.isnan(xgboost_metrics['rmse']):
        st.info("Not enough data to calculate XGBoost metrics.")
    else:
        st.metric("RMSE", f"{xgboost_metrics['rmse']:.2f}")
        st.metric("MAE", f"{xgboost_metrics['mae']:.2f}")
        st.metric("R²", f"{xgboost_metrics['r2']:.2f}")

def show_data_analysis(df):
    """Displays visualizations to analyze relationships between variables with improved aesthetics."""
    st.header("Exploratory Data Analysis")

    # Time Series Analysis
    st.subheader("Time Series Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'], df['new_cases_actual'], label='New Cases', color='#1f77b4')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('New Cases', fontsize=10)
        ax.set_title('Daily New Cases', fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.4)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'], df['new_deaths_actual'], label='New Deaths', color='#d62728')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('New Deaths', fontsize=10)
        ax.set_title('Daily New Deaths', fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.4)
        st.pyplot(fig)

    # Cumulative Analysis
    st.subheader("Cumulative Analysis")
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'], df['total_cases'], label='Total Cases', color='#2ca02c')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Total Cases', fontsize=10)
        ax.set_title('Cumulative Cases', fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.4)
        st.pyplot(fig)
    
    with col4:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'], df['total_deaths'], label='Total Deaths', color='#9467bd')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Total Deaths', fontsize=10)
        ax.set_title('Cumulative Deaths', fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.4)
        st.pyplot(fig)

    # Vaccination Analysis
    st.subheader("Vaccination Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['new_vaccinations_actual'], label='New Vaccinations', color='#17becf')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Vaccinations', fontsize=12)
    ax.set_title('Daily Vaccinations', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='-', alpha=0.6)
    st.pyplot(fig)

    # Mortality Analysis
    st.subheader("Mortality Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='date', y='excess_mortality', hue='location', data=df, palette='plasma')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Excess Mortality', fontsize=12)
    ax.set_title('Excess Mortality Over Time by Location', fontsize=14)
    ax.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, linestyle='-', alpha=0.6)
    st.pyplot(fig)

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    corr_matrix = df[['new_cases_actual', 'new_deaths_actual', 'total_cases', 
                      'total_deaths', 'life_expectancy', 'excess_mortality',
                      'new_vaccinations_actual']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 10})
    ax.set_title('Correlation Matrix between Numerical Variables', fontsize=14)
    st.pyplot(fig)

    # Location Comparison
    st.subheader("Location Comparison")
    compare_col = st.selectbox("Select metric to compare locations", 
                             ['new_cases_actual', 'new_deaths_actual', 'total_cases', 
                              'total_deaths', 'life_expectancy', 'excess_mortality'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='date', y=compare_col, hue='location', data=df, palette='tab10')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(compare_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{compare_col.replace("_", " ").title()} by Location', fontsize=14)
    ax.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, linestyle='-', alpha=0.6)
    st.pyplot(fig)

def interactive_plot(df):
    """Creates an interactive section for plotting custom charts with more options."""
    st.header("Interactive Data Exploration")
    st.write("Customize your own plots by selecting columns and chart types.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
    all_cols = numeric_cols + categorical_cols + date_cols

    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X Axis", all_cols)
    with col2:
        y_col = st.selectbox("Y Axis", all_cols)
    with col3:
        color_col = st.selectbox("Color (Optional)", [None] + categorical_cols)
    
    chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Line Plot", "Bar Chart", 
                                           "Histogram", "Box Plot", "Area Chart", "Violin Plot"])
    
    if st.button("Generate Plot"):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            if chart_type == "Scatter Plot":
                sns.scatterplot(x=x_col, y=y_col, hue=color_col, data=df)
            elif chart_type == "Line Plot":
                sns.lineplot(x=x_col, y=y_col, hue=color_col, data=df)
            elif chart_type == "Bar Chart":
                sns.barplot(x=x_col, y=y_col, hue=color_col, data=df)
            elif chart_type == "Histogram":
                sns.histplot(data=df, x=x_col, hue=color_col, kde=True)
            elif chart_type == "Box Plot":
                sns.boxplot(x=x_col, y=y_col, hue=color_col, data=df)
            elif chart_type == "Area Chart":
                if isinstance(df[x_col].dtype, datetime):
                    df_sorted = df.sort_values(by=x_col)
                    if color_col:
                        for cat in df_sorted[color_col].unique():
                            subset = df_sorted[df_sorted[color_col] == cat]
                            ax.fill_between(subset[x_col], subset[y_col], alpha=0.5, label=cat)
                        ax.legend(title=color_col)
                    else:
                        ax.fill_between(df_sorted[x_col], df_sorted[y_col], alpha=0.5)
                else:
                    st.warning("Area chart best used with a time-based X-axis.")
            elif chart_type == "Violin Plot":
                sns.violinplot(x=x_col, y=y_col, hue=color_col, data=df)

            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"{chart_type}: {y_col} vs {x_col}", fontsize=14)
            ax.grid(True, linestyle='-', alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating plot: {e}")

def show_prediction_errors(df):
    """Visualizes prediction errors over time."""
    st.header("Prediction Error Analysis")
    
    # Calculate errors
    df['cases_error'] = df['new_cases_predicted'] - df['new_cases_actual']
    df['deaths_error'] = df['new_deaths_predicted'] - df['new_deaths_actual']
    
    # Plot error trends
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['cases_error'], label='Cases Prediction Error', color='#1f77b4')
    ax.plot(df['date'], df['deaths_error'], label='Deaths Prediction Error', color='#d62728')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prediction Error', fontsize=12)
    ax.set_title('Prediction Errors Over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='-', alpha=0.6)
    st.pyplot(fig)
    
    # Error distribution
    st.subheader("Error Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['cases_error'], kde=True, color='#1f77b4', ax=ax)
        ax.set_xlabel('Cases Prediction Error', fontsize=10)
        ax.set_title('Distribution of Cases Prediction Errors', fontsize=12)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['deaths_error'], kde=True, color='#d62728', ax=ax)
        ax.set_xlabel('Deaths Prediction Error', fontsize=10)
        ax.set_title('Distribution of Deaths Prediction Errors', fontsize=12)
        st.pyplot(fig)

def show_summary_statistics(df):
    """Displays summary statistics for the dataset."""
    st.header("Summary Statistics")
    
    # Overall statistics
    st.subheader("Overall Statistics")
    st.dataframe(df.describe().style.format("{:.2f}"))
    
    # By location statistics
    st.subheader("Statistics by Location")
    location_stats = df.groupby('location').agg({
        'new_cases_actual': ['mean', 'std', 'max'],
        'new_deaths_actual': ['mean', 'std', 'max'],
        'total_cases': 'last',
        'total_deaths': 'last',
        'life_expectancy': 'mean',
        'excess_mortality': 'mean'
    })
    st.dataframe(location_stats.style.format("{:.2f}"))

def main():
    """Main function to run the Streamlit application with improved layout."""
    st.set_page_config(layout="wide", page_title="COVID-19 Dashboard")
    st.title("COVID-19 Prediction Visualization and Analysis Dashboard")

    # Load data
    df = load_data_from_db()
    if df.empty:
        st.error("No data available in the database. Please ensure the data pipeline has run.")
        return

    df['date'] = pd.to_datetime(df['date'])
    locations = sorted(df['location'].unique())

    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    selected_location = st.sidebar.selectbox("Select Location", locations)
    analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                        ["Predictions", "Data Analysis", "Interactive", 
                                         "Error Analysis", "Summary Statistics"])

    # Filter data for selected location
    df_location = df[df['location'] == selected_location]

    # Main content based on selected analysis type
    if analysis_type == "Predictions":
        st.header(f"Predictions for {selected_location}")
        
        # Row 1: Prediction Plots
        st.subheader("Model Predictions vs. Actual Values")
        col1, col2 = st.columns(2)
        with col1:
            plot_predictions(df, selected_location, 'new_cases_actual', 'new_cases_predicted', 'New Cases')
        with col2:
            plot_predictions(df, selected_location, 'new_deaths_actual', 'new_deaths_predicted', 'New Deaths')

        # Row 2: Model Performance Metrics
        st.subheader("Model Performance Metrics")
        col3, col4 = st.columns(2)
        with col3:
            show_model_performance(df, selected_location, 'new_cases_actual', 'new_cases_predicted', 'New Cases')
        with col4:
            show_model_performance(df, selected_location, 'new_deaths_actual', 'new_deaths_predicted', 'New Deaths')

    elif analysis_type == "Data Analysis":
        show_data_analysis(df_location if st.sidebar.checkbox("Filter by selected location") else df)
    
    elif analysis_type == "Interactive":
        interactive_plot(df_location if st.sidebar.checkbox("Filter by selected location") else df)
    
    elif analysis_type == "Error Analysis":
        show_prediction_errors(df_location if st.sidebar.checkbox("Filter by selected location") else df)
    
    elif analysis_type == "Summary Statistics":
        show_summary_statistics(df_location if st.sidebar.checkbox("Filter by selected location") else df)

if __name__ == "__main__":
    main()