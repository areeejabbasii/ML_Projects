import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Call Center Dashboard", page_icon="📞", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Call Center Data.csv')
    
    # Convert time to seconds
    def time_to_seconds(time_str):
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
    df['Waiting Time (Seconds)'] = df['Waiting Time (AVG)'].apply(time_to_seconds)
    df['Service Level (Numeric)'] = df['Service Level (20 Seconds)'].str.rstrip('%').astype(float)
    df['Answer Rate (Numeric)'] = df['Answer Rate'].str.rstrip('%').astype(float)
    
    return df

df = load_data()

# Title
st.title("📞 Call Center Performance Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
visualization_type = st.sidebar.selectbox(
    "Select Visualization",
    ["Overview", "Call Analysis", "Performance Distribution", 
     "Heatmap Analysis", "Trend Analysis"]
)

# Visualizations based on selection
if visualization_type == "Overview":
    st.header("📊 Complete Overview")
    
    # Key Metrics at top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Incoming", f"{df['Incoming Calls'].sum():,}", 
                 delta=f"{df['Incoming Calls'].mean():.0f} avg/day")
    with col2:
        st.metric("Total Answered", f"{df['Answered Calls'].sum():,}",
                 delta=f"{(df['Answered Calls'].sum()/df['Incoming Calls'].sum()*100):.1f}%")
    with col3:
        st.metric("Total Abandoned", f"{df['Abandoned Calls'].sum():,}",
                 delta=f"{(df['Abandoned Calls'].sum()/df['Incoming Calls'].sum()*100):.1f}%")
    with col4:
        avg_answer_rate = df['Answer Rate (Numeric)'].mean()
        st.metric("Avg Answer Rate", f"{avg_answer_rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📞 Call Volume Trends")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(df.index, df['Incoming Calls'], alpha=0.3, color='#e74c3c', label='Incoming')
        ax.plot(df.index, df['Incoming Calls'], color='#e74c3c', linewidth=2, alpha=0.8)
        ax.fill_between(df.index, df['Answered Calls'], alpha=0.3, color='#2ecc71', label='Answered')
        ax.plot(df.index, df['Answered Calls'], color='#2ecc71', linewidth=2, alpha=0.8)
        ax.set_xlabel('Day Index', fontsize=11, weight='bold')
        ax.set_ylabel('Number of Calls', fontsize=11, weight='bold')
        ax.set_title('Call Volume Over Time', fontsize=13, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Answer Rate Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if x < 85 else '#f39c12' if x < 90 else '#3498db' if x < 95 else '#2ecc71' 
                 for x in df['Answer Rate (Numeric)']]
        ax.hist(df['Answer Rate (Numeric)'], bins=30, color='#3498db', 
                edgecolor='black', alpha=0.7)
        ax.axvline(df['Answer Rate (Numeric)'].mean(), color='#e74c3c', 
                  linestyle='--', linewidth=2, label=f'Mean: {df["Answer Rate (Numeric)"].mean():.1f}%')
        ax.set_xlabel('Answer Rate (%)', fontsize=11, weight='bold')
        ax.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax.set_title('Distribution of Answer Rates', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
    
    # Bottom row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Service Level Performance")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df.index, df['Service Level (Numeric)'], 
                  c=df['Service Level (Numeric)'], cmap='RdYlGn', 
                  s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.axhline(y=80, color='#e74c3c', linestyle='--', linewidth=2, label='Target: 80%')
        ax.set_xlabel('Day Index', fontsize=11, weight='bold')
        ax.set_ylabel('Service Level (%)', fontsize=11, weight='bold')
        ax.set_title('Service Level Performance', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='Service Level %')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 Performance Summary")
        # Create performance categories
        excellent = len(df[df['Answer Rate (Numeric)'] >= 95])
        good = len(df[(df['Answer Rate (Numeric)'] >= 90) & (df['Answer Rate (Numeric)'] < 95)])
        fair = len(df[(df['Answer Rate (Numeric)'] >= 85) & (df['Answer Rate (Numeric)'] < 90)])
        poor = len(df[df['Answer Rate (Numeric)'] < 85])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Excellent\n(≥95%)', 'Good\n(90-95%)', 'Fair\n(85-90%)', 'Poor\n(<85%)']
        values = [excellent, good, fair, poor]
        colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Days', fontsize=11, weight='bold')
        ax.set_title('Performance Category Distribution', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, weight='bold')
        st.pyplot(fig)

elif visualization_type == "Call Analysis":
    st.header("📊 Comprehensive Call Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Volume Analysis", "🔄 Comparison Charts", "📉 Performance Metrics"])
    
    with tab1:
        st.subheader("Call Volume Analysis")
        
        chart_style = st.radio("Select Chart Style", 
                              ["Stacked Area", "Side-by-Side Bars", "Performance Gradient"], 
                              horizontal=True)
        
        if chart_style == "Stacked Area":
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.fill_between(df.index, 0, df['Answered Calls'], 
                           color='#2ecc71', alpha=0.7, label='Answered Calls')
            ax.fill_between(df.index, df['Answered Calls'], 
                           df['Answered Calls'] + df['Abandoned Calls'],
                           color='#e74c3c', alpha=0.7, label='Abandoned Calls')
            ax.plot(df.index, df['Incoming Calls'], color='#34495e', 
                   linewidth=2, linestyle='--', label='Total Incoming', alpha=0.8)
            ax.set_xlabel('Day Index', fontsize=13, weight='bold')
            ax.set_ylabel('Number of Calls', fontsize=13, weight='bold')
            ax.set_title('Stacked Area: Call Distribution Over Time', fontsize=15, weight='bold')
            ax.legend(fontsize=12, loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        elif chart_style == "Side-by-Side Bars":
            fig, ax = plt.subplots(figsize=(16, 7))
            width = 0.35
            x = df.index
            ax.bar([i - width/2 for i in x], df['Answered Calls'], 
                   width, label='Answered', color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.7)
            ax.bar([i + width/2 for i in x], df['Abandoned Calls'], 
                   width, label='Abandoned', color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=0.7)
            ax.set_xlabel('Day Index', fontsize=13, weight='bold')
            ax.set_ylabel('Number of Calls', fontsize=13, weight='bold')
            ax.set_title('Side-by-Side Comparison: Answered vs Abandoned', fontsize=15, weight='bold')
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        else:  # Performance Gradient
            fig, ax = plt.subplots(figsize=(16, 7))
            answer_rates = df['Answered Calls'] / df['Incoming Calls']
            colors = plt.cm.RdYlGn(answer_rates)
            bars = ax.bar(df.index, df['Incoming Calls'], color=colors, 
                         edgecolor='black', linewidth=0.5, alpha=0.9)
            ax.set_xlabel('Day Index', fontsize=13, weight='bold')
            ax.set_ylabel('Incoming Calls', fontsize=13, weight='bold')
            ax.set_title('Call Volume with Performance Gradient (Red=Low Answer Rate, Green=High)', 
                        fontsize=15, weight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=answer_rates.min(), vmax=answer_rates.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Answer Rate', fontsize=11, weight='bold')
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Incoming vs Answered Relationship")
        
        plot_type = st.radio("Select Visualization", 
                            ["Scatter with Trend", "Hexbin Density", "Sorted Line"], 
                            horizontal=True)
        
        if plot_type == "Scatter with Trend":
            fig, ax = plt.subplots(figsize=(12, 7))
            scatter = ax.scatter(df['Incoming Calls'], df['Answered Calls'], 
                      c=df['Answer Rate (Numeric)'], cmap='RdYlGn', 
                      s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(df['Incoming Calls'], df['Answered Calls'], 1)
            p = np.poly1d(z)
            ax.plot(df['Incoming Calls'], p(df['Incoming Calls']), 
                   color='#e74c3c', linewidth=3, linestyle='--', 
                   label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            
            # Add perfect line
            max_val = max(df['Incoming Calls'].max(), df['Answered Calls'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='Perfect Answer Rate')
            
            ax.set_xlabel('Incoming Calls', fontsize=13, weight='bold')
            ax.set_ylabel('Answered Calls', fontsize=13, weight='bold')
            ax.set_title('Incoming vs Answered Calls (Color = Answer Rate)', fontsize=15, weight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Answer Rate %')
            st.pyplot(fig)
        
        elif plot_type == "Hexbin Density":
            fig, ax = plt.subplots(figsize=(12, 7))
            hexbin = ax.hexbin(df['Incoming Calls'], df['Answered Calls'], 
                              gridsize=30, cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.5)
            ax.set_xlabel('Incoming Calls', fontsize=13, weight='bold')
            ax.set_ylabel('Answered Calls', fontsize=13, weight='bold')
            ax.set_title('Call Density Heatmap: Incoming vs Answered', fontsize=15, weight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(hexbin, ax=ax, label='Frequency')
            st.pyplot(fig)
        
        else:  # Sorted Line
            fig, ax = plt.subplots(figsize=(14, 7))
            df_sorted = df.sort_values('Incoming Calls')
            ax.plot(df_sorted['Incoming Calls'], df_sorted['Answered Calls'], 
                   color='#3498db', linestyle='-', marker='o', markersize=4, 
                   linewidth=2, label='Answered Calls', alpha=0.8)
            ax.fill_between(df_sorted['Incoming Calls'], df_sorted['Answered Calls'], 
                           alpha=0.3, color='#3498db')
            ax.set_xlabel('Incoming Calls (Sorted)', fontsize=13, weight='bold')
            ax.set_ylabel('Answered Calls', fontsize=13, weight='bold')
            ax.set_title('Incoming vs Answered Calls (Sorted View)', fontsize=15, weight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Performance Metrics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df['Abandoned Calls'], df['Answer Rate (Numeric)'], 
                      c=df['Service Level (Numeric)'], cmap='RdYlGn', 
                      s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Abandoned Calls', fontsize=12, weight='bold')
            ax.set_ylabel('Answer Rate (%)', fontsize=12, weight='bold')
            ax.set_title('Abandoned Calls vs Answer Rate', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(ax.collections[0], ax=ax, label='Service Level %')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Box plot for answer rates by performance category
            def categorize(rate):
                if rate >= 95: return 'Excellent'
                elif rate >= 90: return 'Good'
                elif rate >= 85: return 'Fair'
                else: return 'Poor'
            
            df['Category'] = df['Answer Rate (Numeric)'].apply(categorize)
            categories = ['Excellent', 'Good', 'Fair', 'Poor']
            data_to_plot = [df[df['Category'] == cat]['Answered Calls'].values for cat in categories]
            
            bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True,
                           boxprops=dict(facecolor='#3498db', alpha=0.7),
                           medianprops=dict(color='#e74c3c', linewidth=2))
            
            ax.set_xlabel('Performance Category', fontsize=12, weight='bold')
            ax.set_ylabel('Answered Calls', fontsize=12, weight='bold')
            ax.set_title('Call Volume by Performance Category', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)

elif visualization_type == "Performance Distribution":
    st.header("🥧 Performance Distribution")
    
    pie_type = st.radio("Select Pie Chart Type", 
                       ["Total Summary", "Answer Rate Categories", "Donut Chart"])
    
    if pie_type == "Total Summary":
        total_answered = df['Answered Calls'].sum()
        total_abandoned = df['Abandoned Calls'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sizes = [total_answered, total_abandoned]
        labels = ['Answered Calls', 'Abandoned Calls']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0, 0.1)
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=explode, shadow=True,
               textprops={'fontsize': 14, 'weight': 'bold'})
        ax.set_title('Call Center Performance: Answered vs Abandoned', 
                    fontsize=16, weight='bold')
        st.pyplot(fig)
    
    elif pie_type == "Answer Rate Categories":
        def categorize_rate(rate):
            if rate >= 95:
                return 'Excellent (≥95%)'
            elif rate >= 90:
                return 'Good (90-95%)'
            elif rate >= 85:
                return 'Fair (85-90%)'
            else:
                return 'Poor (<85%)'
        
        df['Rate Category'] = df['Answer Rate (Numeric)'].apply(categorize_rate)
        category_counts = df['Rate Category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
        explode = (0.05, 0.05, 0.05, 0.1)
        
        ax.pie(category_counts, labels=category_counts.index, colors=colors,
               autopct='%1.1f%%', startangle=45, explode=explode,
               shadow=True, textprops={'fontsize': 12, 'weight': 'bold'})
        ax.set_title('Distribution of Answer Rate Performance', 
                    fontsize=16, weight='bold')
        st.pyplot(fig)
    
    else:  # Donut Chart
        total_answered = df['Answered Calls'].sum()
        total_abandoned = df['Abandoned Calls'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sizes = [total_answered, total_abandoned]
        labels = ['Answered', 'Abandoned']
        colors = ['#3498db', '#e74c3c']
        explode = (0.05, 0.1)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          explode=explode, shadow=True,
                                          textprops={'fontsize': 13, 'weight': 'bold'})
        
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        ax.text(0, 0, f'{total_answered + total_abandoned:,}\nTotal Calls', 
               ha='center', va='center', fontsize=18, weight='bold')
        ax.set_title('Call Center Performance Overview', 
                    fontsize=18, weight='bold')
        st.pyplot(fig)

elif visualization_type == "Heatmap Analysis":
    st.header("🔥 Heatmap: Waiting Time & Service Level")
    
    heatmap_type = st.radio("Select Heatmap Type", 
                           ["Correlation Matrix", "Time Series Heatmap"])
    
    if heatmap_type == "Correlation Matrix":
        # Convert Answer Speed to seconds
        def time_to_seconds(time_str):
            parts = time_str.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        corr_data = df[['Waiting Time (Seconds)', 'Service Level (Numeric)', 
                       'Answered Calls', 'Abandoned Calls']].copy()
        corr_data['Answer Speed (Seconds)'] = df['Answer Speed (AVG)'].apply(time_to_seconds)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=2, cbar_kws={'label': 'Correlation'},
                   square=True, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation Heatmap: Key Metrics', fontsize=16, weight='bold')
        st.pyplot(fig)
    
    else:  # Time Series Heatmap
        sample_df = df.iloc[::3].head(100)
        heatmap_data = sample_df[['Waiting Time (Seconds)', 'Service Level (Numeric)']].T
        heatmap_data.columns = sample_df['Index'].values
        
        fig, ax = plt.subplots(figsize=(18, 5))
        sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=False, 
                   linewidths=0.3, cbar_kws={'label': 'Value'},
                   vmin=0, vmax=100, ax=ax)
        ax.set_title('Waiting Time vs Service Level Over Time', 
                    fontsize=16, weight='bold')
        ax.set_xlabel('Day Index', fontsize=13, weight='bold')
        ax.set_ylabel('Metrics', fontsize=13, weight='bold')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Waiting Time (sec)', 'Service Level (%)'], rotation=0)
        st.pyplot(fig)

else:  # Trend Analysis
    st.header("📈 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Answered Calls with Moving Average")
        window = st.slider("Moving Average Window", 5, 50, 20)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df.index, df['Answered Calls'], color='#3498db', 
               alpha=0.6, label='Answered Calls')
        moving_avg = df['Answered Calls'].rolling(window=window).mean()
        ax.plot(df.index, moving_avg, color='#e74c3c', linewidth=3, 
               label=f'{window}-Day Moving Average', linestyle='--')
        ax.set_xlabel('Day Index', fontsize=11, weight='bold')
        ax.set_ylabel('Answered Calls', fontsize=11, weight='bold')
        ax.set_title('Answered Calls Trend', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Service Level Trend")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Service Level (Numeric)'], 
               color='#9b59b6', linewidth=2, alpha=0.7)
        ax.axhline(y=80, color='#e74c3c', linestyle='--', 
                  linewidth=2, label='Target: 80%')
        ax.fill_between(df.index, df['Service Level (Numeric)'], 80, 
                       where=(df['Service Level (Numeric)'] >= 80), 
                       color='#2ecc71', alpha=0.3, label='Above Target')
        ax.fill_between(df.index, df['Service Level (Numeric)'], 80, 
                       where=(df['Service Level (Numeric)'] < 80), 
                       color='#e74c3c', alpha=0.3, label='Below Target')
        ax.set_xlabel('Day Index', fontsize=11, weight='bold')
        ax.set_ylabel('Service Level (%)', fontsize=11, weight='bold')
        ax.set_title('Service Level Performance', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #7f8c8d;'>"
    f"<b>Total Days:</b> {len(df)} | "
    f"<b>Best Answer Rate:</b> {df['Answer Rate (Numeric)'].max():.1f}% | "
    f"<b>Worst Answer Rate:</b> {df['Answer Rate (Numeric)'].min():.1f}% | "
    f"<b>Avg Service Level:</b> {df['Service Level (Numeric)'].mean():.1f}%"
    f"</div>", 
    unsafe_allow_html=True
)
