import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import plotly.express as px
import io

# Streamlit page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 Customer Segmentation using RFM Analysis & K-Means Clustering")
st.markdown("This application performs customer segmentation based on RFM (Recency, Frequency, Monetary) analysis using the K-Means clustering algorithm. [[5]]")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Source Configuration", "RFM Analysis", "Clustering Results", "Business Insights"])

# Function to calculate RFM metrics
def calculate_rfm(df, config):
    # Ensure date column is in datetime format
    df[config['invoice_date']] = pd.to_datetime(df[config['invoice_date']])
    
    # Calculate reference date (one day after latest transaction)
    current_date = df[config['invoice_date']].max() + pd.DateOffset(1)
    
    # Calculate RFM metrics
    rfm_df = df.groupby(config['customer_id']).agg({
        config['invoice_date']: lambda x: (current_date - x.max()).days,
        config['invoice_no']: 'count',
        'TotalAmount': 'sum'
    })
    
    # Rename columns
    rfm_df.rename(columns={
        config['invoice_date']: 'Recency',
        config['invoice_no']: 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)
    
    return rfm_df

# Function to generate sample data
def generate_sample_data():
    np.random.seed(42)
    
    # Generate sample data
    n_customers = 200
    customer_ids = [f'CUST{i:03d}' for i in range(1, n_customers+1)]
    
    # Generate transaction dates (last 365 days)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Generate sample transactions
    transactions = []
    for customer_id in customer_ids:
        # Random number of transactions per customer (1-50)
        n_transactions = np.random.randint(1, 51)
        
        for _ in range(n_transactions):
            # Random transaction date
            random_days = np.random.randint(0, 365)
            transaction_date = start_date + pd.DateOffset(days=random_days)
            
            # Random invoice number
            invoice_no = f'INV{np.random.randint(10000, 99999)}'
            
            # Random quantity and unit price
            quantity = np.random.randint(1, 10)
            unit_price = round(np.random.uniform(5, 100), 2)
            
            transactions.append([
                customer_id, 
                transaction_date, 
                invoice_no, 
                quantity, 
                unit_price,
                quantity * unit_price
            ])
    
    # Create DataFrame
    columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice', 'TotalAmount']
    df = pd.DataFrame(transactions, columns=columns)
    
    return df

# Page 1: Data Source Configuration
if page == "Data Source Configuration":
    st.header("1. Data Source Configuration")
    
    st.markdown("""
    Select the data source for customer segmentation analysis:
    - **Use Sample Dataset**: Use the provided sample dataset
    - **Upload Your Data**: Upload your own transaction dataset
    """)
    
    data_source = st.radio(
        "Select data source",
        ["Use Sample Dataset", "Upload Your Data"],
        help="Choose between using the provided sample dataset or uploading your own data"
    )
    
    if data_source == "Use Sample Dataset":
        st.info("You have selected to use the provided sample dataset.")
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                df = generate_sample_data()
                st.session_state.df = df
                st.session_state.data_source = "sample"
                
                # Set default config for sample data
                st.session_state.config = {
                    'customer_id': 'CustomerID',
                    'invoice_date': 'InvoiceDate',
                    'invoice_no': 'InvoiceNo',
                    'quantity': 'Quantity',
                    'unit_price': 'UnitPrice'
                }
                
                st.success("Sample data generated successfully!")
                st.experimental_rerun()
        
        if 'df' in st.session_state and st.session_state.data_source == "sample":
            st.subheader("Sample Data Preview")
            st.dataframe(st.session_state.df.head())
            
            st.subheader("Dataset Information")
            buffer = io.StringIO()
            st.session_state.df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            st.subheader("Statistical Summary")
            st.dataframe(st.session_state.df.describe())
            
            if st.button("Proceed to RFM Analysis"):
                st.experimental_rerun()
    
    else:  # Upload Your Data
        st.info("Upload your transaction dataset in CSV format.")
        
        st.markdown("""
        The dataset must contain the following columns:
        - CustomerID: Unique customer identifier
        - InvoiceDate: Transaction date
        - InvoiceNo: Invoice number
        - Quantity: Product quantity
        - UnitPrice: Price per unit
        
        RFM analysis is based on three metrics: Recency (how recently a customer purchased), Frequency (how often they purchase), and Monetary (how much they spend). [[5]]
        """)
        
        # Upload dataset
        uploaded_file = st.file_uploader("Upload your transaction data (CSV)", type="csv")
        
        if uploaded_file is not None:
            # Save dataset to session state
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_source = "upload"
            
            st.success("Dataset uploaded successfully!")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Basic dataset information
            st.subheader("Dataset Information")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
            # Column configuration
            st.subheader("Configure Column Mappings")
            st.markdown("Select columns that match your data:")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                customer_id = st.selectbox("Customer ID", options=df.columns, index=0)
            with col2:
                invoice_date = st.selectbox("Invoice Date", options=df.columns, index=1)
            with col3:
                invoice_no = st.selectbox("Invoice No", options=df.columns, index=2)
            with col4:
                quantity = st.selectbox("Quantity", options=df.columns, index=3)
            with col5:
                unit_price = st.selectbox("Unit Price", options=df.columns, index=4)
            
            # Add TotalAmount column if not present
            if 'TotalAmount' not in df.columns:
                df['TotalAmount'] = df[quantity] * df[unit_price]
            
            # Save configuration to session state
            st.session_state.config = {
                'customer_id': customer_id,
                'invoice_date': invoice_date,
                'invoice_no': invoice_no,
                'quantity': quantity,
                'unit_price': unit_price
            }
            
            st.success("Column mappings configured successfully!")
            
            # Button to proceed to RFM analysis
            if st.button("Proceed to RFM Analysis"):
                st.experimental_rerun()

# Page 2: RFM Analysis
elif page == "RFM Analysis" and 'df' in st.session_state:
    st.header("2. RFM Analysis")
    
    df = st.session_state.df
    config = st.session_state.config
    
    # Calculate RFM
    with st.spinner("Calculating RFM metrics..."):
        rfm_df = calculate_rfm(df, config)
    
    # Save RFM to session state
    st.session_state.rfm_df = rfm_df
    
    st.subheader("RFM Metrics Distribution")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(rfm_df['Recency'], kde=True, ax=ax)
        plt.title('Recency Distribution')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(rfm_df['Frequency'], kde=True, ax=ax)
        plt.title('Frequency Distribution')
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots()
        sns.histplot(rfm_df['Monetary'], kde=True, ax=ax)
        plt.title('Monetary Distribution')
        st.pyplot(fig)
    
    # Show RFM data
    st.subheader("RFM Data Preview")
    st.dataframe(rfm_df.head(10))
    
    # RFM Scoring
    st.subheader("RFM Scoring Method")
    st.markdown("""
    There are two main approaches to RFM segmentation:
    1. Applying K-Means clustering on scaled R, F, M variables
    2. Assigning scores from 1-4 for each R, F, M variable
    """)
    
    scoring_method = st.radio("Choose RFM scoring method", 
                             ["K-Means Clustering (Recommended)", "Manual Scoring (1-4)"])
    
    if scoring_method == "Manual Scoring (1-4)":
        st.info("Manual scoring method selected. This approach assigns scores from 1-4 for each RFM dimension.")
        
        # Sliders for score boundaries
        st.subheader("Set Score Boundaries")
        
        col1, col2 = st.columns(2)
        with col1:
            r_quartiles = st.multiselect("Recency Quartiles (lower is better)", 
                                        [1, 2, 3, 4], default=[1, 2, 3])
        with col2:
            fm_quartiles = st.multiselect("Frequency & Monetary Quartiles (higher is better)", 
                                         [1, 2, 3, 4], default=[2, 3, 4])
        
        # Button to calculate scores
        if st.button("Calculate RFM Scores"):
            st.warning("Manual scoring implementation would go here")
    
    else:
        st.info("K-Means clustering method selected. This is the recommended approach for more accurate segmentation.")
        st.session_state.scoring_method = "kmeans"
        
        # Show preprocessing information
        st.subheader("Data Preprocessing for Clustering")
        st.markdown("""
        Before applying K-Means clustering, RFM data needs:
        1. Outlier removal
        2. Scaling (standardization)
        3. Analysis to determine optimal number of clusters
        """)

# Page 3: Clustering Results
elif page == "Clustering Results" and 'rfm_df' in st.session_state:
    st.header("3. K-Means Clustering Results")
    
    rfm_df = st.session_state.rfm_df
    
    # Remove outliers
    def remove_outliers(df, cols, iqr_factor=1.5):
        df_clean = df.copy()
        for col in cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean
    
    # Remove outliers
    rfm_clean = remove_outliers(rfm_df, ['Recency', 'Frequency', 'Monetary'])
    
    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clean)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm_clean.index, columns=rfm_clean.columns)
    
    # Save to session state
    st.session_state.rfm_scaled = rfm_scaled_df
    
    st.subheader("Optimal Number of Clusters")
    
    # Elbow method
    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled_df)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow method
    fig, ax = plt.subplots()
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    st.pyplot(fig)
    
    st.markdown("""
    Based on the Elbow method, the optimal number of clusters is the point where the inertia decrease begins to slow down.
    For customer segmentation analysis, typically 3-5 clusters provide good results.
    """)
    
    # Select number of clusters
    n_clusters = st.slider("Select number of customer segments", min_value=2, max_value=6, value=4)
    
    # Run K-Means
    if st.button("Run K-Means Clustering"):
        with st.spinner("Performing customer segmentation..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(rfm_scaled_df)
            
            # Add cluster to dataframe
            rfm_clean['Cluster'] = clusters
            rfm_clean['Cluster'] = rfm_clean['Cluster'].astype('category')
            
            # Save to session state
            st.session_state.rfm_results = rfm_clean
            st.session_state.kmeans = kmeans
            
            st.success(f"Customer segmentation completed! {n_clusters} segments created.")
    
    # If clustering results are available
    if 'rfm_results' in st.session_state:
        rfm_results = st.session_state.rfm_results
        
        st.subheader("Cluster Analysis")
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            sns.countplot(x='Cluster', data=rfm_results, ax=ax)
            plt.title('Customer Distribution Across Segments')
            st.pyplot(fig)
        
        with col2:
            # Pie chart
            fig = px.pie(rfm_results, names='Cluster', 
                         title='Percentage of Customers in Each Segment')
            st.plotly_chart(fig)
        
        # Cluster profiles
        st.subheader("Cluster Profiles")
        
        # Calculate average RFM per cluster
        cluster_profile = rfm_results.groupby('Cluster').mean()
        
        # Display as heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cluster_profile, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        plt.title('RFM Values by Customer Segment')
        st.pyplot(fig)
        
        # Interactive scatter plot
        st.subheader("Interactive Visualization")
        
        # Select dimensions for visualization
        x_axis = st.selectbox("X-Axis", ["Recency", "Frequency", "Monetary"], index=0)
        y_axis = st.selectbox("Y-Axis", ["Recency", "Frequency", "Monetary"], index=1)
        color_by = st.selectbox("Color By", ["Cluster"], index=0)
        
        fig = px.scatter(
            rfm_results.reset_index(),
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=['CustomerID'],
            title=f'Customer Segments: {x_axis} vs {y_axis}'
        )
        st.plotly_chart(fig)
        
        # Download results
        st.subheader("Download Results")
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(rfm_results.reset_index())
        
        st.download_button(
            label="Download segmentation results as CSV",
            data=csv,
            file_name='customer_segmentation_results.csv',
            mime='text/csv',
        )

# Page 4: Business Insights
elif page == "Business Insights" and 'rfm_results' in st.session_state:
    st.header("4. Business Insights & Recommendations")
    
    rfm_results = st.session_state.rfm_results
    
    st.markdown("""
    Based on the customer segmentation results using RFM analysis and K-Means clustering, 
    here are the insights and strategic recommendations for each customer segment. [[2]]
    """)
    
    # Analyze each cluster
    cluster_profile = rfm_results.groupby('Cluster').mean()
    
    # Sort clusters based on Recency, Frequency, and Monetary
    cluster_order = cluster_profile.sort_values(
        ['Recency', 'Frequency', 'Monetary'], 
        ascending=[True, False, False]
    ).index
    
    # Assign labels to each cluster
    cluster_labels = {}
    for i, cluster in enumerate(cluster_order):
        recency = cluster_profile.loc[cluster, 'Recency']
        frequency = cluster_profile.loc[cluster, 'Frequency']
        monetary = cluster_profile.loc[cluster, 'Monetary']
        
        if recency < cluster_profile['Recency'].quantile(0.25) and frequency > cluster_profile['Frequency'].quantile(0.75):
            cluster_labels[cluster] = "Champions"
        elif recency < cluster_profile['Recency'].quantile(0.5) and frequency > cluster_profile['Frequency'].quantile(0.5):
            cluster_labels[cluster] = "Loyal Customers"
        elif recency < cluster_profile['Recency'].quantile(0.25) and frequency < cluster_profile['Frequency'].quantile(0.25):
            cluster_labels[cluster] = "Recent Customers"
        elif recency > cluster_profile['Recency'].quantile(0.75) and frequency < cluster_profile['Frequency'].quantile(0.25):
            cluster_labels[cluster] = "Lost Customers"
        elif recency > cluster_profile['Recency'].quantile(0.5) and frequency > cluster_profile['Frequency'].quantile(0.5):
            cluster_labels[cluster] = "Need Attention"
        else:
            cluster_labels[cluster] = f"Segment {cluster}"
    
    # Add labels to dataframe
    rfm_results['Segment_Label'] = rfm_results['Cluster'].map(cluster_labels)
    
    # Save to session state
    st.session_state.rfm_results = rfm_results
    
    # Display analysis per segment
    st.subheader("Customer Segment Analysis")
    
    for cluster in cluster_order:
        label = cluster_labels[cluster]
        recency = cluster_profile.loc[cluster, 'Recency']
        frequency = cluster_profile.loc[cluster, 'Frequency']
        monetary = cluster_profile.loc[cluster, 'Monetary']
        
        with st.expander(f"**{label}** (Cluster {cluster})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recency (days)", f"{recency:.1f}", 
                         help="How recently did the customer purchase? Lower is better")
            with col2:
                st.metric("Frequency", f"{frequency:.1f}", 
                         help="How often do they purchase? Higher is better")
            with col3:
                st.metric("Monetary Value", f"${monetary:.2f}", 
                         help="How much do they spend? Higher is better")
            
            # Provide recommendations based on segment profile
            if label == "Champions":
                st.success("**Profile:** Customers who bought recently, buy often and spend the most.")
                st.markdown("""
                **Recommendations:**
                - Reward these customers with exclusive offers
                - Consider upselling premium products
                - Request reviews and referrals
                - These are your brand advocates
                """)
            
            elif label == "Loyal Customers":
                st.success("**Profile:** Customers who buy often but may not have purchased recently.")
                st.markdown("""
                **Recommendations:**
                - Create personalized loyalty programs
                - Offer special discounts for repeat purchases
                - Send targeted email campaigns
                - Consider win-back offers if recency is increasing
                """)
            
            elif label == "Recent Customers":
                st.warning("**Profile:** Customers who purchased recently but not frequently.")
                st.markdown("""
                **Recommendations:**
                - Focus on converting to repeat customers
                - Send follow-up emails with related products
                - Offer first-repeat purchase discounts
                - Build engagement through content
                """)
            
            elif label == "Lost Customers":
                st.error("**Profile:** Customers who haven't purchased in a long time.")
                st.markdown("""
                **Recommendations:**
                - Consider win-back campaigns with special offers
                - Analyze why they left (if possible)
                - Survey to understand reasons for churn
                - May not be cost-effective to target heavily
                """)
            
            elif label == "Need Attention":
                st.warning("**Profile:** Above average recency and frequency but may need attention.")
                st.markdown("""
                **Recommendations:**
                - Target with personalized retention offers
                - Identify potential reasons for decreased activity
                - Consider special incentives to increase purchase frequency
                - Monitor closely for signs of churn
                """)
            
            else:
                st.info("**Profile:** General customer segment with mixed characteristics.")
                st.markdown("""
                **Recommendations:**
                - Further analyze specific behaviors
                - Test different engagement strategies
                - Consider if additional segmentation is needed
                - Monitor response to marketing campaigns
                """)
            
            # Show sample customers in this segment
            st.subheader("Sample Customers")
            sample_customers = rfm_results[rfm_results['Cluster'] == cluster].head(5)
            st.dataframe(sample_customers[['Recency', 'Frequency', 'Monetary']])
    
    # Overall marketing strategy
    st.subheader("Overall Marketing Strategy Recommendations")
    
    st.markdown("""
    Based on the customer segmentation analysis, here are strategic recommendations for your business:
    
    1. **Focus on High-Value Customers**: Allocate more resources to retain customers in Champions and Loyal Customers segments as they contribute most to revenue. [[7]]
    
    2. **Targeted Retention Programs**: Develop retention programs tailored to each segment's needs, not a one-size-fits-all approach.
    
    3. **New Customer Conversion**: Focus efforts on converting Recent Customers into Loyal Customers through appropriate strategies.
    
    4. **Churn Analysis**: Study Lost Customers' behavior patterns to understand churn causes and prevent future churn.
    
    5. **ROI Measurement**: Implement tracking systems to measure ROI of different marketing strategies for each segment.
    
    This application allows you to interact with the customer segmentation model in real-time and see how parameter changes affect segmentation results. [[4]]
    """)

# Message if data not uploaded
else:
    st.warning("Please configure your data source in the 'Data Source Configuration' section first.")
    st.info("Click on the sidebar to navigate to the data source configuration section.")
