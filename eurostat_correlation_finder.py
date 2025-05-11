import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import plotly.io as pio
import eurostat
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Add OpenAI API key configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=openai_api_key)

# Country code to country name mapping
COUNTRY_CODES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "EL": "Greece",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "UK": "United Kingdom",
    "NO": "Norway",
    "IS": "Iceland",
    "CH": "Switzerland",
    "TR": "Turkey",
    "EA19": "Euro area (19 countries)",
    "EU27_2020": "European Union (27 countries, from 2020)",
    "EU28": "European Union (28 countries, pre-Brexit)",
    # Add more codes as needed
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_ai_explanation(var1: str, var2: str, correlation: float) -> str:
    """
    Generate a satirical academic explanation using AI.
    """
    prompt = f"""Generate a satirical academic explanation for a spurious correlation between {var1} and {var2} (correlation coefficient: {correlation:.2f}).\nThe explanation should:\n1. Sound like a real academic paper.\n2. Reference real academic theories (e.g., cognitive dissonance, game theory, network effects, etc.) rather than fictional ones.\n3. Use academic language and jargon.\n4. Be humorous but maintain a serious academic tone.\n5. Reference modern concepts (AI, blockchain, quantum physics, etc.).\n6. Include the correlation coefficient in proper statistical format (r = X.XX).\n7. Be 3-4 sentences long.\n\nFormat the response as a single paragraph without any additional text or formatting."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a satirical academic researcher who creates humorous but convincing explanations for spurious correlations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating AI explanation: {str(e)}")
        # Fallback to a basic explanation if AI generation fails
        return f"Our groundbreaking research reveals a correlation (r = {correlation:.2f}) between {var1} and {var2}. This finding challenges traditional economic theories and suggests a deeper connection that mainstream science has yet to acknowledge."

def generate_satirical_explanation(var1: str, var2: str, correlation: float) -> str:
    """
    Generate a humorous, satirical explanation for a spurious correlation using AI.
    """
    try:
        return generate_ai_explanation(var1, var2, correlation)
    except Exception as e:
        print(f"Error in generate_satirical_explanation: {str(e)}")
        # Fallback to a basic explanation if AI generation fails
        return f"Our groundbreaking research reveals a correlation (r = {correlation:.2f}) between {var1} and {var2}. This finding challenges traditional economic theories and suggests a deeper connection that mainstream science has yet to acknowledge."

def fetch_eurostat_data() -> Dict[str, pd.DataFrame]:
    """
    Fetch real Eurostat data for various indicators.
    """
    # Core indicators
    core_indicators = {
        'gdp': {
            'code': 'nama_10_gdp',
            'filter_pars': {'na_item': 'B1GQ', 'unit': 'CP_MEUR'}
        },
        'unemployment': {
            'code': 'une_rt_a',
            'filter_pars': {'age': 'Y15-74', 'unit': 'PC_ACT', 'sex': 'T'}
        },
        'population': {
            'code': 'demo_pjan',
            'filter_pars': {'age': 'TOTAL', 'sex': 'T', 'unit': 'NR'}
        },
        'education': {
            'code': 'edat_lfse_03',
            'filter_pars': {'sex': 'T', 'age': 'Y15-64', 'isced11': 'ED5-8', 'unit': 'PC'}
        },
        'life_expectancy': {
            'code': 'demo_mlexpec',
            'filter_pars': {'sex': 'T', 'age': 'Y_LT1', 'unit': 'YR'}
        }
    }

    # Fun random indicators
    fun_indicators = {
        # Eurostat fun/quirky indicators
        'cheese_production': {
            'code': 'apro_cpsh1',
            'filter_pars': {'prod': 'CHS', 'unit': 'KG_HAB'}
        },
        'beer_production': {
            'code': 'apro_cpsh1',
            'filter_pars': {'prod': 'BEER', 'unit': 'L_HAB'}
        },
        'ice_cream_production': {
            'code': 'apro_cpsh1',
            'filter_pars': {'prod': 'ICEC', 'unit': 'KG_HAB'}
        },
        'cinema_admissions': {
            'code': 'cult_ac_cinatt',
            'filter_pars': {'unit': 'PER_INHAB', 'indic_cul': 'ATT'}
        },
        'museums_per_capita': {
            'code': 'cult_ent_mus',
            'filter_pars': {'unit': 'PER_100000', 'indic_cul': 'MUS'}
        },
        'public_libraries': {
            'code': 'cult_ent_lib',
            'filter_pars': {'unit': 'NUMBER', 'indic_cul': 'LIB'}
        },
        'sports_club_membership': {
            'code': 'hlth_silc_10',
            'filter_pars': {'unit': 'PC', 'indic_he': 'SPRTCLUB'}
        },
        'fruit_consumption': {
            'code': 'hlth_ehis_fv3e',
            'filter_pars': {'unit': 'PC', 'sex': 'T', 'age': 'TOTAL'}
        },
        'chocolate_production': {
            'code': 'apro_cpsh1',
            'filter_pars': {'prod': 'CHOC', 'unit': 'T'}
        },
        'mcdonalds_restaurants': {
            'code': 'sbs_na_1a_se_r2',
            'filter_pars': {'nace_r2': 'I5610', 'unit': 'NR'}
        },
        'unesco_sites': {
            'code': 'cult_ent_her',
            'filter_pars': {'unit': 'NUMBER', 'indic_cul': 'HER'}
        },
        'coffee_imports': {
            'code': 'apro_cpsh1',
            'filter_pars': {'prod': 'COFFEE', 'unit': 'KG_HAB'}
        },
        'book_titles_published': {
            'code': 'cult_ent_book',
            'filter_pars': {'unit': 'NUMBER', 'indic_cul': 'BOOK'}
        },
        'electric_cars': {
            'code': 'road_eqs_carmot',
            'filter_pars': {'unit': 'NR', 'fuel': 'ELC'}
        },
        'rooms_per_dwelling': {
            'code': 'ilc_lvho01',
            'filter_pars': {'unit': 'NR', 'tenure': 'TOTAL'}
        },
        'cinema_screens': {
            'code': 'cult_ent_cinsc',
            'filter_pars': {'unit': 'NUMBER', 'indic_cul': 'CINSC'}
        },
        'theme_park_visits': {
            'code': 'tour_occ_ninat',
            'filter_pars': {'unit': 'NR', 'nace_r2': 'R9321'}
        },
        'football_clubs': {
            'code': 'sbs_na_1a_se_r2',
            'filter_pars': {'nace_r2': 'R9312', 'unit': 'NR'}
        },
        'average_tv_time': {
            'code': 'ilc_lvps04',
            'filter_pars': {'unit': 'HOURS', 'indic_il': 'TV'}
        },
        # Placeholders for external/quirky indicators (to be scraped or added manually)
        # 'pets_per_household': 'external',
        # 'nobel_laureates': 'external',
        # 'eurovision_points': 'external',
        # 'ufo_sightings': 'external',
        # 'public_holidays': 'external',
        # 'saunas_per_capita': 'external',
        # 'ski_resorts': 'external',
        # 'selfies_per_year': 'external',
        # 'michelin_starred_restaurants': 'external',
        # 'castles': 'external',
        # 'eurovision_wins': 'external',
        # 'mcdonalds_per_capita': 'external',
    }

    # Combine all indicators
    all_indicators = {**core_indicators, **fun_indicators}
    
    # Fetch data for each indicator
    data_dict = {}
    for name, config in all_indicators.items():
        try:
            # Download data
            data = eurostat.get_data_df(config['code'], filter_pars=config['filter_pars'])
            
            if data is not None:
                print(f"\nSuccessfully fetched data for {name}")
                print(f"Columns: {data.columns.tolist()}")
                print(f"Shape: {data.shape}")
                print(f"Sample data:\n{data.head()}")
                data_dict[name] = data
            else:
                print(f"Error fetching data for {name}: No data returned")
        except Exception as e:
            print(f"Error fetching data for {name}: {str(e)}")
    
    return data_dict

def create_spurious_correlation_plot(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    country: str,
    correlation: float,
    output_dir: str
) -> None:
    """
    Create a custom HTML page with an Apple-like style for a spurious correlation.
    """
    try:
        # Get full country name from code
        country_name = COUNTRY_CODES.get(country, country)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add first variable
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[var1],
            name=var1.replace('_', ' ').title(),
            mode='lines+markers',
            line=dict(color='#007aff', width=3),
            marker=dict(size=7, symbol='diamond')
        ))
        
        # Add second variable
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[var2],
            name=var2.replace('_', ' ').title(),
            mode='lines+markers',
            line=dict(color='#ff3b30', width=3),
            marker=dict(size=7, symbol='circle'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Correlation: {correlation:.3f}",
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title='Year',
                tickfont=dict(size=16),
                gridcolor='#f0f0f0'
            ),
            yaxis=dict(
                title=dict(text=var1.replace('_', ' ').title(), font=dict(color='#007aff')),
                tickfont=dict(color='#007aff', size=16),
                gridcolor='#f0f0f0'
            ),
            yaxis2=dict(
                title=dict(text=var2.replace('_', ' ').title(), font=dict(color='#ff3b30')),
                tickfont=dict(color='#ff3b30', size=16),
                overlaying='y',
                side='right',
                gridcolor='#f0f0f0'
            ),
            legend=dict(
                font=dict(size=16),
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Generate Plotly div
        plotly_div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, config={"displayModeBar": False})
        
        # Satirical explanation
        explanation = generate_satirical_explanation(
            var1.replace('_', ' ').title(),
            var2.replace('_', ' ').title(),
            correlation
        )
        
        # Create filename
        filename = f"spurious_{var1}_vs_{var2}_{country}.html"
        filepath = os.path.join(output_dir, filename)
        
        # Apple-like HTML template
        html = f"""
        <!DOCTYPE html>
        <html lang='en'>
        <head>
          <meta charset='UTF-8'>
          <meta name='viewport' content='width=device-width, initial-scale=1.0'>
          <title>Spurious Correlation: {var1} vs {var2}</title>
          <link rel="preconnect" href="https://fonts.googleapis.com">
          <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
          <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600;700&display=swap" rel="stylesheet">
          <style>
            body {{
              font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
              background: #fff;
              color: #111;
              margin: 0;
              padding: 20px;
              line-height: 1.6;
            }}
            .container {{
              max-width: 1000px;
              margin: 0 auto;
              padding: 32px;
              background: #fafbfc;
              border-radius: 20px;
              box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            }}
            .headline {{
              font-size: 2.5rem;
              font-weight: 700;
              margin-bottom: 0.5em;
              letter-spacing: -0.02em;
              line-height: 1.2;
              color: #1d1d1f;
            }}
            .subtitle {{
              font-size: 1.5rem;
              font-weight: 600;
              color: #ff3b30;
              margin-bottom: 1em;
            }}
            .correlation-details {{
              font-size: 1.2rem;
              color: #86868b;
              margin-bottom: 2em;
            }}
            .plotly-chart {{
              margin: 2em 0;
              border-radius: 10px;
              overflow: hidden;
              box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            }}
            .callout {{
              background: #f5f5f7;
              border-left: 5px solid #007aff;
              border-radius: 12px;
              padding: 1.5em;
              margin: 2em 0;
              font-size: 1.2rem;
              color: #1d1d1f;
            }}
            .callout-title {{
              font-weight: 600;
              color: #007aff;
              font-size: 1.3em;
              margin-bottom: 0.5em;
            }}
            @media (max-width: 768px) {{
              body {{ padding: 10px; }}
              .container {{ padding: 20px; }}
              .headline {{ font-size: 2rem; }}
              .subtitle {{ font-size: 1.2rem; }}
            }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="headline">{var1.replace('_', ' ').title()}<br>vs<br>{var2.replace('_', ' ').title()}</div>
            <div class="subtitle">Country: {country_name}</div>
            <div class="correlation-details">
              <strong>Correlation Coefficient:</strong> {correlation:.3f}<br>
              <strong>Time Period:</strong> {data.index.min().year} - {data.index.max().year}<br>
              <strong>Number of Data Points:</strong> {len(data)}
            </div>
            <div class="plotly-chart">{plotly_div}</div>
            <div class="callout">
              <div class="callout-title">Satirical Academic Explanation</div>
              {explanation}
            </div>
          </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Created visualization: {filename}")
        
    except Exception as e:
        print(f"Error creating visualization for {var1} vs {var2} ({country}): {str(e)}")

def process_eurostat_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Process Eurostat data to make it suitable for correlation analysis.
    """
    processed_data = {}
    print("\nProcessing Eurostat data...")
    
    # Process each dataset
    for name, df in data_dict.items():
        if df is not None and isinstance(df, pd.DataFrame):
            try:
                print(f"\nProcessing {name}")
                print(f"Original columns: {df.columns.tolist()}")
                print(f"Original shape: {df.shape}")
                
                # Get the geo column name (it might be 'geo' or 'geo\TIME_PERIOD')
                geo_col = next((col for col in df.columns if 'geo' in col), None)
                if geo_col is None:
                    print(f"No geo column found in {name}")
                    continue
                
                # Get all numeric columns (these are the year columns)
                year_cols = df.select_dtypes(include=[np.number]).columns
                print(f"Numeric columns: {year_cols.tolist()}")
                
                if len(year_cols) > 0:
                    # For datasets with potential duplicates, we'll take the first entry for each country-year combination
                    melted = pd.melt(
                        df,
                        id_vars=[col for col in df.columns if col not in year_cols],
                        value_vars=year_cols,
                        var_name='year',
                        value_name='value'
                    )
                    
                    # Convert year to datetime
                    melted['year'] = pd.to_datetime(melted['year'], format='%Y')
                    
                    # Remove any rows with missing values
                    melted = melted.dropna(subset=['value'])
                    
                    if not melted.empty:
                        # For each country and year, take the first non-null value
                        # This handles duplicate entries by taking the first valid value
                        pivoted = melted.groupby(['year', geo_col])['value'].first().unstack()
                        
                        # Remove any columns (countries) that have too many missing values
                        min_valid_years = 5  # Require at least 5 years of data
                        valid_countries = pivoted.columns[pivoted.count() >= min_valid_years]
                        pivoted = pivoted[valid_countries]
                        
                        if not pivoted.empty:
                            processed_data[name] = pivoted
                            print(f"Successfully processed {name} with shape {pivoted.shape}")
                            print(f"Sample of processed data:\n{pivoted.head()}")
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
    
    print(f"\nProcessed {len(processed_data)} datasets")
    return processed_data

def find_spurious_correlations(
    data_dict: Dict[str, pd.DataFrame],
    output_dir: str = 'visualizations'
) -> None:
    """
    Find and visualize the most absurd correlations with satirical explanations.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the data
    processed_data = process_eurostat_data(data_dict)
    
    # Get list of countries that appear in most datasets
    country_counts = {}
    for df in processed_data.values():
        for country in df.columns:
            country_counts[country] = country_counts.get(country, 0) + 1
    
    # Keep only countries that appear in at least 5 datasets
    common_countries = [country for country, count in country_counts.items() if count >= 5]
    print(f"\nFound {len(common_countries)} countries with sufficient data")
    
    # For each country, find correlations between different indicators
    for country in common_countries:
        print(f"\nProcessing country: {country}")
        
        # Create a DataFrame for this country's indicators
        country_data = pd.DataFrame()
        
        # Collect all available indicators for this country
        for name, df in processed_data.items():
            if country in df.columns:
                country_data[name] = df[country]
        
        print(f"Found {len(country_data.columns)} indicators for {country}")
        
        if len(country_data.columns) < 2:
            print(f"Skipping {country} - not enough indicators")
            continue
        
        # Calculate correlations between all variable pairs
        correlations = []
        for var1 in country_data.columns:
            for var2 in country_data.columns:
                if var1 != var2:
                    # Remove any rows with NaN values
                    valid_data = country_data[[var1, var2]].dropna()
                    if len(valid_data) >= 5:  # Only calculate if we have enough valid data points
                        corr, p_value = stats.pearsonr(
                            valid_data[var1],
                            valid_data[var2]
                        )
                        correlations.append({
                            'var1': var1,
                            'var2': var2,
                            'correlation': corr,
                            'p_value': p_value,
                            'n_points': len(valid_data)
                        })
        
        print(f"Found {len(correlations)} valid correlations for {country}")
        
        # Convert to DataFrame and sort by absolute correlation
        if correlations:
            corr_df = pd.DataFrame(correlations)
            corr_df['abs_correlation'] = corr_df['correlation'].abs()
            # Sort by correlation strength and number of data points
            corr_df = corr_df.sort_values(['abs_correlation', 'n_points'], ascending=[False, False])
            
            # Create visualizations for correlations with abs_correlation >= 0.8
            print(f"Creating visualizations for correlations with |correlation| >= 0.8 in {country}")
            for _, row in corr_df[corr_df['abs_correlation'] >= 0.8].iterrows():
                var1, var2 = row['var1'], row['var2']
                # Get the data for these variables
                if var1 not in country_data.columns or var2 not in country_data.columns:
                    print(f"Skipping {var1} vs {var2} for {country}: one or both variables missing.")
                    continue
                plot_data = country_data[[var1, var2]].dropna()
                # Require at least 5 data points for both variables
                if plot_data.shape[0] < 5:
                    print(f"Skipping {var1} vs {var2} for {country}: not enough data points ({plot_data.shape[0]}).")
                    continue
                if plot_data[var1].equals(plot_data[var2]):
                    print(f"Warning: {var1} and {var2} are identical for {country}, skipping plot.")
                    continue
                print(f"Plotting {var1} vs {var2} for {country}: {plot_data.head(10)}")
                create_spurious_correlation_plot(
                    plot_data,
                    var1,
                    var2,
                    country,
                    row['correlation'],
                    output_dir
                )

if __name__ == "__main__":
    print("Starting Eurostat Correlation Finder...")
    
    # Fetch real Eurostat data
    print("\nFetching Eurostat data...")
    data_dict = fetch_eurostat_data()
    
    # Create spurious correlation visualizations
    print("\nFinding and visualizing correlations...")
    find_spurious_correlations(data_dict, output_dir='visualizations') 