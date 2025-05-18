import pandas as pd
import json
from pathlib import Path

# Eurostat country codes to names mapping
COUNTRY_CODES = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'HR': 'Croatia',
    'CY': 'Cyprus',
    'CZ': 'Czech Republic',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'FI': 'Finland',
    'FR': 'France',
    'DE': 'Germany',
    'EL': 'Greece',
    'HU': 'Hungary',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LV': 'Latvia',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'SK': 'Slovakia',
    'SI': 'Slovenia',
    'ES': 'Spain',
    'SE': 'Sweden',
    'UK': 'United Kingdom'
}

# ISO 3-letter to 2-letter code mapping
ISO3_TO_ISO2 = {
    'AUT': 'AT',
    'BEL': 'BE',
    'BGR': 'BG',
    'HRV': 'HR',
    'CYP': 'CY',
    'CZE': 'CZ',
    'DNK': 'DK',
    'EST': 'EE',
    'FIN': 'FI',
    'FRA': 'FR',
    'DEU': 'DE',
    'GRC': 'EL',
    'HUN': 'HU',
    'IRL': 'IE',
    'ITA': 'IT',
    'LVA': 'LV',
    'LTU': 'LT',
    'LUX': 'LU',
    'MLT': 'MT',
    'NLD': 'NL',
    'POL': 'PL',
    'PRT': 'PT',
    'ROU': 'RO',
    'SVK': 'SK',
    'SVN': 'SI',
    'ESP': 'ES',
    'SWE': 'SE',
    'GBR': 'UK'
}

# Reverse mapping for easier lookup
NAME_TO_CODE = {v: k for k, v in COUNTRY_CODES.items()}

def create_correlation_map():
    # Get the list of visualization files
    vis_dir = Path('visualizations')
    if not vis_dir.exists():
        print("Visualizations directory not found!")
        return
    
    # Get all HTML files and organize them by country
    correlations_by_country = {}
    for file in vis_dir.glob('*.html'):
        # Extract country code from filename (last part before .html)
        country_code = file.stem.split('_')[-1]
        # Only include if it's a valid country code
        if country_code in COUNTRY_CODES:
            country_name = COUNTRY_CODES[country_code]
            if country_name not in correlations_by_country:
                correlations_by_country[country_name] = []
            correlations_by_country[country_name].append(file.name)
    
    # Create a DataFrame with country names and number of correlations
    country_data = pd.DataFrame([
        {'country': country, 'correlations': len(files)}
        for country, files in correlations_by_country.items()
    ])
    
    # Sort countries by number of correlations
    country_data = country_data.sort_values('correlations', ascending=False)
    
    # For JS: country name to code
    country_name_to_code = {name: code for code, name in COUNTRY_CODES.items()}
    
    # Create the HTML content in parts
    html_head = f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Pastel Europe Map</title>
    <style>
        html, body {{
            background: #fff;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            gap: 20px;
            padding: 40px 20px;
        }}
        #map {{
            flex: 1;
            background: #f7f3ec;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            padding: 20px;
            min-width: 900px;
            min-height: 600px;
        }}
        #correlations {{
            width: 400px;
            background: white;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-left: 0;
        }}
        h2 {{
            color: #1d1d1f;
            margin-top: 0;
        }}
        #selected-country {{
            font-size: 1.5em;
            font-weight: 600;
            color: #007aff;
            margin-bottom: 20px;
        }}
        .correlation-link {{
            display: block;
            padding: 15px;
            margin: 10px 0;
            background: #f5f5f7;
            border-radius: 10px;
            color: #1d1d1f;
            text-decoration: none;
            transition: all 0.3s ease;
        }}
        .correlation-link:hover {{
            background: #e5e5ea;
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <div class='container'>
        <div id='map'>
            <svg id='svgmap' width='900' height='600' style='background:#f7f3ec;'></svg>
        </div>
        <div id='correlations'>
            <h2>Correlations</h2>
            <div id='selected-country'>Select a country</div>
            <div id='correlation-list'></div>
        </div>
    </div>
    <script src='https://d3js.org/d3.v7.min.js'></script>
    <script>
        const correlationsByCountry = {json.dumps(correlations_by_country)};
        // More vibrant pastel palette
        const pastelPalette = [
            // Browns (Deep to Pale)
            '#3B2F2F', '#5E3A2F', '#8B5E3C', '#B97A57', '#D4A373',
            // Reds/Rusts (Earthy Red Spectrum)
            '#7B3F00', '#A0522D', '#B85744', '#C6725B', '#D99882',
            // Oranges/Golds
            '#A65E2E', '#CC7722', '#DAA520', '#EDC373', '#F1D3A2',
            // Greens (Natural Spectrum)
            '#4B5320', '#6B8E23', '#8F9779', '#A3B18A', '#CAD2C5',
            // Cool Greys / Stone Colors
            '#6E6658', '#8B8378', '#A9A9A9', '#C0B9A9', '#D3D0C4',
            // Earth-Inspired Accents
            '#5F8575', '#8FB996', '#A7988A', '#C89B7B', '#B3A580',
            // Dark Earth Colors
            '#3D2B1F', '#2F2F2F', '#46483C', '#5A4E3C', '#624A2E',
            // Light Neutrals
            '#EEE5D6', '#F2E2C4', '#E4D6C0', '#E6DBB9', '#F9F5ED'
        ];
        // List of European country names (from your COUNTRY_CODES)
        const europeanNames = Object.keys({json.dumps(NAME_TO_CODE)});
        // Map country name to ISO code for matching GeoJSON
        const nameToCode = {json.dumps(NAME_TO_CODE)};
        // Add ISO3 to ISO2 mapping
        const ISO3_TO_ISO2 = {json.dumps(ISO3_TO_ISO2)};
        // Add COUNTRY_CODES mapping
        const COUNTRY_CODES = {json.dumps(COUNTRY_CODES)};
        // Assign earth tone colors to each European country
        let colorMap = {{}};
        europeanNames.forEach((name, i) => {{
            colorMap[name] = pastelPalette[i % pastelPalette.length];
        }});
        const width = 900, height = 600;
        const svg = d3.select('#svgmap')
            .attr('width', width)
            .attr('height', height);
        // Draw ocean/sea as a warm beige rect behind everything
        svg.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', width)
            .attr('height', height)
            .attr('fill', '#D2B48C');  // Warm Beige for ocean
        const projection = d3.geoMercator()
            .center([15, 54])
            .scale(400)
            .translate([width / 2, height / 2]);
        const path = d3.geoPath().projection(projection);
        d3.json('world.geojson').then(function(geojson) {{
            svg.selectAll('path')
                .data(geojson.features)
                .enter()
                .append('path')
                .attr('d', path)
                .attr('fill', d => {{
                    // Try to match by ISO code or country name
                    const code = d.id || d.properties.ISO_A2 || d.properties.iso_a2 || d.properties.ADM0_A3 || d.properties.ADM0_A3_IS || d.properties.ISO2 || d.properties.ISO3;
                    const name = d.properties.name || d.properties.NAME || d.properties.ADMIN;
                    // First try to match by ISO3 code
                    if (code && ISO3_TO_ISO2[code]) {{
                        const iso2Code = ISO3_TO_ISO2[code];
                        const countryName = COUNTRY_CODES[iso2Code];
                        if (countryName && colorMap[countryName]) return colorMap[countryName];
                    }}
                    
                    // If that fails, try to match by name
                    if (europeanNames.includes(name)) {{
                        if (colorMap[name]) return colorMap[name];
                    }}
                    
                    // If it's land but not Europe, fill with a muted earth tone
                    if (d.geometry && d.geometry.type && d.geometry.type !== 'Polygon' && d.geometry.type !== 'MultiPolygon') return '#D2B48C';  // Warm Beige for ocean
                    return '#f7f3ec';  // Light beige for non-European land
                }})
                .attr('stroke', '#40342A')  // Peat color for borders
                .attr('stroke-width', 1)
                .attr('cursor', d => {{
                    const code = d.id || d.properties.ISO_A2 || d.properties.iso_a2 || d.properties.ADM0_A3 || d.properties.ADM0_A3_IS || d.properties.ISO2 || d.properties.ISO3;
                    const name = d.properties.name || d.properties.NAME || d.properties.ADMIN;
                    
                    // First try to match by ISO3 code
                    if (code && ISO3_TO_ISO2[code]) {{
                        const iso2Code = ISO3_TO_ISO2[code];
                        const countryName = COUNTRY_CODES[iso2Code];
                        if (countryName && correlationsByCountry[countryName]) return 'pointer';
                    }}
                    
                    // If that fails, try to match by name
                    if (europeanNames.includes(name) && correlationsByCountry[name]) return 'pointer';
                    
                    return 'default';
                }})
                .on('click', function(event, d) {{
                    const code = d.id || d.properties.ISO_A2 || d.properties.iso_a2 || d.properties.ADM0_A3 || d.properties.ADM0_A3_IS || d.properties.ISO2 || d.properties.ISO3;
                    const name = d.properties.name || d.properties.NAME || d.properties.ADMIN;
                    
                    // First try to match by ISO3 code
                    if (code && ISO3_TO_ISO2[code]) {{
                        const iso2Code = ISO3_TO_ISO2[code];
                        const countryName = COUNTRY_CODES[iso2Code];
                        if (countryName && correlationsByCountry[countryName]) {{
                            showCorrelations(countryName);
                            return;
                        }}
                    }}
                    
                    // If that fails, try to match by name
                    if (europeanNames.includes(name) && correlationsByCountry[name]) {{
                        showCorrelations(name);
                    }}
                }});
        }});
        function showCorrelations(country) {{
            const correlations = correlationsByCountry[country] || [];
            const countryElement = document.getElementById('selected-country');
            const listElement = document.getElementById('correlation-list');
            countryElement.textContent = country;
            listElement.innerHTML = '';
            if (correlations.length === 0) {{
                listElement.innerHTML = '<p>No correlations found for this country.</p>';
                return;
            }}
            correlations.forEach(file => {{
                const link = document.createElement('a');
                link.href = 'visualizations/' + file;
                link.className = 'correlation-link';
                link.target = '_blank';
                // Extract variable names and correlation from filename
                const parts = file.replace('.html', '').split('_');
                const vsIndex = parts.indexOf('vs');
                
                // Get all parts before 'vs' for first variable (skip 'spurious')
                const var1 = parts.slice(1, vsIndex).join(' ')
                    .split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                    .join(' ');
                
                // Get all parts between 'vs' and country code for second variable
                const var2 = parts.slice(vsIndex + 1, -1).join(' ')
                    .split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                    .join(' ');
                
                // Extract correlation value from the visualization file
                fetch('visualizations/' + file)
                    .then(response => response.text())
                    .then(html => {{
                        const match = html.match(/Correlation: ([-+]?\d*\.\d+)/);
                        const correlation = match ? match[1] : 'N/A';
                        link.textContent = var1 + ' vs ' + var2 + ' (' + correlation + ')';
                    }})
                    .catch(() => {{
                        link.textContent = var1 + ' vs ' + var2;
                    }});
                
                listElement.appendChild(link);
            }});
        }}
    </script>
</body>
</html>"""

    # Assign the HTML to html_content before writing
    html_content = html_head
    with open('correlation_map.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Created D3.js interactive correlation map: correlation_map.html")

if __name__ == "__main__":
    create_correlation_map()