import pandas as pd
import plotly.express as px
import json
import os


# Data Exploration
def dataset_exploration(df):
    table_html = df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
    )
    return table_html


# Data description
def data_exploration(df):
    table_html = df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
    )
    return table_html


def generate_rwanda_map(df):
    geojson_path = 'dummy-data/rwanda_districts.geojson'
    if not os.path.exists(geojson_path):
        return "<p>GeoJSON file not found.</p>"

    with open(geojson_path, 'r', encoding='utf-8') as f:
        rwanda_geojson = json.load(f)

    # Calculate centroids for labels
    centroids = []
    for feature in rwanda_geojson['features']:
        name = feature['properties']['shapeName']
        geom = feature['geometry']
        
        if geom['type'] == 'Polygon':
            coords = geom['coordinates'][0]
        elif geom['type'] == 'MultiPolygon':
            # Use the first polygon in multipolygon for centroid
            coords = geom['coordinates'][0][0]
        else:
            continue
        
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        avg_lon = sum(lons) / len(lons)
        avg_lat = sum(lats) / len(lats)
        
        centroids.append({
            'district': name,
            'lon': avg_lon,
            'lat': avg_lat
        })
    centroids_df = pd.DataFrame(centroids)

    # Aggregate client counts
    df['district'] = df['district'].astype(str).str.strip()
    district_counts = df.groupby('district').size().reset_index(name='client_count')
    
    # Merge with all districts
    all_dist_list = [f['properties']['shapeName'] for f in rwanda_geojson['features']]
    district_counts = pd.merge(pd.DataFrame({'district': all_dist_list}), district_counts, on='district', how='left').fillna(0)
    
    # Merge counts with centroids for display
    display_df = pd.merge(centroids_df, district_counts, on='district')
    # Clear text format: NAME [COUNT]
    display_df['label'] = display_df.apply(lambda r: f"{r['district']}<br>({int(r['client_count'])} clients)", axis=1)

    # Create choropleth
    fig = px.choropleth_mapbox(
        district_counts,
        geojson=rwanda_geojson,
        locations='district',
        featureidkey="properties.shapeName",
        color='client_count',
        color_continuous_scale="Viridis",
        mapbox_style="white-bg",
        center={"lat": -1.9403, "lon": 29.8739},
        zoom=7.6, # Slightly zoomed out for better label spacing
        opacity=0.8,
        labels={'client_count': 'Count'},
        title='<b>Client Distribution by Rwanda District</b>'
    )
    
    # Add text labels on top - High visibility
    fig.add_scattermapbox(
        lat=display_df['lat'],
        lon=display_df['lon'],
        mode='text+markers',
        text=display_df['label'],
        textposition='top center',
        textfont=dict(size=13, color='black'),
        marker=dict(size=5, color='black', opacity=0.3), # Small dot to锚定label
        hoverinfo='skip',
        showlegend=False
    )
    
    fig.update_traces(marker_line_width=1, marker_line_color="white", selector=dict(type='choropleth_mapbox'))
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        paper_bgcolor='white',
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')
