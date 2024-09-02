#import json
#import geojson
import datetime
import pandas as pd
import xarray as xr
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from matplotlib.colors import rgb2hex
from geojson import Feature, LineString, FeatureCollection

da  = xr.open_dataset("emulator_data.nc")['SO2_con_surface']
da = da.assign_coords({"source_start":da.source_start.values.astype(int)/1000})


app = FastAPI(debug=True)


app.mount("/static", StaticFiles(directory="static"), name='static')

# The user sends the data to perform inference on as json via a POST method
@app.post("/")
async def get_body(request: Request):
    
    #with open('/home/talfan/Documents/projects/dt-geo/d3_test/test_json.json') as f:
    #    d = json.load(f)

    d = await request.json()
        
    df = pd.DataFrame(d).reset_index()
    
    df['date'] = df['index'].astype(int).astype('datetime64[ms]' )
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['timestamp'] = df.apply(lambda r: datetime.datetime(year=r['year'],month=r['month'], day=r['day'], hour=r['hour']), axis=1).astype(int)/1000
    
    
    
    temp = sum([ r['fluxes']*da.sel(source_start=r['timestamp']).interp(height_above_vent=r['heights'],method='linear') for i, r in df.iterrows()])
    
    feature_collections = []
    for time in temp.time:
        contour = temp.sel(time=time).plot.contour(
                x='lon',
                y='lat',
               # cmap="Reds",
                levels=5,
                cmap='Reds'
                #cbar_kwargs=cbar_kwargs
            );
            
        line_features = []
        for collection in contour.collections:
            paths = collection.get_paths()
            color = collection.get_edgecolor()
            for path in paths:
                v = path.vertices
                coordinates = []
                for i in range(len(v)):
                    lat = v[i][0]
                    lon = v[i][1]
                    coordinates.append((lat, lon))
                line = LineString(coordinates)
                properties = {
                    "stroke-width": 3,
                    "stroke": rgb2hex(color[0]),
                }
                line_features.append(Feature(geometry=line, properties=properties))
        
        feature_collection = FeatureCollection(line_features)
        #geojson_dump = geojson.dumps(feature_collection, sort_keys=True)
        feature_collections.append({ 
            'date':time.values.astype(int)/1e6,
            'feature_collection':feature_collection
        })
    return feature_collections
