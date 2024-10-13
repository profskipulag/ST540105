import re
import os
import glob
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import arviz as az
from fastapi import FastAPI, Request
from shapely.geometry import Polygon
from geojson import Feature, LineString, FeatureCollection
import geojson
from matplotlib.colors import rgb2hex



class WebsiteApi:

    def __init__(self, data_dir="app/data/"):

        self.data_dir = data_dir

        # get list of all netcdf files in the data directory ...
        glob_string = os.path.join( data_dir,"*.nc")
        
        files = glob.glob(glob_string)
        
        # ... split the filenames to get the run details. All DT5404 files are of the form 
        # YYYYMMDD_HH_HH_<model>_<emulatortype>_emulator.nc and all DT5405 files are of type 
        # YYYMMDD_HH__HH_<model>_<emulatortype>inference.nc ...
        data = [re.findall(r'.*((\d{8}_\d{2})_(\d*)_(.*))_(.*)_(.*)_data.nc', f)[0] for f in files]

        # ... put the run information in a dataframe ...
        df = pd.DataFrame(data,columns=['name','date','hours','model','type','data'])

        df['display_text'] = df.apply(
            lambda r: "|START " +
                      "H".join([r['date'][:8], r['date'][9:]]) + 
                      " |HOURS " + r['hours'] +
                      " |MODEL " + r['model'] +
                      " |EMULATOR " + r['type'] + "|",
            axis=1
        )

        # ... fix the run start dates ..
        df['date'] = df['date'].apply(lambda r: datetime.datetime.strptime(r,"%Y%m%d_%H"))

        # ... add the filenames ...
        df['file'] = files

        # ... pivot so there's one row in the dataframe per emulator inference pair
        self.df = df.pivot( 
                index=['date','hours','model','type','name', 'display_text'], 
                values=['file'], 
                columns=['data']
            ).reset_index()
        

    def get_available_runs(self):
        """Return json of available runs
        """
        
        return(self.df[['name','display_text']].values.tolist())
        

    def load_run(self, run):
        """Loads the selected run into memory
        """

        self.current_inference_file  = self.df.loc[ self.df['name'] == '20210720_00_49_Fall3D' , ('file','inference')].item() 

        self.current_emulator_file = self.df.loc[ self.df['name'] == '20210720_00_49_Fall3D' , ('file','emulator')].item() 

        self.current_observation_file = self.df.loc[ self.df['name'] == '20210720_00_49_Fall3D' , ('file','observation')].item() 



        #self.current_inference_file  = self.df.loc[ self.df['name'] == '20210407_00_24_Fall3D' , ('file','inference')].item() 
        #self.current_emulator_file = self.df.loc[ self.df['name'] == '20210407_00_24_Fall3D' , ('file','emulator')].item() 

        #self.current_observation_file = self.df.loc[ self.df['name'] == '20210407_00_24_Fall3D' , ('file','observation')].item() 

        self.inference = az.from_netcdf(self.current_inference_file)

        self.emulator = xr.load_dataset(self.current_emulator_file)

        self.observation = xr.load_dataset(self.current_observation_file)

        # fix times!!!
        self.inference.posterior['source_start'] = self.inference.posterior.source_start.astype('datetime64[h]')
        self.inference.posterior['time'] = self.inference.posterior.time.astype('datetime64[h]')
        self.emulator['date'] = self.emulator['date'].values.astype('datetime64[h]')
        self.emulator['source.source_start'] = self.emulator['source_start'].values.astype('datetime64[h]')
        

    def get_bounding_box(self,n_pts = 20):
        """Returns the bounding box of the currently loaded run as a geodict object
        for serialisation into json
        """
        
        
        
        lon_min = self.inference.posterior.lon.min() #-23.0
        lon_max = self.inference.posterior.lon.max() #-21.5
        lat_min = self.inference.posterior.lat.min() #63.7
        lat_max  = self.inference.posterior.lat.max() #64.3
        
        bbox_N = np.array([np.linspace(lon_min, lon_max, n_pts),[lat_max]*n_pts]).T 
        bbox_E = np.array([[lon_max]*n_pts, np.linspace(lat_max, lat_min, n_pts)]).T
        bbox_S = np.array([np.linspace(lon_max, lon_min, n_pts), [lat_min]*n_pts]).T 
        bbox_W = np.array([[lon_min]*n_pts, np.linspace(lat_min, lat_max, n_pts)]).T
        bbox = np.vstack([bbox_N, bbox_E, bbox_S, bbox_W])
        
        polygon_geom = Polygon(bbox)
        polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])       
        bbox_as_geodict = polygon.to_geo_dict()
        
        return(bbox_as_geodict)

    def get_station_coords(self):
        """Returns json friendly liost of station ids and locations
        """
        
        coords = []
        for name in self.observation:
            local_id, species, value_type = name.split("#")
            if value_type == 'value':
                attrs = self.observation[name].attrs
                coords.append({
                    'local_id':local_id,
                    'lat':attrs['lat'],
                    'lon':attrs['lon']
                })
        
        return(coords)
       


    def get_exceedence_contours(self, source_start, threshold):
        """
        """

        N_samples = len(self.inference.posterior.sample)    

        source_start0 = self.inference.posterior.time.values[source_start]
        
        exceedance_posterior = (
            (
                self
                .inference
                .posterior['surface_grid']
                .sel(time = source_start0) 
                 > threshold
            )
                 .sum(dim='sample')/N_samples
        )

                
        contour = exceedance_posterior.plot.contour(
                x='lon',
                y='lat',
                cmap='Reds',
                levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                #cbar_kwargs=cbar_kwargs
            );
        
        feature_collections = []
        
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
        geojson_dump = geojson.dumps(feature_collection, sort_keys=True)
        feature_collections.append({ 
            'date':source_start0.astype(int)/1e6,
            'feature_collection':feature_collection
        })

        return(feature_collections)


    def get_exceedence_stations(self, source_start, threshold):

        N_samples = len(self.inference.posterior.sample)    

        
        exceedence = (
            self
            .inference
            .posterior['conc_matrix']
            .stack({'sample':['chain','draw']})
            >threshold
        ).sum(dim='sample')/N_samples
        

        return([
            {
            'local_id':da.local_id.item(),
            'data':[ {'timestamp':daa.date.astype(int).item()/1e6, 'value' : daa.values.item()} for daa in da]
            }
            for da in exceedence])
        

    def get_cumulative_exceedence_stations(self, threshold):

        N_samples = len(self.inference.posterior.sample)    

        
        exceedence = (
                self
                .inference
                .posterior['conc_matrix']
                .stack({'sample':['chain','draw']})
                >threshold
            ).sum(dim='sample')/N_samples
            
        non_exceedence = 1-exceedence
        
        cum_non_exceedence = non_exceedence.cumprod(dim='date')
        
        cum_exceedence = 1- cum_non_exceedence
        
        return([
            {
            'local_id':da.local_id.item(),
            'data':[ {'timestamp':daa.date.astype(int).item()/1e6, 'value' : daa.values.item()} for daa in da]
            }
            for da in cum_exceedence])   


    def get_violin_flux(self):
        """Get flux data for violins
        """
        stack = self.inference.posterior.stack({'sample':['chain','draw']})
        
        fluxes = [{
            'timestamp':f.source_start.values.astype(int)/1e6,
            'samples':f.values.tolist() }
            for f in stack.flux]

        return(fluxes)
        

    def get_violin_height(self):
        """Get height data for violins
        """
        stack = self.inference.posterior.stack({'sample':['chain','draw']})

        heights = [{
            'timestamp':f.source_start.values.astype(int)/1e6,
            'samples':f.values.tolist() }
            for f in stack.height]

        return(heights)
        
        
    def get_violin_station(self):

        stack = self.inference.posterior.stack({'sample':['chain','draw']})
        
        conc_dict ={
           f.local_id.values.item() :[{'timestamp': ff.date.values.astype(int)/1e6, 'samples':(ff.values*1e6).tolist() }for ff in f] 
            for f in stack.conc_matrix
        }

        return(conc_dict)

    
    def get_sample_contours(self, sample):
        
        print("sample:", sample)

        temp = self.inference.posterior['surface_grid'].sel(sample=sample)
        
        
        feature_collections = []
        for time in temp.time:
            contour = temp.sel(time=time).plot.contour(
                    x='lon',
                    y='lat',
                   # cmap="Reds",
                    levels=[1e-5,1e-4,1e-3,1e-2],
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
        return(feature_collections)  

    
    def get_esp_contours(self, esps):
        
        da = self.emulator['SO2_con_surface']
        
        df = pd.DataFrame(esps).reset_index()
    
        
        df['date'] = df['index'].astype(int).astype('datetime64[ms]' )
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['timestamp'] = df.apply(lambda r: datetime.datetime(year=r['year'],month=r['month'], day=r['day'], hour=r['hour']), axis=1).astype(int)/1000
        df['timestamp'] = df['timestamp'].astype('datetime64[us]')
        
        
        temp = sum([ r['fluxes']*da.sel(source_start=r['timestamp']).interp(height_above_vent=r['heights'],method='linear') for i, r in df.iterrows()])
        
        feature_collections = []
        for time in temp.time:
            contour = temp.sel(time=time).plot.contour(
                    x='lon',
                    y='lat',
                   # cmap="Reds",
                    levels=[1e-5,1e-4,1e-3,1e-2],
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

    def get_esp_station(self,esps):
                
        df = pd.DataFrame(esps).reset_index()
        
        
        df['date'] = df['index'].astype(int).astype('datetime64[ms]' )
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['timestamp'] = df.apply(lambda r: datetime.datetime(year=r['year'],month=r['month'], day=r['day'], hour=r['hour']), axis=1).astype(int)/1000
        df['timestamp'] = df['timestamp'].astype('datetime64[us]')
        
        
            
        temp = sum([    
            self
            .emulator['conc. ground']
            .sel({'source.source_start':r['timestamp']})
            .interp({'source.height_above_vent':r['heights']},method='linear')
            *r['fluxes'] 
            
            for i, r in df.iterrows()])
        
        return([ 
            {
            'local_id':local_id.item(),
            'data': [
                {
                    'timestamp': date.item(),
                    'value': temp.sel(local_id=local_id, date=date).item()
                    } for date in temp.date]
            }  for local_id in temp.local_id
        ])

    def get_esp_drag(self,esps,lat,lon):
        
        df = pd.DataFrame(esps).reset_index()
        
        
        df['date'] = df['index'].astype(int).astype('datetime64[ms]' )
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['timestamp'] = df.apply(lambda r: datetime.datetime(year=r['year'],month=r['month'], day=r['day'], hour=r['hour']), axis=1).astype(int)/1000
        df['timestamp'] = df['timestamp'].astype('datetime64[us]')
        
        
        temp = self.emulator['SO2_con_surface'].interp(lat=lat, lon=lon,method='linear')
        
        
        temp2 = sum([
            temp
            .sel(source_start=r['timestamp'])
            .interp(height_above_vent = r['heights'], method='linear')
            *r['fluxes'] 
            for i, r in df.iterrows()])
        
        return([{'timestamp':time.item(), 'data':temp2.sel(time=time).item()} for time in temp2.time])
                            

    def get_station_obs(self):
        
        conc_dict ={
           f.local_id.values.item() :[{'timestamp': ff.date.values.astype(int)/1e6, 'value':ff.values.tolist() }for ff in f] 
            for f in self.inference.observed_data['obs_conc']
        }
        return(conc_dict)
        

    def get_violin_drag(self, lat, lon):
        
        stack = self.inference.posterior.stack({'sample':['chain','draw']})['surface_grid'].interp(lat=lat, lon=lon)

                
        drag_dict = [{
            'timestamp':f.time.values.astype(int)/1e6,
            'samples':(f.values*1e6).tolist() }
        for f in stack]

        return(drag_dict)

    def get_exceedence_drag(self, lat, lon, threshold):

        N_samples = len(self.inference.posterior.sample)    

        
        exceedence = (
            self
                .inference
                .posterior
                .stack({'sample':['chain','draw']})['surface_grid']
                .interp(lat=lat, lon=lon)
                >threshold
                ).sum(dim='sample')/N_samples
        
        
        drag_dict = [{
            'timestamp':f.time.values.astype(int)/1e6,
            'samples':f.values.tolist() }
        for f in exceedence]        

        return(drag_dict)

    def get_cumulative_exceedence_drag(self, lat, lon, threshold):

        N_samples = len(self.inference.posterior.sample)    

        exceedence = (
            self
                .inference
                .posterior
                .stack({'sample':['chain','draw']})['surface_grid']
                .interp(lat=lat, lon=lon)
                >threshold
                ).sum(dim='sample')/N_samples
        
            
        non_exceedence = 1-exceedence
        
        cum_non_exceedence = non_exceedence.cumprod(dim='time')
        
        cum_exceedence = 1- cum_non_exceedence

        drag_dict = [{
            'timestamp':f.time.values.astype(int)/1e6,
            'samples':f.values.tolist() }
        for f in cum_exceedence]        

        return(drag_dict)
        



    def on_load_website(self, **kwargs):
        """ When the website
        """
        return({
            "runs" : self.get_available_runs()
        })

    
    def on_change_run(self,run, **kwargs):

        self.load_run(run)

        default_sample = 0
        
        default_source_start = 0 #self.inference.posterior.source_start.values[0]
        
        default_threshold = 1e-5
        
        default_drag_lat = 63.93970143318896 #self.inference.posterior.lat.values.mean()
        
        default_drag_lon = -21.983136226023895 #self.inference.posterior.lon.values.mean()

        return({
            
            # emulator extent bounding box
            "bounding_box" : self.get_bounding_box(),
            
            # station locations
            "station_coords": self.get_station_coords(),

            # contours
            "exceedence_contours" : self.get_exceedence_contours(
                source_start = default_source_start,
                threshold = default_threshold
            ),
            "sample_contours" : self.get_sample_contours(
                sample = default_sample
            ),

            # violins
            "violin_flux" : self.get_violin_flux(
            ),

            "violin_height" : self.get_violin_height(
            ),
            
            "violin_station" : self.get_violin_station(
            ),
            
            "violin_drag" : self.get_violin_drag(
                lat = default_drag_lat,
                lon = default_drag_lon
            ),

            # observations
            "station_obs" : self.get_station_obs(
            ),
    
        

        })


        
    def on_change_sample(self, sample, **kwargs):
        return({
            "sample_contours" : self.get_sample_contours(sample=sample),
        })
        

    def on_change_source_start(self, source_start, threshold, **kwargs):

        return({
            "exceedence_contours" : self.get_exceedence_contours(source_start = source_start, threshold = threshold)
        })
        


    def on_change_esps(self, esps, **kwargs): # lat, lon,
        
        return({
            "esp_contours" : self.get_esp_contours(esps=esps),
            "esp_station" : self.get_esp_station(esps=esps),
            #"esp_drag" : self.get_esp_drag(esps=esps, lat=lat, lon=lon)
        })


    def on_change_threshold(self, threshold, source_start, lat, lon, **kwargs):

        return({
            "exceedence_contours" : self.get_exceedence_contours(threshold=threshold, source_start=source_start),
            "exceedence_stations" : self.get_exceedence_stations(threshold=threshold, source_start=source_start),
            "exceedence_drag"     : self.get_exceedence_drag(threshold=threshold, lat=lat, lon=lon),
            "cumulative_exceedence_stations" : self.get_cumulative_exceedence_stations(threshold=threshold),
            "cumulative_exceedence_drag"     : self.get_cumulative_exceedence_drag(threshold=threshold, lat=lat, lon=lon)
        })

    
    def on_change_draggable_station(self,  lat, lon, **kwargs): #threshold

        return({
            "drag_violin" : self.get_violin_drag(lat=lat, lon=lon),
            #"exceedence_drag"     : self.get_exceedence_drag(threshold=threshold, lat=lat, lon=lon),
            #"cumulative_exceedence_drag"     : self.get_cumulative_exceedence_drag(threshold=threshold, lat=lat, lon=lon)


        })


    def on_json_request(self, json):

        request_type = json['type']

        request_data = json['data']

        switch = {
            "load_website" : self.on_load_website,
            "change_run"  : self.on_change_run,
            "change_sample": self.on_change_sample,
            "change_source_start" : self.on_change_source_start,
            "change_sample": self.on_change_sample,
            "change_threshold" : self.on_change_threshold,
            "change_draggable_station" : self.on_change_draggable_station,
            "change_esps" : self.on_change_esps
        }
        
        function = switch[request_type]
        
        result = function(**request_data)

        return(result)
