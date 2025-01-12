import numpy as np
from numpy import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

def haversine(lat0, lon0, lat, lon):

      R = 6378.137e3 # this is in meters.  For Earth radius in kilometers use 6372.8 km

      dLat = radians(lat - lat)
      dLon = radians(lon - lon)
      lat1 = radians(lat)
      lat2 = radians(lat)

      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2*asin(sqrt(a))

      return R * c


def get_time_distance(lat, lon, time, lat0, lon0, time0):
    lat0=lat0*np.ones_like

    distance_from_origin = (haversine(lat0,lon0,lat,lon))
    time_from_origin = ((time-time0).astype('timedelta64[ns]').astype(float)/1e9)
    return distance_from_origin, time_from_origin


def time_distance_plot_interactive(interactive_lma, ax):
    lat = interactive_lma.this_lma_lat
    lon = interactive_lma.this_lma_lon
    alt = interactive_lma.this_lma_alt
    time = interactive_lma.this_lma_time
    first = np.nanargmin(time)

    distance_from_origin, time_from_origin = get_time_distance(lat, lon, time, 
                                                               lat[first], lon[first], time[first])
    
    art_out = time_distance_plot(ax, time_from_origin, distance_from_origin)
    
    return art_out
    

def time_distance_plot(ax, time, distance, m_reference_lines = 10, **kwargs):
    m = -m_reference_lines
    x = np.linspace(0, time*-1+m_reference_lines, 100)
    y = x * 2 * 10**4
    yy = x * 10**5
    yyy = x * 10**6
    
    art_out = []
    
    while (m < m_reference_lines):
        art = ax.plot(x+(m/m_reference_lines), y, color = 'b')
        art_out.extend(art)
        art = ax.plot(x+(m/m_reference_lines), yy, color = 'r')
        art_out.extend(art)
        art = ax.plot(x+(m/m_reference_lines), yyy, color = 'g')
        art_out.extend(art)
        m+=1
    
    art = ax.plot(x, y, color = 'b', label = 'Positive Leader')
    art_out.extend(art)
    art = ax.plot(x, yy, color = 'r', label = 'Negative Leader')
    art_out.extend(art)
    art = ax.plot(x, yyy, color = 'g', label = 'Dart Leader')
    art_out.extend(art)
        
    ax.set_ylim(0, max(distance)+1000)
    ax.set_xlim(-0.05, max(time)+0.1) 
    
    sc = ax.scatter(time, distance, **kwargs)
    art_out.append(sc)
    #art = ax.legend(title = "Leader Type Key", loc = 2, fontsize=12, title_fontsize=16, framealpha=1)
    #art_out.extend(art)
    ax.set_title("Lightning Leader Speed", size=24)
    ax.set_xlabel("Time from Origin (s)", size=12)
    ax.set_ylabel("Distance from Origin (m)", size=12)
    
    return art_out
    
    
    
    