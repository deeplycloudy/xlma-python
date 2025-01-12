def haversine(lat0, lon0, lat, lon):

      R = 6378.137e3 # this is in meters.  For Earth radius in kilometers use 6372.8 km

      dLat = radians(lat - lat)
      dLon = radians(lon - lon)
      lat1 = radians(lat)
      lat2 = radians(lat)

      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2*asin(sqrt(a))

      return R * c

# Usage\n",
lon1 = interactive_lma.bounds['x'][1]
lat1 = interactive_lma.bounds['y'][1]
lon2 = interactive_lma.bounds['x'][0]
lat2 = interactive_lma.bounds['x'][0]

print(haversine(lat1, lon1, lat2, lon2))
print('meters')


def get_time_distance(lat, lon, time, lat0, lon0, time0):
    
    lat0=lat0*np.ones_like

    distance_from_origin = (haversine(lat0,lon0,lat,lon))
    time_from_origin = ((time-time0).astype('timedelta64[ns]').astype(float)/1e9)
    return distance_from_origin, time_from_origin


def time_distance_plot_interactive(interactive_lma, ax):
    


def time_distance_plot(ax, time, distance, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    
    
    while (m < 10):
        x = np.linspace(0, (float(interactive_lma.this_lma_time[interactive_lma.this_lma_time.index[first]].strftime('%S.%f'))-float(interactive_lma.this_lma_time[interactive_lma.this_lma_time.index[last]].strftime('%S.%f')))*-1+10, 100)
        y = x * 2 * 10**4
        plt.plot(x+(m/10), y, color = 'b')
        yy = x * 10**5
        plt.plot(x+(m/10), yy, color = 'r')
        yyy = x * 10**6
        plt.plot(x+(m/10), yyy, color = 'g')
        m+=1
    
    
    x = np.linspace(0, (float(interactive_lma.this_lma_time[interactive_lma.this_lma_time.index[first]].strftime('%S.%f'))-float(interactive_lma.this_lma_time[interactive_lma.this_lma_time.index[last]].strftime('%S.%f')))*-1+10, 100)
    y = x * 2 * 10**4
    plt.plot(x, y, color = 'b', label = 'Positive Leader')
    yy = x * 10**5
    plt.plot(x, yy, color = 'r', label = 'Negative Leader')
    yyy = x * 10**6
    plt.plot(x, yyy, color = 'g', label = 'Dart Leader')
        
    plt.ylim(0, max(distance_from_origin)+1000)
    plt.xlim(-0.05, max(time_from_origin)+0.1) 
    
    sc = plt.scatter(time_from_origin, distance_from_origin, zorder=10, c=altitude, cmap="cool", vmin=0, vmax=6)
    plt.colorbar(sc, label='Altitude (km)', spacing='proportional', ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.legend(title = "Leader Type Key", loc = 2, fontsize=12, title_fontsize=16, framealpha=1)
    plt.title("Lightning Leader Speed", size=24)
    plt.xlabel("Time from Origin (s)", size=12)
    plt.ylabel("Distance from Origin (m)", size=12)
    
    plt.show()




    if (len(flash_events['alt'])==0):
        print(" ")
    else:
        m = 0
        while (m < 20):
            x = np.linspace(0, (float((pd.to_datetime(pd.Timestamp(flash_events['time'].values[0]))).strftime('%S.%f'))-float((pd.to_datetime(pd.Timestamp(flash_events['time'].values[-1]))).strftime('%S.%f')))*-1+10, 100)
            #print(x)
            y = x * 2 * 10**4
            bk_plot.ax_vel.plot(x+(m/10), y, color = 'b')
            yy = x * 10**5
            bk_plot.ax_vel.plot(x+(m/10), yy, color = 'r')
            yyy = x * 10**6
            bk_plot.ax_vel.plot(x+(m/10), yyy, color = 'g')
            m+=1
            #print(m)
            
        x = np.linspace(0, (float((pd.to_datetime(pd.Timestamp(flash_events['time'].values[0]))).strftime('%S.%f'))-float((pd.to_datetime(pd.Timestamp(flash_events['time'].values[-1]))).strftime('%S.%f')))*-1+10, 100)
        #print(x)
        y = x * 2 * 10**4
        bk_plot.ax_vel.plot(x, y, color = 'b', label = 'Positive Leader')
        yy = x * 10**5
        bk_plot.ax_vel.plot(x, yy, color = 'r', label = 'Negative Leader')
        yyy = x * 10**6
        bk_plot.ax_vel.plot(x, yyy, color = 'g', label = 'Dart Leader')
    
        bk_plot.ax_vel.set_ylim(0, max(dist_data)+1000)
        bk_plot.ax_vel.set_xlim(-0.05, max(time2_data)+0.1) 