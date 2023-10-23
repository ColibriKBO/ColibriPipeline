"""
@author: Jilly Ryan

modified by Rachel A. Brown
Jan. 24, 2022
"""

from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord
import pandas as pd
import numpy as np

#RAB modification: make into function with field centre and SR as input

def makeQuery(field_centre, SR):
    '''make Vizier Gaia catalogue query
    input: field centre [RA, dec] in degrees, Search radius [degrees]
    returns: dataframe with query'''
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(coord.SkyCoord(ra=field_centre[0], dec=field_centre[1], 
                                            unit=(u.deg, u.deg), frame = 'icrs'),  
                             radius=coord.Angle(SR, "deg"), catalog='I/345/gaia2',
                             column_filters={'Gmag': '<16'})

    table = result['I/345/gaia2']
    query = np.array(table['RA_ICRS','DE_ICRS','Gmag', 'BP-RP'])
    query = pd.DataFrame(query)
    
    return query

