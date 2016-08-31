import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

class xsrc: 
	def __init__(self, event_file, coordinate_file): 
		self.evts = Table.read(event_file,hdu=1)
		# read ra/dec of events (had to use CIAO: dmlist "evt_1229.fits[cols EQPOS]" data > radec_1229.txt (trim leading lines, '(', ')', and ',') 
		ec = Table.read(coordinate_file,format='ascii')
		self.evts.add_column(ec['RA'])
		self.evts.add_column(ec['DEC'])
		self.deg2arcs = 3600.0
	def query_position(self,ra,dec,r = 10.): 
		''' query_position: get events within some distance (r in arcseconds) of position (ra,dec) '''
		offx = self.deg2arcs*(self.evts['RA']-ra) 
		offy = self.deg2arcs*(self.evts['DEC']-dec)
		gevts = np.where(sqrt(offx**2+offy**2)<=r)
		return gevts 


# make class: 
e = xsrc('evt_1229.fits','radec_1229.txt')
# read the 2mass sources (RA/Dec in RAJ2000,DEJ2000): 
a = Table.read('src_1229_2mass',format='ascii')	


