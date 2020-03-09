from openalpr import Alpr
alpr_dir = "C:\Users\ShobhitGangwar\Downloads\openalpr-2.3.0\openalpr"
alpr = Alpr('us', alpr_dir + '/config/openalpr.conf.defaults', alpr_dir + '/runtime_data')
print(alpr.is_loaded())