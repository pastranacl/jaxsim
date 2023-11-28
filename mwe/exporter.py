import numpy as np

OUT_FOLDER = "./output/"

def save_xyz(r_coords, nparts, fileid) -> None:
        """
            Save extended file in XYZ for representation with OVITO
        """
        
        filexyz = open(OUT_FOLDER + str(fileid) + "_particles.xyz","w")
        filexyz.writelines(str(nparts) + "\n\n")
        
        for i in range(0, nparts):        
            filexyz.writelines(str(r_coords[i,0]) + "\t" + str(r_coords[i,1]) + "\t" + str(r_coords[i,2]) + "\n")
                                       
        filexyz.close() 
        