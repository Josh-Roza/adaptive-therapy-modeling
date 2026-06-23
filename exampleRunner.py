import simvis

#simvis takes one of netlogos output csvs as input, 
#it finds what parameters were varied in the experiment and creats a plot with the step count. 
simvis.visualize('abt.csv')

#It will find whether or not 1 or 2 parameters vary and create either a 2d or 3d graph. 
# For this example 2 parameters vary so it will create a 3d graph.
simvis.visualize('ABTandPROtoDP.csv')

#If more than 2 parameters vary you have to specify which ones to graph.
simvis.visualize('prolifAndDrugMolecule.csv', var1='drugMolecule', var2='proliferationP-DP')