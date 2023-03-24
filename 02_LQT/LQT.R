#devtools::install_github('jdwor/LQT')
# Load the needed library
library(LQT)
library(doParallel)

# Setup parallel
cl <- makeCluster(3)
registerDoParallel(cl)

########## Multi patients script ##########
# This script was ran on Windows! Remember to change the BASE_DIR to your current project folder
########## Set up config structure ###########
BASE_DIR = 'E:/petTOAD'
DATA_DIR = paste0(BASE_DIR, '/data')
UTL_DIR = paste0(DATA_DIR, '/utils')
PRE_DIR = paste0(DATA_DIR, '/preprocessed')
WMH_DIR = paste0(PRE_DIR, '/WMH_segmentation')
MSK_DIR = paste0(PRE_DIR, '/WMH_lesion_masks')

RES_DIR = paste0(BASE_DIR, '/results')
LQT_DIR = paste0(RES_DIR, '/LQT')
LQT_DF_DIR = paste0(LQT_DIR, '/dataframes')

if (! dir.exists(RES_DIR)){
  dir.create(RES_DIR)
  }

if (! dir.exists(LQT_DIR)){
  dir.create(LQTT_DIR)
}

pat_ids = dir(WMH_DIR) 
pat_ids = pat_ids[!pat_ids %in% list("WMH_checklist.csv", "broken_h5_transform.csv")]

lesion_paths = list.files(MSK_DIR,
                          full.names = TRUE)

not_in_pat_ids = pat_ids[!pat_ids %in% lesion_paths]
parcel_path = system.file("extdata","Schaefer_Yeo_Plus_Subcort",
                          "200Parcels17Networks.nii.gz",package="LQT")

start_time <- Sys.time()

cfg = create_cfg_object(pat_ids=pat_ids,
                        lesion_paths=lesion_paths,
                        parcel_path=parcel_path,
                        out_path=LQT_DIR,
                        )

########### Create Damage and Disconnection Measures ###########
# Get parcel damage for patients
get_parcel_damage(cfg, cores=1)
# Get tract SDC for patients
get_tract_discon(cfg, cores=1)
# Get parcel SDC and SSPL measures for patients
get_parcel_cons(cfg, cores=1)


end_time <- Sys.time()
end_time - start_time

########### Build and View Summary Plots ###########
#plot_lqt_subject(cfg, pat_ids[1], "parcel.damage")
#plot_lqt_subject(cfg, pat_ids[1], "tract.discon")
#plot_lqt_subject(cfg, pat_ids[1], "parcel.discon")
#plot_lqt_subject(cfg, pat_ids[1], "parcel.sspl")

########### Compile Datasets for Analysis ###########
data = compile_data(cfg, cores = 1)
list2env(data, .GlobalEnv);

if (! dir.exists(LQT_DF_DIR)){
  dir.create(LQT_DF_DIR)
}

######### Save Analysis-ready Datasets #############
write.csv(net.discon, paste0(LQT_DF_DIR, "/net_discon.csv"))
write.csv(net2net.discon, paste0(LQT_DF_DIR, "/net2net_discon.csv"))
write.csv(parc.damage, paste0(LQT_DF_DIR, "/parc_damage.csv"))
write.csv(parc.discon, paste0(LQT_DF_DIR, "/parc_discon.csv"))
write.csv(parc2parc.discon, paste0(LQT_DF_DIR, "/parc2parc_discon.csv"))
write.csv(tract.discon, paste0(LQT_DF_DIR, "/tract_discon.csv"))


######## Load and save SC matrix as csv ###########
load(paste0(LQT_DIR, "/Atlas/atlas_Yeo.200.17_connectivity.RData"))
write.csv(connectivity, paste0(UTL_DIR, '/Schaefer200_sc.csv'))
