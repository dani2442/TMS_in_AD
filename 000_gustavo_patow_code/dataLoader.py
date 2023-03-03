# =====================================================================================
# Methods to input AD data
# =====================================================================================
import numpy as np
import os, csv


base_folder = "../../Data_Raw/from_Ritter"


def characterizeConnectivityMatrix(C):
    return np.max(C), np.min(C), np.average(C), np.std(C), np.max(np.sum(C, axis=0)), np.average(np.sum(C, axis=0))


def checkClassifications(subjects):
    # ============================================================================
    # This code is to check whether we have the information of the type of subject
    # They can be one of:
    # Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer Disease (AD) or Significant Memory Concern (SMC)
    # ============================================================================
    input_classification = csv.reader(open(base_folder+"/subjects.csv", 'r'))
    classification = dict((rows[0], rows[1]) for rows in input_classification)
    mistery = []
    for pos, subject in enumerate(subjects):
        if subject in classification:
            print('{}: Subject {} classified as {}'.format(pos, subject, classification[subject]))
        else:
            print('{}: Subject {} NOT classified'.format(pos, subject))
            mistery.append(subject)
    print("Misisng {} subjects:".format(len(mistery)), mistery)
    print()
    return classification


def getClassifications():
    # ============================================================================
    # This code is to check whether we have the information of the type of subject
    # They can be one of:
    # Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer Disease (AD) or Significant Memory Concern (SMC)
    # ============================================================================
    input_classification = csv.reader(open(base_folder+"/subjects.csv", 'r'))
    classification = dict(filter(None, input_classification))
    return classification


# =====================================================================================
# Methods to input AD data
# =====================================================================================

# ===================== compute the Avg SC matrix over the HC sbjects
def computeAvgSC_HC_Matrix(classification, baseFolder):
    HC = [subject for subject in classification.keys() if classification[subject] == 'HC']
    print("SC + HC: {} (0)".format(HC[0]))
    sc_folder = baseFolder+'/'+HC[0]+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")

    sumMatrix = SC
    for subject in HC[1:]:
        print("SC + HC: {}".format(subject))
        sc_folder = baseFolder+'/'+subject+"/DWI_processing"
        SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
        sumMatrix += SC
    return sumMatrix / len(HC)  # but we normalize it afterwards, so we probably do not need this...


# ===================== Load one specific subject data
def loadSubjectData(subject, correcSCMatrix=True):
    def loadBurden(modality):
        pet_path = base_folder+"/PET_loads/"+subject+"/PET_PVC_MG/" + modality
        RH_pet = np.loadtxt(pet_path+"/"+"L."+modality+"_load_MSMAll.pscalar.txt")
        LH_pet = np.loadtxt(pet_path+"/"+"R."+modality+"_load_MSMAll.pscalar.txt")
        subcort_pet = np.loadtxt(pet_path+"/"+modality+"_load.subcortical.txt")[-19:]
        all_pet = np.concatenate((LH_pet,RH_pet,subcort_pet))
        return all_pet/np.max(all_pet)

    sc_folder = base_folder + '/connectomes/'+subject+"/DWI_processing"
    SC = np.loadtxt(sc_folder + "/connectome_weights.csv")
    if correcSCMatrix:
        SCnorm = correctSC(SC)
    else:
        SCnorm = np.log(SC + 1)

    abeta_burden = loadBurden("Amyloid")
    tau_burden = loadBurden("Tau")

    fMRI_path = base_folder+"/fMRI/"+subject+"/MNINonLinear/Results/Restingstate"
    series = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt")
    subcSeries = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt")
    fullSeries = np.concatenate((series,subcSeries))

    return SCnorm, abeta_burden, tau_burden, fullSeries


# ===================== Load all fMRI data
def load_all_HC_fMRI(classification, baseFolder):
    HC = [subject for subject in classification.keys() if classification[subject] == 'HC']
    all_fMRI = {}
    for subject in HC:
        print("fMRI HC: {}".format(subject))
        fMRI_path = baseFolder + "/fMRI/" + subject + "/MNINonLinear/Results/Restingstate"
        series = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt")
        subcSeries = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt")
        fullSeries = np.concatenate((series,subcSeries))
        all_fMRI[subject] = fullSeries
    return all_fMRI


# ===================== Normalize a SC matrix
normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
maxNodeInput66 = 0.7275543904602363
def correctSC(SC):
    N = SC.shape[0]
    logMatrix = np.log(SC+1)
    # areasSC = logMatrix.shape[0]
    # avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum, as in Gus' codes
    # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
    return finalMatrix


def analyzeMatrix(name, C):
    max, min, avg, std, maxNodeInput, avgNodeInput = characterizeConnectivityMatrix(C)
    print(name + " => Shape:{}, Max:{}, Min:{}, Avg:{}, Std:{}".format(C.shape, max, min, avg, std), end='')
    print("  => impact=Avg*#:{}".format(avg*C.shape[0]), end='')
    print("  => maxNodeInputs:{}".format(maxNodeInput), end='')
    print("  => avgNodeInputs:{}".format(avgNodeInput))


# This is used to avoid "infinite" computations for some cases (i.e., subjects) that have fMRI
# data that is way longer than any other subject, causing almost impossible computations to perform,
# because they last several weeks (~4 to 6), which seems impossible to complete with modern Windows SO,
# which restarts the computer whenever it want to perform supposedly "urgent" updates...
force_Tmax = True
limit_forcedTmax = 200


# This method is tu perform the timeSeries cutting when excessively long...
def cutTimeSeriesIfNeeded(timeseries):
    if force_Tmax and timeseries.shape[1] > limit_forcedTmax:
        print(f"cutting lengthy timeseries: {timeseries.shape[1]} to {limit_forcedTmax}")
        timeseries = timeseries[:,0:limit_forcedTmax]
    return timeseries


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
