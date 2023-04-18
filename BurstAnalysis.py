# Check if packages are installed, install them if not
import pip
import sys
import subprocess
import os

try:
    import pyabf
    import matplotlib
    import numpy
    import pandas
    import scipy

    print("All packages found")
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyabf"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    print("Packages installed successfully")

# ignore runtime warnings (arise from cases where no bursts are detected in a channel)
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import packages
import pyabf
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
matplotlib.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import signal
from scipy import stats
import time
import scipy as sci
from itertools import combinations

matplotlib.rcParams["figure.autolayout"] = True

# Create an instance of toolbox
root = Tk()
start_time = time.time()

# Set the geometry of the toolbox window
root.title("Command window")
root.resizable(True, True)

# Function to open a file in the system, open new window for file manipulation, and save data for manipulation in other functions
def open_file():
    global filepath
    filepath = filedialog.askopenfilename(
        title="Open a .abf file",
        filetypes=((".abf files", "*.abf"), ("all files", "*.*")),
    )
    print(f"Opening: {filepath}")
    global abf
    abf = pyabf.ABF(filepath)
    if clamp_button.config("text")[-1] == "ON":
        abf.launchInClampFit()
    # abf.headerLaunch() #uncomment in order to see all abf details
    global ax
    fig, ax = plt.subplots(1, len(abf.channelList), figsize=(12.8, 6.8))
    fig.canvas.manager.set_window_title(abf.abfFilePath)
    sett = Toplevel(root)  # New window to hold burst detection settings
    sett.resizable(False, False)
    sett.title("Burst analysis settings window")
    sett.geometry("+50+400")
    global chanlab
    chanlab = {}
    global chan_dropdown
    chan_dropdown = {}
    global chanthresh
    chanthresh = {}
    chantitle = Label(sett, text="Channel")
    bursttitle = Label(sett, text="Burst detection")
    manualtitle = Label(sett, text="Threshold")
    sidetitle = Label(sett, text="Side")
    leveltitle = Label(sett, text="Level")
    chantitle.grid(row=2, column=0)
    bursttitle.grid(row=2, column=1)
    manualtitle.grid(row=2, column=2)
    sidetitle.grid(row=2, column=3)
    leveltitle.grid(row=2, column=4)
    writebutton = Button(sett, text="Write to .csv", width=12, command=write_to_csv)
    injurylab = Label(sett, text="Injury")
    global injuryvar
    injuryvar = StringVar()
    injury_options = ["Naive", "Sham", "SCI"]
    injury_dropdown = OptionMenu(sett, injuryvar, *injury_options)
    druglab = Label(sett, text="Drug condition")
    global drugvar
    drugvar = StringVar()
    # add any necessary drug options to the dropdown list below
    drugoptions = [
        "None",
        "1μM 4-AP",
        "2μM 4-AP",
        "3μM 4-AP",
        "5μM 4-AP",
        "10μM 4-AP",
        "20μM 4-AP",
        "50μM 4-AP",
        "100μM 4-AP",
        "VU0240551",
        "Bicuculline",
        "Picrotoxin",
        "Gabazine",
        "CNQX+APV",
        "Retigabine",
        "Bumetanide",
        "TEA",
        "L655,708",
        "THDOC",
        "Nipecotic acid",
        "Pregnanolone",
        "Methiothepin",
        "Zolmitriptan",
        "Carbenoxolone"
    ]
    drug_dropdown = OptionMenu(sett, drugvar, *drugoptions)
    # arrange buttons inside input window (opens upon exporting to csv)
    writebutton.grid(row=0, column=0, padx=4, pady=2)
    injurylab.grid(row=1, column=1, padx=4, pady=2)
    injury_dropdown.grid(row=1, column=2, padx=4, pady=2)
    druglab.grid(row=1, column=3, padx=4, pady=2)
    drug_dropdown.grid(row=1, column=4, padx=4, pady=2)
    levellab = {}
    # set these variables as global so they can be read into a dataframe for .csv output later
    global side_dropdown
    side_dropdown = {}
    global level
    level = {}
    global levelvars
    levelvars = []
    global sidevars
    sidevars = []
    global dropvars
    dropvars = []
    global stimside
    stimside = []
    global stimlevel
    stimlevel = []
    # The rest of the buttons and options are created dynamically based on the number of channels in the loaded .abf file
    for (
        ch
    ) in abf.channelList:  # Create buttons and entry boxes for channel threshold value
        chanlab[ch] = Label(sett, text=abf.adcNames[ch])
        dropvar = StringVar(value="3.5x RMS")
        dropvars.append(dropvar)
        chanoptions = ["3.5x RMS", "Manual", "5x Mean"]
        chan_dropdown[ch] = OptionMenu(sett, dropvar, *chanoptions)
        chanthresh[ch] = Entry(sett, width=12)
        chanlab[ch].grid(row=ch + 3, column=0, sticky=W, padx=4, pady=2)
        chan_dropdown[ch].grid(row=ch + 3, column=1, sticky=W, padx=4, pady=2)
        chanthresh[ch].grid(row=ch + 3, column=2, padx=4)
        sidevar = StringVar()
        sidevars.append(sidevar)
        levelvar = StringVar()
        levelvars.append(levelvar)
        # add levels here
        leveloptions = [
            "T6",
            "T7",
            "T8",
            "T9",
            "T10",
            "T11",
            "T12",
            "T13",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L6",
            "S1",
            "S2",
        ]
        level[ch] = OptionMenu(sett, levelvar, *leveloptions)
        sideoptions = ["Right", "Left"]
        side_dropdown[ch] = OptionMenu(sett, sidevar, *sideoptions)
        level[ch].grid(row=ch + 3, column=4, padx=4)
        side_dropdown[ch].grid(row=ch + 3, column=3, sticky=W, padx=4, pady=2)
    stimlab = Label(sett, text="Stimulus Level")
    stimlab.grid(row=len(abf.channelList) + 3, column=0, sticky=W, padx=4, pady=2)
    stimlevel = StringVar()
    stimlevelmenu = OptionMenu(sett, stimlevel, *leveloptions)
    stimside = StringVar()
    stimsidemenu = OptionMenu(sett, stimside, *sideoptions)
    stimlevelmenu.grid(row=len(abf.channelList) + 3, column=4, sticky=W, padx=4, pady=2)
    stimsidemenu.grid(row=len(abf.channelList) + 3, column=3, sticky=W, padx=4, pady=2)
    for ch in abf.channelList:  # Plot each channel of the file
        global filtered
        filtered = {}
        global sweepx
        sweepx = []
        if (
            abf.sweepCount == 1
        ):  # if data is gap-free, open all data; if it is fixed-length sweeps, concatenate all sweeps into single list
            abf.setSweep(sweepNumber=0, channel=abf.channelList[ch])
            sweepx = abf.sweepX
        else:
            for sweeps in abf.sweepList:
                abf.setSweep(sweepNumber=sweeps, channel=abf.channelList[ch])
                sweepx.extend(list(abf.sweepX + abf.sweepNumber * abf.sweepLengthSec))
        ax[ch].plot(sweepx, abf.data[ch], color="k")
        ax[ch].title.set_text(abf.adcNames[ch])
    # Loop over other axes and link to first axes
    parent = ax[0]
    for i in range(1, len(ax)):
        ax[i].sharex(parent)
        ax[i].sharey(parent)
    plt.show()


# Function to plot all bursts in breakout window
def plot_bursts():
    newplot, breakch = plt.subplots(len(abf.adcNames), 1, figsize=(3, 8))
    newplot.canvas.manager.set_window_title(abf.abfID + " Burst breakout window")
    for i in range(len(burstsave)):
        breakch[i].cla
        parent = breakch[0]
        for sels in burstsave[i][1:-1]:
            selsx = list(np.arange(min(sels) - 50, max(sels)))
            selsxlen = range(len(selsx))
            if breakouttype_button.config("text")[-1] == "Bursts":
                breakch[i].plot(selsxlen, filtered[i][selsx])
            else:
                breakch[i].plot(selsxlen, np.cumsum(abs(filtered[i][selsx])))
            breakch[i].title.set_text(
                abf.adcNames[i] + " : " + str(burstcountsave[i]) + " Bursts"
            )
            breakch[i].sharex(parent)
            breakch[i].sharey(parent)
    plt.show()


# Function to identify bursts and save locations and measurements
def find_bursts():
    global peaks
    peaks = {}
    global burstsave
    burstsave = {}
    global burstcountsave
    burstcountsave = {}
    # print(f"Channel names: {abf.adcNames}")
    # re-plot data in case changes were made during filtering
    for ch in abf.channelList:
        ax[ch].cla()
        ax[ch].title.set_text(abf.adcNames[ch])
        if abf.sweepCount == 1:
            abf.setSweep(sweepNumber=0, channel=abf.channelList[ch])
        else:
            abf.setSweep(sweepNumber=1, channel=abf.channelList[ch])
        if len(filtered):
            ax[ch].plot(sweepx, abs(filtered[ch]), color="b", alpha=0.5)
            if raw_button.config("text")[-1] == "ON":
                ax[ch].plot(sweepx, abf.data[ch], color="k", alpha=0.5)
    plt.show()
    ML = {}
    {"ch1": {"peaks": [], "peak_diffs": []}}
    burstshold = {}
    burstlocs = {}
    bursts = {}
    global store  # this will be used to hold data for output to csv
    store = []
    peakidx = {}
    peak_diffs = {}
    peaklist = {}
    burstdiffs = {}
    amplist = {}
    bursts = {}
    peaks = {}
    global reduced
    reduced = {}
    # for each channel, identify and measure bursts
    for i, ch in enumerate(abf.channelList):
        axis = i
        datastore = [0] * len(sweepx)
        if dropvars[i].get() == "Manual":
            peakheight = float(chanthresh[i].get())
        elif dropvars[i].get() == "5x Mean":
            peakheight = 5 * np.mean(filtered[i])
        else:
            peakheight = 3.5 * np.sqrt(np.mean(filtered[i] ** 2))
        peaks, properties = find_peaks(abs(filtered[i]), [peakheight, 300])
        peak_diffs = np.diff(peaks)
        leaders_to_idx = []
        mark_og_idx = []
        mark_amp = []
        if np.std(peak_diffs) / np.mean(peak_diffs) > 1:
            for idx, peak_diff in enumerate(
                peak_diffs
            ):  # for each of the peak differences, keep only the indeces from peaks that are closer together than 20 ms
                if peak_diff < 20 * (abf.dataRate / 1000):
                    leaders_to_idx.append((peak_diff, idx))
                    mark_og_idx.append(peaks[idx])
                    mark_amp.append(properties["peak_heights"][idx])
        m = []
        b_matches = []
        if mark_og_idx:
            first_item = mark_og_idx[0]
            for item in mark_og_idx[
                1:
            ]:  # break spike data from each burst into its own array
                diff = abs(item - first_item)
                if diff < 50 * (
                    abf.dataRate / 1000
                ):  # this value sets the maximum distance between consecutive spikes within a burst (best to use same value as initial ISI search)
                    m.append(first_item)
                else:
                    if m:
                        b_matches.append(m)
                    m = []
                first_item = item
        for t, j in enumerate(mark_og_idx):
            datastore[j] = mark_amp[t]
        bursts[i] = b_matches
        burstsonly = [0] * len(
            sweepx
        )  # vector of zeros that burst lcoations will be added to
        burstspikes = []
        burststart = []
        burstend = []
        burstlength = []
        burstrectint = []
        burstrawrectint = []
        if abf.sweepCount == 1:
            abf.setSweep(sweepNumber=0, channel=i)
        else:
            abf.setSweep(sweepNumber=1, channel=i)
        trace = abs(filtered[i])
        burstcounter = []
        bursthold = []
        for t, j in enumerate(bursts[i]):
            b_matchidx = []
            if len(bursts[i][t]) > int(
                peakthresh_entry.get()
            ):  # restrict bursts to groups of 7 or more spikes, generate burst measurement values, and save to a dataframe
                for n, inds in enumerate(j):
                    burstsonly[inds] = datastore[inds]
                selsx = list(np.arange(min(j) - 50, max(j) + 10))
                bursthold.append(selsx)
                burstday = abf.abfID
                burstchannel = abf.adcNames[i]
                burstspikes = len(bursts[i][t])  # number of spikes per bursts
                burststart = (int(bursts[i][t][0]) - 10) / abf.dataRate
                burststartidx = bursts[i][t][0] - 10
                burstend = (int(bursts[i][t][-1]) + 50) / abf.dataRate
                burstendidx = bursts[i][t][-1] + 50
                burstlength = burstend - burststart
                subnoise = sum(
                    abs(
                        trace[
                            burststartidx
                            - (burstendidx - burststartidx - 100) : burststartidx
                            - 100
                        ]
                    )
                )
                rawnoise = sum(
                    abs(
                        abf.data[i][
                            burststartidx
                            - (burstendidx - burststartidx - 100) : burststartidx
                            - 100
                        ]
                    )
                )
                burstrectint = sum(abs(trace[burststartidx:burstendidx])) - subnoise
                burstrawrectint = (
                    sum(abs(abf.data[i][burststartidx:burstendidx])) - rawnoise
                )
                for spikes in range(len(mark_og_idx)):
                    b_matchidx.append(int(mark_og_idx[spikes]))
                peaks_for_mean, properties_for_mean = find_peaks(
                    burstsonly[burststartidx:burstendidx], height=peakheight
                )
                try: 
                    burstmeanamp = sum(properties_for_mean["peak_heights"]) / len(properties_for_mean["peak_heights"])
                except ZeroDivisionError: 
                    burstmeanamp = 0
                else: 
                    burstmeanamp = sum(properties_for_mean["peak_heights"]) / len(properties_for_mean["peak_heights"]) 
                burstcounter.append(1)
                if marker_button.config("text")[-1] == "ON":
                    ax[i].axvline(
                        x=burststart, color="g", alpha=0.5
                    )  # green line to mark beginning of burst as defined by ISI algorithm
                    ax[i].axvline(
                        x=burstend, color="r", alpha=0.5
                    )  # red line to mark end of burst as defined by ISI algorithm
                    ax[i].title.set_text(abf.adcNames[i])
                recside = sidevars[axis].get()
                reclevel = levelvars[axis].get()
                injury = injuryvar.get()
                drug = drugvar.get()
                if (
                    "stimtimes" in globals()
                ):  # if file includes a stimulus, add stimulus time information
                    stimdiff = []
                    for stims in stimtimes:
                        stimdiff.append(burststartidx - stims)
                    stimdiff = [i for i in stimdiff if i > 0]
                    if stimdiff:
                        timefromstim = min(stimdiff)
                        stimnum = stimtimes.index(burststartidx - timefromstim)
                        stimindex = stimtimes[stimnum]
                        stimtime = stimtimes[stimnum] / abf.dataRate
                        stimlr = stimside.get()
                        stimlev = stimlevel.get()
                    else:
                        timefromstim = 0
                        stimnum = 0
                        stimindex = 0
                        stimtime = 0
                        stimlr = 0
                        stimlev = 0
                else:
                    stimdiff = 0
                    timefromstim = 0
                    stimnum = 0
                    stimindex = 0
                    stimtime = 0
                    stimlr = 0
                    stimlev = 0
                if bursthold:
                    burstsave[i] = bursthold
                    burstcountsave[i] = len(bursthold)
            else:
                burstday = abf.abfID
                burstchannel = abf.adcNames[i]
                burstspikes = 0
                burststart = 0
                burststartidx = 0
                burstend = 0
                burstendidx = 0
                burstlength = 0
                subnoise = 0
                rawnoise = 0
                burstrectint = 0
                burstrawrectint = 0
                for spikes in range(len(mark_og_idx)):
                    b_matchidx.append(0)
                burstpeakamps = 0
                burstmeanamp = 0
                burstfreq = 0
                reduced[i] = 0
                timefromstim = 0
                stimnum = 0
                stimindex = 0
                stimtime = 0
                stimlr = 0
                stimlev = 0
                recside = sidevars[axis].get()
                reclevel = levelvars[axis].get()
                injury = 0
                drug = 0
            store.append(
                [
                    burstday,
                    burstchannel,
                    burstspikes,
                    burststart,
                    burststartidx,
                    burstend,
                    burstendidx,
                    burstlength,
                    burstrectint,
                    burstrawrectint,
                    burstmeanamp,
                    len(burstcounter) / abf.dataLengthSec,
                    timefromstim / abf.dataRate,
                    stimnum,
                    stimindex,
                    stimtime,
                    stimlr,
                    stimlev,
                    recside,
                    reclevel,
                    injury,
                    drug,
                ]
            )  # add all measurement values to storage list
        reduced[i] = burstsonly
        # if new variables are added above, be sure to include a new title in the correct order here
        namelist = [
            "Burst_Day",
            "Burst_Channel",
            "Burst_Spikes",
            "Burst_Start",
            "Burst_Start_Idx",
            "Burst_End",
            "Burst_End_Idx",
            "Burst_Length",
            "Burst_Rect_Int",
            "Burst_Raw_Rect_Int",
            "Burst_Mean_Amp",
            "Burst_Frequency_Cumulative",
            "Time_From_Stim",
            "Stim_Number",
            "Stim_Index",
            "Stim_Time",
            "Stim_Side",
            "Stim_Level",
            "Rec_Side",
            "Rec_Level",
            "Injury",
            "Drug",
        ]  # column titles; ensure that abf.adcNames match order of measurements in for loop above
    global df
    df = pd.DataFrame(store, columns=namelist)  # save values to dataframe
    # print(f"Burst dataframe: {df}")


# function to write burst measurements and cross correlation data to .csv files
def write_to_csv():
    dir1 = askdirectory(
        title="Select folder to write to"
    )  # shows dialog box and return the path
    currdir = dir1
    if "Python_Data" in currdir:
        currpath = currdir
    else:
        if os.path.exists(currdir + "\\" + "Python_Data") == False:
            os.mkdir(currdir + "\\" + "Python_Data")
        currpath = currdir + "\\" + "Python_Data" + "\\"
    df.to_csv(currpath + abf.abfID + "_Burst_Quant.csv", index=False)
    df2.to_csv(currpath + abf.abfID + "_Cross_Correlation.csv", index=False)
    print(f".csv files written to {currpath}")


# function to perform cross correlation on all possible combinations of channels
def perform_cross_correlation():
    from itertools import combinations

    store2 = []  # will be used to hold cross correlation output information
    perm = combinations(
        range(len(reduced)), 2
    )  # determine possible channel combinations
    corrlist = {}
    localmax = []
    corr = {}
    combo = []
    for ind, i in enumerate(perm):
        ch1 = i[0]
        ch2 = i[1]
        sig1 = reduced[ch1]
        sig2 = reduced[ch2]
        corr = sci.signal.correlate(sig1, sig2)  # perform cross correlation
        lags = sci.signal.correlation_lags(
            len(sweepx), len(sweepx)
        )  # generate corresponding lags
        lags = lags / abf.dataRate
        lagstart = np.where(lags == -0.2)[0][
            0
        ]  # restricts lags to 200 ms window around burst
        lagend = np.where(lags == 0.2)[0][0]
        lags = lags[lagstart:lagend]
        corr = corr[lagstart:lagend]
        corrlist[ind] = corr
        combo.append(i)
        localmax.append(max(corr))
    fig2, ax2 = plt.subplots(1, len(combo))
    fig2.canvas.manager.set_window_title("Cross correlation plots")
    store2 = []
    for ind, corrs in enumerate(corrlist):
        interm = []
        corrfind = corrlist[ind] / max(localmax)
        ax2[ind].plot(lags, corrfind)
        legendtext = [combo[ind][0] + 1, combo[ind][1] + 1]
        ax2[ind].legend([legendtext], loc="upper center")
        ax2[ind].set_xlim(-0.2, 0.2)
        ax2[ind].set_ylim(0, 1)
        corrpeaks, _ = find_peaks(corrfind, height=0.2, distance=5)
        toplag = []
        topcorr = []
        if len(corrpeaks):
            burstday = abf.abfID
            chan1 = combo[ind][0] + 1
            chan2 = combo[ind][1] + 1
            highlags = lags[corrpeaks]
            highcorr = corrfind[corrpeaks]
            lagsout = []
            corrout = []
            for i in range(5):
                placement = np.where(highcorr == max(highcorr))[0][0]
                lagsout.append(highlags[placement])
                corrout.append(highcorr[placement])
                highcorr[highcorr == max(highcorr)] = 0
            interm.append(burstday)
            interm.append(chan1)
            interm.append(chan2)
            interm.extend(lagsout)
            interm.extend(corrout)
            store2.append(interm)
  #          ax2[ind].plot(lags[corrpeaks], corrfind[corrpeaks], "r*")
    # like above, be sure to add variable names here if more measurements are taken
    namelist2 = [
        "Burst_Day",
        "Channel_1",
        "Channel_2",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_4",
        "Lag_5",
        "Corr_1",
        "Corr_2",
        "Corr_3",
        "Corr_4",
        "Corr_5",
    ]
    global df2
    plt.show()
    df2 = pd.DataFrame(store2, columns=namelist2)  # save values to dataframe
    # print(f"Correlation dataframe: {df2}")


# function to remove electrical or optical stimulus artifacts and store locations
def remove_artifact():
    inter = {}
    rounded = {}
    for ch in abf.channelList:
        ax[ch].cla()
        sweepx = []
        if (
            abf.sweepCount == 1
        ):  # if data is gap-free, open all data; if it is fixed-length sweeps, concatenate all sweeps into single list
            abf.setSweep(sweepNumber=0, channel=abf.channelList[ch])
            sweepx = abf.sweepX
        else:  # append each consecutive sweep to the end of the previous sweep
            stimpeaks = {}
            vals = np.array(abs(abf.data[ch]))
            slope = np.diff(vals)
            for sweeps in abf.sweepList:
                abf.setSweep(sweepNumber=sweeps, channel=abf.channelList[ch])
                if "opto" in abf.adcNames:
                    stimpeaks[sweeps], _ = find_peaks(
                        abf.data[abf.adcNames.index("opto")],
                        prominence=1,
                        distance=10000,
                    )
                else:
                    stimpeaks[sweeps], _ = find_peaks(
                        slope, height=max(slope) / 15, distance=1000
                    )
                sweepx.extend(list(abf.sweepX + abf.sweepNumber * abf.sweepLengthSec))
                stimset = set(stimpeaks[sweeps].flatten())
            inter[ch] = set(stimset).intersection(stimset)
            rounded[ch] = []
            for i, val in enumerate(inter[ch]):
                rounded[ch].append(round(val / 10) * 10)
    testlist = []
    for i in range(len(inter) - 1):
        testlist.append(rounded[i] == rounded[i + 1])
    global stimtimes
    stimtimes = []
    for ch in abf.channelList:
        if (
            len(inter[ch]) == 3 * abf.sweepCount
            or len(inter[ch]) == 2 * abf.sweepCount
            or all(testlist) == True
        ):
            for i, val in enumerate(inter[ch]):
                stimtimes.append(val)
    if stimtimes:
        for ch in abf.channelList:
            for i, val in enumerate(stimtimes):
                startidx = val - 30  # value based on stimulus artifact waveform
                endidx = (
                    val + 100
                )  # value based on polarization that sometimes appears after stimulation (filtering tends to introduce more artifact)
                for j in range(startidx, endidx):
                    filtered[ch][j] = 0
    for ch in abf.channelList:
        ax[ch].plot(sweepx, abs(filtered[ch]), color="b", alpha=0.5)
        ax[ch].title.set_text(abf.adcNames[ch])
        if raw_button.config("text")[-1] == "ON":
            ax[ch].plot(sweepx, abf.data[ch], color="k", alpha=0.5)
            ax[ch].title.set_text(abf.adcNames[ch])
        for i in stimtimes:
            ax[ch].plot(sweepx[i], 0, "r*")
        chanthresh[ch].delete(0, END)
        chanthresh[ch].insert(END, 3.5 * np.sqrt(np.mean(filtered[ch] ** 2)))
    plt.show()


# Function to filter data based on parameters from command window
# filtfilt is used in order to cancel out phase shift
def filter_data():
    for i in abf.channelList:
        ax[i].cla()
        filt = []
        if raw_button.config("text")[-1] == "ON":
            ax[i].plot(sweepx, abf.data[i], color="k", alpha=0.5)
            ax[i].title.set_text(abf.adcNames[i])
        if abf.sweepCount == 1:
            abf.setSweep(sweepNumber=0, channel=abf.channelList[i])
        else:
            abf.setSweep(sweepNumber=1, channel=abf.channelList[i])
        if (
            high_button.config("text")[-1] == "ON"
            and low_button.config("text")[-1] == "OFF"
        ):
            filt = 1
            sos = signal.butter(
                4, int(highp.get()), btype="highpass", fs=abf.dataRate, output="sos"
            )
            filtered[i] = signal.sosfiltfilt(sos, abf.data[i])
            if median_button.config("text")[-1] == "OFF":
                ax[i].plot(sweepx, abs(filtered[i]), color="b", alpha=0.5)
                ax[i].title.set_text(abf.adcNames[i])
        elif (
            high_button.config("text")[-1] == "ON"
            and low_button.config("text")[-1] == "ON"
        ):
            filt = 1
            sos = signal.butter(
                4,
                [int(lowp.get()), int(highp.get())],
                btype="bandpass",
                fs=abf.dataRate,
                output="sos",
            )
            filtered[i] = signal.sosfiltfilt(sos, abf.data[i])
            if median_button.config("text")[-1] == "OFF":
                ax[i].plot(sweepx, abs(filtered[i]), color="b", alpha=0.5)
                ax[i].title.set_text(abf.adcNames[i])
        elif (
            high_button.config("text")[-1] == "OFF"
            and low_button.config("text")[-1] == "ON"
        ):
            filt = 1
            sos = signal.butter(
                4, int(lowp.get()), btype="lowpass", fs=abf.dataRate, output="sos"
            )
            filtered[i] = signal.sosfiltfilt(sos, abf.data[i])
            if median_button.config("text")[-1] == "OFF":
                ax[i].plot(sweepx, abs(filtered[i]), color="b", alpha=0.5)
                ax[i].title.set_text(abf.adcNames[i])
        if median_button.config("text")[-1] == "ON":
            if filt == 1:
                filtered[i] = signal.medfilt(filtered[i])
            else:
                filtered[i] = signal.medfilt(abf.data[i])
            ax[i].plot(sweepx, abs(filtered[i]), color="b", alpha=0.5)
            ax[i].title.set_text(abf.adcNames[i])
        chanthresh[i].delete(0, END)
        chanthresh[i].insert(END, 3.5 * np.sqrt(np.mean(filtered[i] ** 2)))
    plt.show()


# Define simple toggle functions for button switches
def Lowtoggle():
    if low_button.config("text")[-1] == "ON":
        low_button.config(text="OFF")
    else:
        low_button.config(text="ON")


def Hightoggle():
    if high_button.config("text")[-1] == "ON":
        high_button.config(text="OFF")
    else:
        high_button.config(text="ON")


def Mediantoggle():
    if median_button.config("text")[-1] == "ON":
        median_button.config(text="OFF")
    else:
        median_button.config(text="ON")


def Rawtoggle():
    if raw_button.config("text")[-1] == "ON":
        raw_button.config(text="OFF")
    else:
        raw_button.config(text="ON")


def Clamptoggle():
    if clamp_button.config("text")[-1] == "ON":
        clamp_button.config(text="OFF")
    else:
        clamp_button.config(text="ON")


def Markertoggle():
    if marker_button.config("text")[-1] == "ON":
        marker_button.config(text="OFF")
    else:
        marker_button.config(text="ON")


def Breakouttoggle():
    if breakouttype_button.config("text")[-1] == "Bursts":
        breakouttype_button.config(text="Integral")
    else:
        breakouttype_button.config(text="Bursts")


# Create buttons for main window
loadbutton = Button(root, text="Open .abf file", width=12, command=open_file)
filterbutton = Button(root, text="Filter channels", width=12, command=filter_data)
burstbutton = Button(root, text="Identify bursts", width=12, command=find_bursts)
artifactbutton = Button(
    root, text="Find stimulation", width=12, command=remove_artifact
)
crossbutton = Button(
    root, text="Cross correlation", width=12, command=perform_cross_correlation
)
lowlab = Label(root, text="Low pass filter (Hz)")
low_button = Button(text="OFF", width=10, command=Lowtoggle)
lowp = Entry(root, width=5)
highlab = Label(root, text="High pass filter (Hz)")
high_button = Button(text="ON", width=10, command=Hightoggle)
highp = Entry(root, width=5)
highp.insert(END, "10")
medianlab = Label(root, text="Median filter toggle")
median_button = Button(text="ON", width=10, command=Mediantoggle)
rawlab = Label(root, text="Show raw trace")
raw_button = Button(text="OFF", width=10, command=Rawtoggle)
markerlab = Label(root, text="Display burst markers")
marker_button = Button(text="ON", width=10, command=Markertoggle)
clamplab = Label(root, text="Open in Clampfit")
clamp_button = Button(text="OFF", width=10, command=Clamptoggle)
breakouttype_button = Button(text="Bursts", width=10, command=Breakouttoggle)
breakoutlab = Label(root, text="Open in breakout window")
breakout_button = Button(text="Open", width=10, command=plot_bursts)
peakthreshlab = Label(root, text="Minimum peaks for burst")
peakthresh_entry = Entry(root, width=2)
peakthresh_entry.insert(END, "5")

# Arrange items inside command window
loadbutton.grid(row=1, column=0, padx=4, pady=2)
filterbutton.grid(row=0, column=1, padx=4, pady=2)
burstbutton.grid(row=1, column=1, padx=4, pady=2)
artifactbutton.grid(row=0, column=2, padx=4, pady=2)
crossbutton.grid(row=1, column=2, padx=4, pady=2)
lowlab.grid(row=4, column=0, sticky=W, padx=4, pady=2)
highlab.grid(row=5, column=0, sticky=W, padx=4, pady=2)
low_button.grid(row=4, column=1, sticky=W, padx=4, pady=2)
high_button.grid(row=5, column=1, sticky=W, padx=4, pady=2)
lowp.grid(row=4, column=2, sticky=W, padx=4, pady=2)
highp.grid(row=5, column=2, sticky=W, padx=4, pady=2)
medianlab.grid(row=6, column=0, sticky=W, padx=4, pady=2)
median_button.grid(row=6, column=1, sticky=W, padx=4, pady=2)
rawlab.grid(row=7, column=0, sticky=W, padx=4, pady=2)
raw_button.grid(row=7, column=1, sticky=W, padx=4, pady=2)
markerlab.grid(row=8, column=0, sticky=W, padx=4, pady=2)
marker_button.grid(row=8, column=1, sticky=W, padx=4, pady=2)
clamplab.grid(row=9, column=0, sticky=W, padx=4, pady=2)
clamp_button.grid(row=9, column=1, sticky=W, padx=4, pady=2)
breakoutlab.grid(row=10, column=0, sticky=W, padx=4, pady=2)
breakout_button.grid(row=10, column=1, sticky=W, padx=4, pady=2)
breakouttype_button.grid(row=10, column=2, sticky=W, padx=4, pady=2)
peakthreshlab.grid(row=11, column=0, sticky=W, padx=4, pady=2)
peakthresh_entry.grid(row=11, column=1, sticky=W, padx=4, pady=2)

root.mainloop()
