import mne

edf_path = "../Hiroki/Data/Data/sub-hupHUP060/ses-01/ieeg/sub-hupHUP060_ses-01_task-ictal_run-01_ieeg.edf"

raw = mne.io.read_raw_edf(edf_path, preload=False)

print(raw)
print(raw.info)
print("Channels:", len(raw.ch_names))
print("Sampling Rate:", raw.info['sfreq'])