from dotenv import load_dotenv
import os

# Access environmental variables
load_dotenv(verbose=True)

input_path = os.environ.get('pathname')
output_path = os.environ.get('dataname')
YEARS = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
passed = True

for year in YEARS:
    try:
        year = str(year)
        files = os.listdir(os.path.join(input_path, year))
        NOTE_GRAPHS_PATH = os.path.join(output_path, 'note_graphs', year) + "/"
        NOTES_GENERATED_PATH = os.path.join(
            output_path, 'notes_generated', year) + "/"
        SPECTROGRAM_GENERATED_PATH = os.path.join(
            output_path, 'spectrograms_generated', year) + "/"
        SPECTROGRAM_REAL_PATH = os.path.join(
            output_path, 'spectrograms_real', year) + "/"
        GENERATED_AUDIO_PATH = os.path.join(
            output_path, 'generated_audio', year) + "/"
        assert(len(files)%2 == 0)
        print(f"Year:{year} Expected:{len(files) // 2}")
        print(f"generated_audio:{len(os.listdir(GENERATED_AUDIO_PATH))} "+\
            f"note_graphs:{len(os.listdir(NOTE_GRAPHS_PATH))} " +\
            f"notes_generated:{len(os.listdir((NOTES_GENERATED_PATH)))} "+\
            f"spectrograms_generated:{len(os.listdir((SPECTROGRAM_GENERATED_PATH)))} "+\
            f"spectrograms_real:{len(os.listdir((SPECTROGRAM_REAL_PATH)))} "
        )

        passed = passed and len(files)//2                               ==\
                            len(os.listdir(GENERATED_AUDIO_PATH))       ==\
                            len(os.listdir(NOTE_GRAPHS_PATH))           ==\
                            len(os.listdir(NOTES_GENERATED_PATH))       ==\
                            len(os.listdir(SPECTROGRAM_GENERATED_PATH)) ==\
                            len(os.listdir(SPECTROGRAM_REAL_PATH))

        # check if spectrogram real and generated match
        print("--------------------------")
        print("Checking generated spectrograms...")
        for spec in os.listdir(SPECTROGRAM_REAL_PATH):
            if spec not in os.listdir(SPECTROGRAM_GENERATED_PATH):
                print(f"Spectrogram {spec} not found in generated spectrograms")
                passed = False
        
        # check if spectrogram real and note graph match
        print("--------------------------")
        print("Checking note graphs...")
        for spec in os.listdir(SPECTROGRAM_REAL_PATH):
            if spec not in os.listdir(NOTE_GRAPHS_PATH):
                print(f"Spectrogram {spec} not found in note graphs")
                passed = False

    except FileNotFoundError:
        print(f"is missing")
        passed = False

print("--------------------------")
print("Passed!" if passed else "Failed!")