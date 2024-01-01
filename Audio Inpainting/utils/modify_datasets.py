import os
import wave

# List of dataset directories
datasets = ['testMusic16kHz', 'testSpeech8kHz_from16kHz', 'testSpeech16kHz']

for dataset in datasets:
    # Create a new directory for the modified sounds
    new_dir = os.path.join('Data', dataset + '_modified')
    os.makedirs(new_dir, exist_ok=True)

    # Get a list of all .wav files in the dataset directory
    wav_files = [f for f in os.listdir(os.path.join('Data', dataset)) if f.endswith('.wav')]

    for wav_file in wav_files:
    # Open the .wav file
        with wave.open(os.path.join('Data', dataset, wav_file), 'rb') as wf:
            # Get the number of frames in the file
            n_frames = wf.getnframes()

            # Check if the audio file is long enough
            if n_frames >= 80000:  # Less than 3 seconds of audio
                start_frame = 16000  # Start at the 1-second mark
                end_frame = start_frame + 32000  # End 3 seconds later
            else:
                start_frame = 8000
                end_frame = start_frame + 16000

            # Calculate the start and end frame for the 3 seconds of audio starting from 1 second
            
            # Set the position to the start frame and read the frames
            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)

        # Write the frames to a new file in the new directory
        with wave.open(os.path.join(new_dir, wav_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(frames)
