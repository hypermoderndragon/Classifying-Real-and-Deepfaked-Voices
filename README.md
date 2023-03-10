# Classifying Real and Deepfaked Presidential Voices Using Signal Processing and Machine Learning Techniques
## Problem Statement
Deepfake technology has recently become more accessible to normal people due to advances in AI. This has prompted concern amongst political journalists that the technology could be abused to create false videos, resulting in an increase in misinformation.

In this project, I use machine learning algorithms and signal processing techniques to differentiate between real and deepfaked presidential voices. This method can be used by journalists to ensure the veracity of the news they are reporting.

I use two signal processing methods to extract features from audio:
  - Mel Frequency Cepstral Coefficients
  - Linear Predictive Coding
  
I then run these features through three statistical or machine learning models:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  
## Data Collection
The real presidential voices were sourced from YouTube and the UVA Miller Center Presidential Speeches Archive. The deepfaked voices were self-created using ElevenLabs or stripped from videos created by other people.

# A) MFCC Approach
## Mel Frequency Cepstral Coefficients Theory
The Mel-frequency cepstrum is used in signal processing to convert the frequency distribution of a sound signal, mimicking the way humans hear. The coefficients that make up a MF cepstrum are commonly used for speech processing as they are scaled to the vocal and auditory systems of humans.
  - Each coefficient represents how much of a speaker's vocal output lies within a given frequency range, as measured at various timeframes.
  - The calculation of MFCCs contains many steps that are not practical to manually attempt under the time constraint of this project, so here the library Librosa is used for extraction.

## 1. Creating DataFrames for each speaker with extracted components for all audio samples
As source files, we have two sets of "speakers" - Donald Trump and Joe Biden. Each set consists of the human and their respective deepfaked clone. There are four "speakers" in total.
  - Four source folders are used, one for each speaker, containing unprocessed sample audio files.
  - Four 'split' folders are used, one for each speaker, which will contain the sample audio files split into five-second clips.
The audio files in each source folder are imported, split, and processed using the two below functions.
  - The function 'loadAndSplit' loads loads a soundfile from the source folder, splits it into five-second clips, and saves these audio clips to the 'split' folder for that speaker.
  - The function 'readSplitFiles' gets a list of filenames in the 'split' folder for a speaker, loads each soundfile, and extracts the MFCC components for that soundfile.
These processed MFCC lists are then turned into DataFrames representing voice information, one for each speaker. A new column 'real' is added to the DataFrame, with a numerical value indicating which speaker the voice belongs to. The four DataFrames are finally concatenated to create one DataFrame representing all audio information.

### Visualizing original and processed files
  - The first graph shows the time series of the original audio sample.
  - The second graph shows the audio sample represented as a spectrogram (graph of sound frequencies) after conversion to the Mel frequency cepstrum.
  - The third graph shows the MFC coefficients, which are used here for analysis.
![download](https://user-images.githubusercontent.com/103140702/224368987-0c3c95df-41e0-4a28-a56f-a7a1f978c34e.png)

## 2. Analyzing samples
### Logistic regression:  
  - ROC/AUC score: 0.9992
  ![download](https://user-images.githubusercontent.com/103140702/224369241-4f60a3ec-eccd-485b-bf6d-43ed64fdd085.png)
### K-nearest:
  - ROC/AUC score: 1.0
![download](https://user-images.githubusercontent.com/103140702/224369309-263dfa0f-60fb-475d-ab48-a3f4ad0ced75.png)
### Decision tree:
  - ROC/AUC score: 0.9835
  ![download](https://user-images.githubusercontent.com/103140702/224369419-8b9fac0a-9c23-4cc0-a4d3-056e7488c7c7.png)

# B) LPC (Linear Predictive Coding) Approach
## Linear Predictive Coding Theory
Linear predictive coding is a signal processing method designed for speech transmission and compression. It is used to numerically approximate the sound of the human voice - instead of attempting to transmit an entire sound signal, communication systems will transmit only the linear predictive coefficients.

Human speech is modeled as two components:
  - The 'source' or initial sound produced by the body, and
  - The 'filter' or change in this sound produced by resonance in the vocal tract.
  
The intention behind LPC is to model the human voice by separating the source components of human speech from the filter or resonance components. It can also calculate an 'error' component containing noise. It does this by using a technique called autoregression.
  - Autoregression is used to process signals in which there is a correlation between sample values of the signal and delayed values of the same signals. It can be used when a signal contains repeating patterns that contribute to its overall values, in this case, the resonance component of speech.
  
Below I use a technique called Levinson-Durbin recursion to return the linear predictive coefficients a for a given signal. These coefficients will be used to numerically represent the filter components of the human voice. The Levinson-Durbin recursion can also return a prediction error e and reflection/resonance coefficients k, but here these are not used.
## 1. Creating DataFrames for each speaker with extracted components for all samples
This process is essentially the same as above.
  - A function called 'levinson_durbin' is created to extract the LPC components from a sound sample.
  - The 'readSplitFiles' function is modified to extract the LPC components for each read five-second sample, instead of the MFCC components.  This function is renamed to 'readSplitFilesLPC'. 
  - These LPC components are appended as a list to a list representing all vocal information for that speaker across all samples.
  
### Visualizing original and processed files
  - The first graph shows the time series of the original audio sample.
  - The second graph shows the LPC coefficients of that time series, which are used here for analysis.
![download](https://user-images.githubusercontent.com/103140702/224369933-57671e3c-3265-4b2a-bb3f-55e7fc8f9a3f.png)

## 2. Analyzing samples
### Logistic regression:
  - ROC/AUC score: 0.9979
  ![download](https://user-images.githubusercontent.com/103140702/224370472-8489afc9-0464-4384-9361-4f1ba2e47221.png)
### K-nearest:
  - ROC/AUC score: 0.9828
  ![download](https://user-images.githubusercontent.com/103140702/224370595-a89c9ceb-a2d6-4ade-8703-72bd5efd7882.png)
### Decision tree:
  - ROC/AUC score: 0.9189
![download](https://user-images.githubusercontent.com/103140702/224371809-70230ba9-0688-40dd-8d67-fde4116c706c.png)

# Conclusion 
The MLCC approach worked better overall than the LPCC approach. This is likely due to the fact that MLCCs are a newer invention, presumably more sophisticated, and possibly better suited for this sort of analysis.

The highest performing model in the MFCC approach was K-nearest, followed by logistic regression and decision tree. The difference between logistic regression and K-nearest was small.

The highest performing model in the LPCC approach was logistic regression, followed by K-nearest and decision tree.

Decision tree performed significantly worse in both approaches.

# Potential Further Steps
  - The audio files sourced for this project contained low to moderate unwanted noise. If the audio of interest is especially noisy, further processing techniques could be applied before extracting the features.
  - All the source audio samples contained one speaker for the entire duration. It was difficult to find deepfaked audio that only had one speaker - most videos had short clips of multiple speakers, ex. Obama arguing with Trump, rapidly alternating and each voice segment only lasting a few seconds.
    - Splitting on silence instead of timeframe would likely allow analysis of these videos.
  - Combining this speaker recognition technique with a neural net for word recognition could yield a audio transcription algorithm.
