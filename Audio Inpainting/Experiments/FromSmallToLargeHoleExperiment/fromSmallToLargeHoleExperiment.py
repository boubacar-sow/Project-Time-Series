from Problems.generateMissingGroupsProblem import generateMissingGroupsProblemPeriodic
from utils.evaluation.SNRInpaintingPerformance import SNRInpaintingPerformance
from typing import List, Dict, Any, Optional, Tuple
from scipy.io import wavfile as wav
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from Solvers.inpaintFrame_OMP_Gabor import inpaintFrame_OMP_Gabor
from Solvers.inpaintFrame_janssenInterpolation import inpaintFrame_janssenInterpolation
from utils.dictionaries.Gabor_dictionary import Gabor_Dictionary
from utils.dictionaries.DCT_Dictionary import DCT_Dictionary
from utils.wRect import wRect
from utils.wSine import wSine
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import read
import librosa

from Solvers.inpaintSignal_IndependentProcessingOfFrames import inpaintSignal_IndependentProcessingOfFrames
from Problems.generateDeclippingProblem import generateDeclippingProblem
from utils.evaluation.SNRInpaintingPerformance import SNRInpaintingPerformance
from utils.dictionaries.Gabor_dictionary import Gabor_Dictionary
from utils.dictionaries.DCT_Dictionary import DCT_Dictionary
from Solvers.inpaintFrame_janssenInterpolation import inpaintFrame_janssenInterpolation
from Solvers.inpaintFrame_OMP import inpaintFrame_OMP
from Solvers.inpaintFrame_OMP_Gabor import inpaintFrame_OMP_Gabor
from utils.wSine import wSine
from utils.wRect import wRect
from Solvers.inpaintFrame_OMP_Gabor import inpaintFrame_OMP_Gabor
from Solvers.inpaintFrame_consOMP import inpaintFrame_consOMP
from Solvers.inpaintFrame_consOMP_Gabor import inpaintFrame_consOMP_Gabor


def periodicHoleInpainting(exp_param=None):
    if exp_param is None:
        exp_param = {}  
    
    if 'soundDir' not in exp_param:
        exp_param['soundDir'] = 'Data/testSpeech8kHz_from16kHz/'
        exp_param['soundDir'] = 'Data/shortTest/'
        warnings.warn('soundDir has only one sound to have faster computations. Recommended soundDir: ../../Data/testSpeech8kHz_from16kHz/')
    if 'destDir' not in exp_param:
        exp_param['destDir'] = 'tmp/declip/'
    
    if 'interval_duration' not in exp_param:
        exp_param['interval_duration'] = 100
    if 'missing_duration' not in exp_param:
        exp_param['missing_duration'] = np.arange(1, 10, 1)
    
    # Set parameters
    if 'solvers' not in exp_param:
        warnings.warn('Frame length=256 is used to have faster computations. Recommended frame length is 512 at 8kHz.')
        warnings.warn('Overlap factor=2 is used to have faster computations. Recommended value: 4.')
        n_solver = 0
        
        exp_param['solvers'] = []
        n_solver += 1
        exp_param['solvers'].append({'name': 'consOMP-G', 'function': inpaintSignal_IndependentProcessingOfFrames, 
                                     'param': {'N': 256, 'inpaintFrame': inpaintFrame_consOMP_Gabor, 'OMPerr': 0.001, 'sparsityDegree': 256/4, 
                                               'D_fun': Gabor_Dictionary, 'OLA_frameOverlapFactor': 2, 'redundancyFactor': 2, 'wd': wRect, 
                                               'wa': wRect, 'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 'MULTITHREAD_FRAME_PROCESSING': False}})
        n_solver += 1
        exp_param['solvers'].append({'name': 'Janssen', 'function': inpaintSignal_IndependentProcessingOfFrames, 
                                     'param': {'N': 256, 'inpaintFrame': inpaintFrame_janssenInterpolation, 'OLA_frameOverlapFactor': 2, 
                                               'wa': wRect, 'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 'MULTITHREAD_FRAME_PROCESSING': False}})
        
        n_solver += 1
        exp_param['solvers'].append({'name': 'OMP-G', 'function': inpaintSignal_IndependentProcessingOfFrames, 
                                     'param': {'N': 256, 'inpaintFrame': inpaintFrame_OMP_Gabor, 'OMPerr': 0.001, 'sparsityDegree': 256/4, 
                                               'D_fun': Gabor_Dictionary, 'OLA_frameOverlapFactor': 2, 'redundancyFactor': 2, 'wd': wRect, 
                                               'wa': wRect, 'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 'MULTITHREAD_FRAME_PROCESSING': False}})

        n_solver += 1
        exp_param['solvers'].append({'name': 'OMP-C', 'function': inpaintSignal_IndependentProcessingOfFrames, 
                                     'param': {'N': 256, 'inpaintFrame': inpaintFrame_OMP, 'OMPerr': 0.001, 
                                               'sparsityDegree': 256/4, 'D_fun': DCT_Dictionary, 'OLA_frameOverlapFactor': 2, 
                                               'redundancyFactor': 2, 'wd': wRect, 'wa': wRect, 'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 
                                               'MULTITHREAD_FRAME_PROCESSING': False}})

        n_solver += 1
        exp_param['solvers'].append({'name': 'consOMP-C', 'function': inpaintSignal_IndependentProcessingOfFrames, 
                                     'param': {'N': 256, 'inpaintFrame': inpaintFrame_consOMP, 'OMPerr': 0.001, 'sparsityDegree': 256/4, 
                                               'D_fun': DCT_Dictionary, 'OLA_frameOverlapFactor': 2, 'redundancyFactor': 2, 'wd': wRect, 'wa': wRect, 
                                               'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 'MULTITHREAD_FRAME_PROCESSING': False}})




    print(f'Folder {exp_param["soundDir"]}')
    if not os.path.exists(exp_param['destDir']):
        os.mkdir(exp_param['destDir'])
    sound_files = [f for f in os.listdir(exp_param['soundDir']) if f.endswith('.wav')]
    SNRClip = np.zeros((len(sound_files), len(exp_param['missing_duration']), len(exp_param['solvers'])))

    for kf in range(len(sound_files)):
        sound_file = os.path.join(exp_param['soundDir'], sound_files[kf])
        print(f' File {sound_file}')
        # Read test signal
        fs, data = wav.read(sound_file)
        
        # Split the signal into frames
        frame_length = 512
        hop_length = frame_length // 2  # 50% overlap between frames
        frames = librosa.util.frame(data, frame_length=frame_length, hop_length=hop_length).T

        
        for kmd, missing_duration in enumerate(exp_param['missing_duration']):
            print(f'  Missing duration {missing_duration}')
            # Generate the missing intervals
            problemParameters = {'interval_duration': exp_param['interval_duration'], 'missing_duration': missing_duration}

            for n_solver in range(len(exp_param['solvers'])):
                SNRs = []
                for frame in frames:
                    problem_data, solution_data = generateMissingGroupsProblemPeriodic(frame, problemParameters)
                    print(problem_data.keys())
                    solver_param = exp_param['solvers'][n_solver]['param']
                    x_est1, x_est2 = exp_param['solvers'][n_solver]['function'](problem_data, solver_param)
                    
                    # Compute performance
                    L = len(x_est1)
                    N = solver_param['N']
                    SNR_all, SNR_miss = SNRInpaintingPerformance(solution_data['xClean'][N:L-N], problem_data['x'][N:L-N], x_est2[N:L-N], problem_data['IMiss'][N:L-N])
                    print(SNR_all, SNR_miss)
                    SNRs.append(SNR_miss[0])              
                avg_SNR = np.mean(SNRs)
                print(f'Average SNR: {avg_SNR}')
                SNRClip[kf, kmd, n_solver] = avg_SNR        

    average_SNR = np.mean(SNRClip, axis=0)

    plt.figure()
    for i in range(average_SNR.shape[0]):
        plt.plot(average_SNR[i, :])
    plt.xlabel('Solver')
    plt.ylabel