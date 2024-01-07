import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import read

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
from Solvers.inpaintFrame_beamOMP_Gabor import inpaintFrame_beamOMP_Gabor

def declipping_experiment(exp_param=None):
    # Add paths to necessary directories
    # In Python, you can use sys.path.append(directory_path)
    # However, this is generally not recommended. It's better to properly install any packages you're using.
    
    # If no experiment parameters are provided, initialize an empty dictionary
    if exp_param is None:
        exp_param = {}
    
    # Set default values for missing parameters
    if 'clippingScale' not in exp_param:
        exp_param['clippingScale'] = np.arange(0.7, 9, 0.1)
    if 'soundDir' not in exp_param:
        exp_param['soundDir'] = 'Data/shortTest/'
        # exp_param['soundDir'] = 'Data/testSpeech8kHz_from16kHz/'
        warnings.warn('soundDir has only one sound to have faster computations. Recommended soundDir: ../../Data/testSpeech8kHz_from16kHz/')
    if 'destDir' not in exp_param:
        exp_param['destDir'] = 'tmp/declip/'
    
    # Set parameters
    if 'solvers' not in exp_param:
        warnings.warn('Frame length=256 is used to have faster computations. Recommended frame length is 512 at 8kHz.')
        warnings.warn('Overlap factor=2 is used to have faster computations. Recommended value: 4.')
        n_solver = 0
        
        exp_param['solvers'] = []
        n_solver += 1
        exp_param['solvers'].append({'name': 'consOMP_Gabor', 'function': inpaintSignal_IndependentProcessingOfFrames, 
                                     'param': {'N': 256, 'inpaintFrame': inpaintFrame_consOMP_Gabor, 'OMPerr': 0.001, 'sparsityDegree': 256/4, 
                                               'D_fun': Gabor_Dictionary, 'OLA_frameOverlapFactor': 2, 'redundancyFactor': 2, 'wd': wRect, 
                                               'wa': wRect, 'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 'MULTITHREAD_FRAME_PROCESSING': False}})
        

        n_solver += 1
        exp_param['solvers'].append({'name': 'beamOMP_Gabor', 'function': inpaintSignal_IndependentProcessingOfFrames, 
                                     'param': {'N': 256, 'inpaintFrame': inpaintFrame_beamOMP_Gabor, 'OMPerr': 0.001, 'sparsityDegree': 256/4, 
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

        # n_solver += 1
        # exp_param['solvers'].append({'name': 'OMP-C', 'function': inpaintSignal_IndependentProcessingOfFrames, 
        #                              'param': {'N': 256, 'inpaintFrame': inpaintFrame_OMP, 'OMPerr': 0.001, 
        #                                        'sparsityDegree': 256/4, 'D_fun': DCT_Dictionary, 'OLA_frameOverlapFactor': 2, 
        #                                        'redundancyFactor': 2, 'wd': wRect, 'wa': wRect, 'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 
        #                                        'MULTITHREAD_FRAME_PROCESSING': False}})

        # n_solver += 1
        # exp_param['solvers'].append({'name': 'consOMP-C', 'function': inpaintSignal_IndependentProcessingOfFrames, 
        #                              'param': {'N': 256, 'inpaintFrame': inpaintFrame_consOMP, 'OMPerr': 0.001, 'sparsityDegree': 256/4, 
        #                                        'D_fun': DCT_Dictionary, 'OLA_frameOverlapFactor': 2, 'redundancyFactor': 2, 'wd': wRect, 'wa': wRect, 
        #                                        'OLA_ws': wSine, 'SKIP_CLEAN_FRAMES': True, 'MULTITHREAD_FRAME_PROCESSING': False}})




    print(f'Folder {exp_param["soundDir"]}')
    if not os.path.exists(exp_param['destDir']):
        os.mkdir(exp_param['destDir'])
    sound_files = [f for f in os.listdir(exp_param['soundDir']) if f.endswith('.wav')]
    SNRClip = np.zeros((len(sound_files), len(exp_param['clippingScale']), len(exp_param['solvers'])))

    for kf in range(len(sound_files)):
        sound_file = os.path.join(exp_param['soundDir'], sound_files[kf])
        print(f' File {sound_file}')
        # Read test signal
        fs, x = read(sound_file)
        
        for kClip in range(len(exp_param['clippingScale'])):
            clipping_level = exp_param['clippingScale'][kClip]
            print(f'  Clip level {clipping_level}')
            
            # Generate the problem
            problem_data, solution_data = generateDeclippingProblem(x, clipping_level)
            
            for n_solver in range(len(exp_param['solvers'])):
                # Declip with solver
                solver_param = exp_param['solvers'][n_solver]['param']
                x_est1, x_est2 = exp_param['solvers'][n_solver]['function'](problem_data, solver_param)
                
                # Compute performance
                L = len(x_est1)
                N = solver_param['N']
                SNR_all, SNR_miss = SNRInpaintingPerformance(solution_data['xClean'][N:L-N], problem_data['x'][N:L-N], x_est2[N:L-N], problem_data['IMiss'][N:L-N])
                print(SNR_all, SNR_miss)
                SNRClip[kf, kClip, n_solver] = SNR_miss[1]
            
                # Normalize and save both the reference and the estimates!
                norm_x = 1.1 * np.max(np.abs(np.concatenate([x_est1, x_est2, solution_data['xClean']])))
                
                L = min([len(x_est2), len(x_est1), len(solution_data['xClean']), len(problem_data['x'])])
                x_est1 = x_est1[:L] / norm_x
                x_est2 = x_est2[:L] / norm_x
                x_clipped = problem_data['x'][:L] / norm_x
                x_clean = solution_data['xClean'][:L] / norm_x
                wavfile.write(f'{exp_param["destDir"]}{sound_files[kf][:-4]}Est1{clipping_level}.wav', fs, x_est1)
                wavfile.write(f'{exp_param["destDir"]}{sound_files[kf][:-4]}Est2{clipping_level}.wav', fs, x_est2)
                wavfile.write(f'{exp_param["destDir"]}{sound_files[kf][:-4]}Clipped{clipping_level}.wav', fs, x_clipped)
                wavfile.write(f'{exp_param["destDir"]}{sound_files[kf][:-4]}Ref{clipping_level}.wav', fs, x_clean)
                
                print('\n')
                # clear a x_est1 x_est2 x_clipped x_clean I_clipped
                # save(f'{exp_param["destDir"]}clipping_exp.mat')
    average_SNR = np.mean(SNRClip, axis=0)

    plt.figure()
    for i in range(average_SNR.shape[1]):
        plt.plot(average_SNR[:, i])
    plt.xlabel('Clipping level')
    plt.ylabel('SNR')
    plt.show()
    
    return SNRClip, average_SNR