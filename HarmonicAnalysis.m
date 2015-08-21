clear all
close all

inFile = 'C:\Users\SShivappa\Documents\ZooStuff\PikaWatch\AudioMonitoring\May26-2014-BeaconRockSP\DR0000_0005.mp3';
% inFile = 'C:\Users\SShivappa\Dropbox\Public\TASCAM_0006_1.mp3';

% read in the mp3 file
[x,fs] = mp3read(inFile);

% resample audio to 48kHz (if sampling rate is already 48kHz, nothing happens in
% this step.)
x = resample(x(1:end,1),48000,fs);
fs = 48000;

%%
% size of the FFT - determines frequency/temporal resolution
fftSize=4096;
pikaFound = [];

% for each frame of audio (selected with 50% overlap with previous frame)
for i= 1:fftSize/2:length(x)-fftSize
    
    % find the magnitude of the FFT
    F = abs(fft(x(i:i+fftSize-1)));
    
    % chop off frequencies that are irrelevant to pika detection
    F=F(fftSize/32:end/2);
    
    % convert to log scale after normalizing (now we are looking at scaled,
    % normalized energy at different frequency bins)
    F = log10(F/sum(F));
    
    % compute thresholds for de-noising based on mean energy of the frame
    meanF = mean(F);
    
    % threshold the frequencies (TF = 1 if energy in freq bin > 1+threshold and 0 otherwise)
    % noise components (weak energy bins) will be set to zero
    TF = F>(meanF+1);
    
    % smooth the thresholded frequency bin values using a approx. Gaussian
    % filter to combine nearby non-zero values into single peak
    TF = conv(TF*1.0,[0.125 0.25 0.4 0.5 0.9 1 1 1 0.9 0.5 0.4 0.25 0.125]);

    % find the peaks in the smoothed frequency domain (peaks should be
    % atleast 50 frequency bins apart
    [pks, locs] = findpeaks(TF,'minpeakdistance',50);
    % if more than 4 peaks are found in current frame (harmonic structure might exist), use them to find the
    % average (median) of the interpeak distance. Interpeak distance
    % corresponds to the fundamental harmonic frequency. If fundamental
    % harmonic is between expected range for pika, mark frame as pikaFound
    if length(locs)>=4
        interPeakDist = conv(locs,[1 -1],'same');
        harmonicFreq = median(interPeakDist(2:end-1));
        if (harmonicFreq < 75) & ( harmonicFreq > 55 )
            % i/fs is the time from the beginning of the audio file
            % to current frame
            pikaFound = [pikaFound i/fs];
        end
    end    

end

%%
%Bunching nearby pikaFound frames into unique pika detections and cropping
%the corresponding one second of audio and writing those audio segments to an output wav file 

pikaFound = pikaFound';

% settling for one second accuracy
pikaFound = floor(pikaFound);

% remove duplicates
pikaFound = unique(pikaFound)


X = [];
x = [x; zeros(fs,1)]; % zero pad to prevent out of bound errors in the loop that follows

% clip audio segments of one second duration around each unique pika call
for i=1:length(pikaFound)
    % append new audio segment to previous ones.. max value is written in between to act as a boundary sample.. makes it look nice when displayed as a spectrogram in Audacity! 
    X = [X; max(X); x(pikaFound(i)*fs+1:pikaFound(i)*fs+fs)];
end

% X = 0.05*X/max(X);

% write detected pika calls out to new .wav file
wavwrite(X,fs,strcat(inFile(1:end-4),'_pika.wav'));
    