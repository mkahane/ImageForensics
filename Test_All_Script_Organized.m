%A somewhat inconclusive script that traverses a file looking at all the
%images in the file and plotting the histograms of their DCT coefficients.
%Since these are the decompressed DCT Coefficients, there are high
%frequency artifacts, but as far as I can tell they are inconclusive.  If
%the image loading and coefficient extracting could be replaced by direct
%coefficient extraction from the .jpg file, this could work as double
%compression detection.

function Test_All_Script_Organized
clear all
close all
clc


imdir = '/Users/mattkahane/Documents/MATLAB/psych221/project/training_images/*.jpg';
all_images = dir(imdir);
imfolder ='/Users/mattkahane/Documents/MATLAB/psych221/project/training_images/';

for run = 12:14
    [iut filename] = getImage(run,all_images,imfolder);

    figure
    histogram_n = 2;%Initializing Plots
    fourier_n = 1;

    for f_component = [1 2 3]
        row = f_component;
        col = f_component; 
        coeffs = extract_coeffs(iut,row,col);
        %coeffs = zeroMean(coeffs);
        [xcenters xedges nbins] = histParams(coeffs);
        hist_fun = histc(coeffs,xedges);
        hist_fun = zeroMean(hist_fun);
        FT = abs(fftshift(fft(hist_fun)));
        makePlots(histogram_n,fourier_n,xcenters,hist_fun,row,col,coeffs...
            ,FT,filename)
        histogram_n = histogram_n+2;
        fourier_n = fourier_n+2;
    end

end
end

function [iut filename] = getImage(run,all_images,imfolder)
    cur_image = all_images(run);
    filename = cur_image.name;
    path_name = strcat(imfolder,filename);
    iut = imread(path_name);
    iut = rgb2ycbcr(iut);
    iut = double(iut);
    iut = iut(:,:,1);
    sz = size(iut);
    sz = floor(sz/8)*8;
    iut = iut(1:sz(1),1:sz(2));
end

function zeroed = zeroMean(data)
    m = mean(data);
    zeroed = data-m;
end

function coeffs = extract_coeffs(iut,r,c)
    nrows = size(iut,1);
    ncols = size(iut,2);
    coeffs = zeros((nrows/8),ncols/8);
    for i = 1:nrows/8
        for j = 1:ncols/8
            topleft_r = (i-1)*8;
            topleft_c = (j-1)*8;
            Block = iut(topleft_r+1:topleft_r+8,topleft_c+1:topleft_c+8);
            %Block = Block -128;
            DBlock = dct2(Block);
            coeffs(i,j) = DBlock(r,c);
        end
    end
    coeffs = reshape(coeffs,((nrows/8)*(ncols/8)),1);
end

function [xcenters xedges nbins] = histParams(data)
    minimum = min(data);
    maximum = max(data);
    binsize = 1;
    xcenters = minimum:binsize:maximum;
    xedges = xcenters - binsize/2;
    nbins = length(xcenters);
end

function makePlots(histogram_n,fourier_n,xcenters,hist_fun,row,col,coeffs,...
    FT,filename)
    subplot(3,2,histogram_n)
    hist(coeffs,xcenters);
    hold on
    plot(xcenters,hist_fun,'r');
    if(histogram_n == 2)
        title({filename,strcat('coefficient (',num2str(row),',',num2str(col),')')}...
            ,'interpreter','none')
    else
        title(strcat('coefficient (',num2str(row),',',num2str(col),')'));
    end
    hold off
    subplot(3,2,fourier_n)
    plot(xcenters,FT)
    if(fourier_n == 1)
        title({filename,strcat('coefficient (',num2str(row),',',num2str(col),')')}...
            ,'interpreter','none')
    else
        title(strcat('coefficient (',num2str(row),',',num2str(col),')'));
    end
end

